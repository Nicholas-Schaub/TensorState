import abc
import contextlib
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import ClassVar

import duckdb
import numpy as np
import pyarrow as pa
import torch

import TensorState
import TensorState.States as ts  # noqa: N813 -- deliberate package alias

# Stage this many rows in memory before flushing a batch to DuckDB. DuckDB is
# slow at many tiny appends, so capture buffers and inserts in Arrow batches.
_STORE_FLUSH_ROWS = 1 << 20


class _StateStore:
    """DuckDB-backed store for bit-packed microstates.

    Each observed microstate is a row of ``ncols`` packed bytes, stored as a
    single ``BLOB``. Counting unique microstates — the core analytical
    operation — is a ``GROUP BY``, which DuckDB runs in parallel and out of
    core with automatic spill-to-disk. Rows are staged in memory and inserted
    in Arrow batches; all mutating operations are guarded by a lock because
    capture runs on a small thread pool and a DuckDB connection is not safe
    for concurrent use.
    """

    def __init__(
        self,
        ncols: int,
        *,
        path: str | None = None,
        memory_limit: str | None = None,
        flush_rows: int = _STORE_FLUSH_ROWS,
    ) -> None:
        self._ncols = int(ncols)
        self._flush_rows = int(flush_rows)
        self._lock = threading.Lock()
        self._con = duckdb.connect(path if path is not None else ":memory:")
        if memory_limit is not None:
            self._con.execute(f"SET memory_limit='{memory_limit}'")
        # CREATE OR REPLACE so reopening an on-disk database resets cleanly.
        self._con.execute("CREATE OR REPLACE TABLE states (s BLOB)")
        self._buffer: list[np.ndarray] = []
        self._buffered = 0
        self._count = 0
        self._grouped: tuple[list[bytes], np.ndarray] | None = None

    def append(self, rows: np.ndarray) -> None:
        """Stage ``(k, ncols)`` uint8 rows; flush to DuckDB past the threshold."""
        rows = np.ascontiguousarray(rows, dtype=np.uint8)
        if rows.ndim != 2 or rows.shape[1] != self._ncols:
            raise ValueError(
                f"expected (k, {self._ncols}) uint8 rows, got shape {rows.shape}"
            )
        if rows.shape[0] == 0:
            return
        with self._lock:
            self._buffer.append(rows)
            self._buffered += rows.shape[0]
            self._count += rows.shape[0]
            self._grouped = None
            if self._buffered >= self._flush_rows:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        arr = self._buffer[0] if len(self._buffer) == 1 else np.vstack(self._buffer)
        n = arr.shape[0]
        # Pack the contiguous (n, ncols) bytes as one fixed-size-binary buffer;
        # DuckDB maps Arrow fixed_size_binary to BLOB.
        fsb = pa.FixedSizeBinaryArray.from_buffers(
            pa.binary(self._ncols), n, [None, pa.py_buffer(arr.tobytes())]
        )
        tbl = pa.table({"s": fsb})  # noqa: F841 -- referenced by replacement scan
        self._con.execute("INSERT INTO states SELECT s FROM tbl")
        self._buffer = []
        self._buffered = 0

    def count(self) -> int:
        return self._count

    def grouped(self) -> tuple[list[bytes], np.ndarray]:
        """Return ``(unique_state_blobs, counts)`` via a single GROUP BY."""
        with self._lock:
            if self._grouped is None:
                self._flush_locked()
                rows = self._con.execute(
                    "SELECT s, COUNT(*) FROM states GROUP BY s"
                ).fetchall()
                blobs = [bytes(r[0]) for r in rows]
                counts = np.array([r[1] for r in rows], dtype=np.int64)
                self._grouped = (blobs, counts)
            return self._grouped

    def raw_rows(self) -> np.ndarray:
        """Materialize all stored states as a ``(n, ncols)`` uint8 array.

        Insertion order is preserved. Used by the per-instance ``counts``
        path; loads everything into memory, so it is not the out-of-core path.
        """
        with self._lock:
            self._flush_locked()
            rows = self._con.execute("SELECT s FROM states").fetchall()
        if not rows:
            return np.empty((0, self._ncols), dtype=np.uint8)
        return np.stack([np.frombuffer(r[0], dtype=np.uint8) for r in rows])

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._con.close()


logging.basicConfig(
    format="%(asctime)s - %(name)-10s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("TensorState.Layers")


class AbstractStateCapture(abc.ABC):
    """Base class for capturing state space information in a neural network.

    This class implements the infrastructure used to capture, quantize, and
    process state space information. A PyTorch subclass implements these
    methods as forward hooks.

    This class captures state information and quantizes layer outputs as firing
    or not firing based on whether the values are >0 or <=0 respectively.
    Although this layer is intended to be attached before or after a neural
    layer, this can actually be attached to any layer type. After recording the
    firing state of all neurons, the original input is returned unaltered. Thus,
    this layer can be thought of as a "probe", since it does not add or subtract
    from the function of a network.

    Layer states are stored in a DuckDB-backed store, one bit-packed
    microstate per row. Counting unique microstates is a ``GROUP BY``, which
    DuckDB runs in parallel and out of core with automatic spill-to-disk. By
    default the store is in memory; pass the ``disk_path`` keyword to back it
    with an on-disk database instead.

    NOTE: This layer currently works with PyTorch ``nn.Module`` and Lightning
    ``LightningModule`` instances.

    """

    capture_on: bool = True
    memory_device: str | int
    raise_on_capture_error: bool = False
    _entropy: float | None = None
    _capture_error: BaseException | None = None

    @property
    def state_count(self):
        """Total number of observed states, including repeats."""
        self._drain()
        return self._store.count()

    @state_count.setter
    def state_count(self, value):
        raise AttributeError("state_count attribute is read-only.")

    @property
    def states(self):
        """Decompressed unique states as a boolean array."""
        self._drain()
        if not isinstance(self._states, np.ndarray):
            blobs, _ = self._store.grouped()
            if blobs:
                packed = np.stack([np.frombuffer(b, dtype=np.uint8) for b in blobs])
            else:
                packed = np.empty((0, self._store._ncols), dtype=np.uint8)
            self._states = ts.decompress_states(packed, int(self.max_entropy()))

        return self._states

    @states.setter
    def states(self, value):
        raise AttributeError("states attribute is read-only.")

    @property
    def raw_states(self):
        """Raw bit-packed states as a ``(n, ncols)`` uint8 array (insertion order)."""
        self._drain()
        return self._store.raw_rows()

    @raw_states.setter
    def raw_states(self, value):
        raise AttributeError("states attribute is read-only.")

    _states = None
    _store: _StateStore | None = None
    _db_path: str | None = None
    _memory_limit: str | None = None
    _input_shape = None
    _state_ids = None
    _threads: ClassVar[list[Future]] = []
    _channel_index = -1
    _state_cache_index = 0
    _gpu_buffer_mb: float = 256.0

    def __init__(
        self,
        name,
        disk_path=None,
        memory_device: str | int | None = None,
        gpu_buffer_size: float = 256.0,
        *,
        raise_on_capture_error: bool = False,
        **kwargs,
    ):
        """Abstract State Capture Layer.

        Args:
            name: Name of the state capture layer.
            disk_path: Directory under which to back the state store with an
                on-disk DuckDB database. If None (the default), the store is
                held in memory.
            memory_device: ``"cpu"``, ``"gpu"``, or a CUDA device index. When
                set to ``"gpu"`` (or an int) and a CUDA-capable PyTorch is
                available, captured states are accumulated in a GPU-resident
                cache before being flushed to main memory in batches. When
                ``None`` (the default), resolves to ``"gpu"`` if CUDA is
                available, otherwise ``"cpu"``.
            gpu_buffer_size: Size of the GPU-resident compressed-state buffer,
                in megabytes. Larger buffers mean fewer device-to-host
                transfers at the cost of more VRAM. Defaults to 256 MB.
                Ignored when ``memory_device`` resolves to ``"cpu"``.
            raise_on_capture_error: When ``True``, a capture failure aborts the
                next capture call by re-raising the stored error (fail-fast).
                When ``False`` (the default), capture failures are logged when
                they occur and re-raised only when results are read
                (``state_count`` / ``states`` / ``counts`` / ``entropy``), so a
                single bad batch never aborts a long training run.
            **kwargs: Keyword arguments. Used for passing arguments to other
                classes that inherit from AbstractStateCapture.
        """
        self._executor = ThreadPoolExecutor(4)
        self._capture_error = None
        self._capture_error_lock = threading.Lock()
        self.raise_on_capture_error = raise_on_capture_error

        # Assign a name to the layer. Some inheriting classes make name
        # protected, so catch the error just in case.
        with contextlib.suppress(AttributeError):
            self.name = name

        # Resolve the on-disk DuckDB database path (None -> in-memory). The
        # store itself is created in reset_states once the neuron count (and
        # thus the packed-byte width) is known.
        self._store = None
        self._memory_limit = None
        if disk_path is not None:
            base = disk_path if isinstance(disk_path, Path) else Path(disk_path)
            base = base / "tensor_states"
            base.mkdir(parents=True, exist_ok=True)
            self._db_path = str((base / (name + ".duckdb")).absolute())
        else:
            self._db_path = None

        # Default the memory device based on CUDA availability when the caller
        # did not specify one.
        if memory_device is None:
            memory_device = "gpu" if torch.cuda.is_available() else "cpu"
            logger.info(
                "StateCaptureHook: memory_device not specified; defaulting to "
                "'%s' (torch.cuda.is_available()=%s)",
                memory_device,
                torch.cuda.is_available(),
            )

        if memory_device == "gpu" and not torch.cuda.is_available():
            logger.warning(
                "Memory device set to gpu, but torch.cuda is not available. "
                "Changing memory device to cpu."
            )
            self.memory_device = "cpu"
        else:
            self.memory_device = memory_device

        self._gpu_buffer_mb = gpu_buffer_size
        logger.debug(
            "StateCaptureHook: memory_device=%s, gpu_buffer_size=%.1f MB",
            self.memory_device,
            self._gpu_buffer_mb,
        )

    def _record_capture_error(self, exc: BaseException) -> None:
        """Store the first capture failure (sticky) and log it once.

        Logged by whichever path records it first — the done-callback (so the
        failure surfaces during the forward pass even if results are never
        read) or the read path (so a read is guaranteed to have logged before
        it re-raises). Only the first failure logs, to avoid flooding the log
        when every batch fails for the same reason.
        """
        with self._capture_error_lock:
            if self._capture_error is not None:
                return
            self._capture_error = exc
        logger.error(
            "State capture failed in probe %r: %s",
            getattr(self, "name", "<unnamed>"),
            exc,
            exc_info=exc,
        )

    def _on_capture_done(self, future: Future) -> None:
        """Done-callback: surface a capture-thread failure when it happens.

        Runs as soon as the capture thread finishes, so the failure lands in
        the logs during the forward pass rather than only when results are
        later read (or never, if they aren't).
        """
        exc = future.exception()
        if exc is not None:
            self._record_capture_error(exc)

    def _raise_capture_error(self) -> None:
        if self._capture_error is not None:
            raise RuntimeError(
                "State capture failed in probe "
                f"{getattr(self, 'name', '<unnamed>')!r}; the original error is "
                "chained below. Captured state is incomplete."
            ) from self._capture_error

    def _wait_for_threads(self):
        wait(self._threads)
        for thread in self._threads:
            # thread.exception() blocks until done and returns the exception
            # (or None) without re-raising; the done-callback already logged it.
            exc = thread.exception()
            if exc is not None:
                self._record_capture_error(exc)
        self._threads = []
        self._raise_capture_error()

    def _drain(self):
        """Join capture threads and flush the GPU cache before a read.

        Read paths call this so a subsequent store query reflects every
        captured state. Safe on the main thread: the workers are joined first,
        so the GPU-cache flush runs single-threaded.
        """
        self._wait_for_threads()
        self._collect_cache()

    def _compress_and_store(self, inputs):
        # Reshape to (num_states, num_neurons) using the matching library.
        if isinstance(inputs, torch.Tensor):
            states = inputs.reshape(-1, int(inputs.shape[-1]))
        else:
            states = np.reshape(inputs, (-1, int(inputs.shape[-1])))

        # Compress states. Output backend matches input: numpy for numpy,
        # torch.Tensor for torch input.
        states = ts.compress_states(states)

        # GPU cache path: accumulate compressed bytes on GPU before flushing
        # to main memory in chunks.
        if self.memory_device != "cpu" and isinstance(states, torch.Tensor):
            if self._state_cache_index + states.shape[0] > self._state_cache.shape[0]:
                if self._state_cache_index > 0:
                    states = torch.vstack(
                        (self._state_cache[: self._state_cache_index], states)
                    )
                self._state_cache_index = 0
            else:
                self._state_cache[
                    self._state_cache_index : self._state_cache_index + states.shape[0]
                ] = states
                self._state_cache_index += states.shape[0]
                return True

        # Move any torch tensor to numpy before the store write.
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()

        self._store.append(states)
        self._states = None  # invalidate the decompressed-states cache
        return True

    def _collect_cache(self):
        """Flush any GPU-resident cached states to the store."""
        if self._state_cache_index == 0 or self._state_cache is None:
            return True
        states = self._state_cache[: self._state_cache_index].cpu().numpy()
        self._state_cache_index = 0
        self._store.append(states)
        self._states = None
        return True

    def states_per_instance(self):
        """Average number of states per input instance."""
        # TODO: Verify this is correct for PyTorch
        return np.prod(self._input_shape[1:]) / self._input_shape[self._channel_index]

    def _instance_indices(self, index: int | list[int] | np.ndarray) -> np.ndarray:
        # Convert to numpy array if needed
        if not isinstance(index, np.ndarray):
            if isinstance(index, int):
                index = [index]
            index = np.asarray(index).squeeze()

        # Make sure the array is one dimensional
        index = index.squeeze()
        if index.ndim > 2:
            raise ValueError("index must be a 1-dimensional numpy array.")

        # If numpy array indices are boolean, convert them to indices
        if index.dtype == np.bool_:
            index = np.argwhere(index)

        # Convert instance indices to state indices
        states_per_instance = self.states_per_instance()
        state_offsets = np.arange(states_per_instance, dtype=int).reshape(1, -1)
        state_indices = states_per_instance * index.reshape(-1, 1) + state_offsets

        return state_indices.flatten()

    def reset_states(self, input_shape=None):
        """Initialize the state space, resetting any previously held data.

        Creates (or recreates) the DuckDB-backed state store. The first call
        must provide ``input_shape``.

        Args:
            input_shape (TensorShape, tuple, list): Shape of the input.
        """
        if input_shape is not None:
            self._input_shape = input_shape

        if self._input_shape is None:
            raise ValueError(
                "The input_shape is None, and no previous input shape "
                "information was provided. The first time reset_states is "
                "called, an input_shape must be provided."
            )

        # Packed-byte width: one bit per neuron, 8 neurons per byte.
        ncols = int(np.ceil(self._input_shape[self._channel_index] / 8))

        self._state_ids = None
        self._states = None
        self._entropy = None
        self._threads = []
        self._capture_error = None

        # (Re)create the state store.
        if self._store is not None:
            self._store.close()
        self._store = _StateStore(
            ncols, path=self._db_path, memory_limit=self._memory_limit
        )

        # GPU cache: hold compressed bytes on device before flushing to the
        # store in batches. Sized from the MB budget, floored so we never
        # buffer less than a reasonable batch between flushes.
        chunk_rows = 2**22 // ncols
        if (
            self.memory_device != "cpu"
            and input_shape is not None
            and input_shape[0] < chunk_rows
        ):
            if self._state_cache is None:
                device_idx = (
                    self.memory_device if isinstance(self.memory_device, int) else 0
                )
                buffer_rows = max(chunk_rows, int(self._gpu_buffer_mb * 2**20 // ncols))
                self._state_cache = torch.zeros(
                    (buffer_rows, ncols),
                    dtype=torch.uint8,
                    device=f"cuda:{device_idx}",
                )
            self._state_cache_index = 0

    def state_ids(self):
        r"""Identity of observed states.

        This method returns a list of byte arrays. Each byte array corresponds
        to a unique observed state, where each bit in the byte array corresponds
        to a neuron. The list returned by this method matches the list returned
        by ``counts``, so that the value in ``state_ids`` at position i is
        associated with the ``counts`` value at position i.

        For example, if the StateCapture layer is attached to a convolutional
        layer with 8 neurons, then each item in the list will be a byte array of
        length 1. If one of the bytes is ``\\x00`` (a null byte), then the state
        has no firing neurons.

        NOTE: Only observed states are contained in the list.

        Returns:
            Unique states observed by the layer
        """
        self._drain()
        # Aligned with counts(): both read the same cached GROUP BY result.
        return list(self._store.grouped()[0])

    def counts(self, index: int | list[int] | np.ndarray | None = None) -> np.ndarray:
        """Layer state counts.

        This method returns a numpy.array of integers, where each integer is the
        number of times a state is observed. The identity of the states can be
        obtained by calling the ``state_ids`` method.

        NOTE: The list only contains counts for observed states, so all values
        will be >0

        Args:
            index: Indices of instances to retrieve state counts for. If
                ``None``, then all counts are returned. Defaults to ``None``.

        Returns:
            Counts of stat occurrences
        """
        self._drain()
        if index is None:
            # Out-of-core unique-microstate count via DuckDB GROUP BY.
            return self._store.grouped()[1]

        # Per-instance counts: materialize the instance's rows and count
        # unique microstates among them with numpy.
        rows = self._instance_indices(index)  # type: ignore
        subset = self._store.raw_rows()[rows, :]
        _, counts = np.unique(subset, axis=0, return_counts=True)
        return counts

    def max_entropy(self):
        """Theoretical maximum entropy for the layer.

        The maximum entropy for the layer is equal to the number of neurons in
        the layer. This is different than the maximum entropy value that would
        be returned from the ``TensorState.entropy`` method with ``alpha=0``,
        which is a count of the observed states.

        Returns:
            [float]: Theoretical maximum entropy value
        """
        return float(self._input_shape[self._channel_index])

    def entropy(self, alpha=1):
        """Calculate the entropy of the layer.

        Calculate the entropy from the observed states. The alpha value is the
        order of entropy calculated using the formula for Renyi entropy. When
        alpha=1, this returns Shannon's entropy.

        Args:
            alpha (int, None): Order of entropy to calculate. If ``None``, then
                use ``max_entropy()``

        Returns:
            [float]: The entropy of the layer
        """
        if alpha is None:
            return self.max_entropy()
        return TensorState.entropy(self.counts(), alpha)

    def efficiency(self, alpha1=1, alpha2=None):
        """Calculate the efficiency of the layer.

        This method returns the efficiency of the layer. Originally, the
        efficiency was defined as the ratio of Shannon's entropy to the
        theoretical maximum entropy based on the number of neurons in the layer.
        This method with no inputs will return that value. However, this method
        will also now permit defining the alpha value for the Renyi entropy, so
        that the efficiency will be calculated as the Renyi entropy of order
        alpha1 divided by the maximum theoretical entropy.

        Args:
            alpha1 ([float, int], optional): Order of Renyi entropy in numerator
            alpha2 ([float, int, None], optional): Order of Renyi entropy in
                denominator

        Returns:
            [float]: The efficiency of the layer
        """
        assert isinstance(alpha1, (float, int)), "alpha1 must be a float or int"
        assert isinstance(alpha2, (float, int, None.__class__)), (
            "alpha2 must be a float, int, or None"
        )
        if alpha2 is not None:
            assert alpha1 > alpha2, "alpha1 must be larger than alpha 2"

        return self.entropy(alpha1) / self.entropy(alpha2)


class Probe(torch.nn.Module):
    """Marker base for observational capture submodules.

    A Probe is an ``nn.Module`` so its buffers travel with ``.to()`` /
    ``.cuda()`` and it is discoverable via ``model.named_modules()``
    (filter with ``isinstance(m, Probe)``). Probes carry no parameters
    and are NOT meant to be called directly — their capture logic runs
    as a forward (or forward-pre) hook on the watched module, and the
    probe object itself is owned by a top-level ``_tensorstate_probes``
    container that no forward pass iterates.
    """

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            f"{type(self).__name__} is an observational probe and is not "
            "meant to be called directly; its capture runs as a forward "
            "hook on the watched module."
        )


class StateCaptureHook(AbstractStateCapture, Probe):
    """State-capture probe for PyTorch.

    Implements all of :class:`AbstractStateCapture` and is an
    :class:`Probe` (``nn.Module``). The capture logic in :meth:`_capture`
    is registered as a pre- or post-forward hook on the watched module;
    the probe object is owned by the model's ``_tensorstate_probes``
    container so its buffers travel with the model without sitting in any
    module's forward path.
    """

    def __init__(
        self,
        name,
        disk_path=None,
        memory_device=None,
        *,
        raise_on_capture_error=False,
        **kwargs,
    ):
        # nn.Module.__init__ must run before any buffer/submodule
        # assignment, so initialize the Probe (nn.Module) side first.
        Probe.__init__(self)
        AbstractStateCapture.__init__(
            self,
            name,
            disk_path,
            memory_device=memory_device,
            raise_on_capture_error=raise_on_capture_error,
            **kwargs,
        )

        # Transient GPU cache as a non-persistent buffer: it travels with
        # the module under .to()/.cuda() but never lands in state_dict
        # (observational state is tied to data, not weights).
        self.register_buffer("_state_cache", None, persistent=False)

        self._channel_index = 1

    def _thread(self, tensor: torch.Tensor):
        if tensor.device.type == "cuda" and self.memory_device != "cpu":
            # Keep on GPU; compress_states handles torch tensors natively.
            pass
        else:
            tensor = (tensor > 0).cpu().numpy()
        self._compress_and_store(tensor)

    @torch._dynamo.disable
    def _capture(self, *args):
        """Forward-hook callable registered on the watched module.

        Matches both the forward-hook ``(module, inputs, output)`` and
        forward-pre-hook ``(module, inputs)`` signatures by operating on
        the last positional argument (output for post-hooks, inputs for
        pre-hooks), preserving the prior bare-callable behavior.

        Marked ``torch._dynamo.disable`` so that under ``torch.compile`` the
        hook triggers a clean graph break and runs eagerly. Without this,
        TorchDynamo tries to trace the host-side bit-pack / store path
        (numpy / numcodecs), which it cannot, and silently captures nothing
        (see benches/compile_graph_breaks.py / AIQ-22).
        """
        # Fail-fast: if a previous capture failed and the caller opted in,
        # abort this capture by re-raising the stored error.
        if self.raise_on_capture_error:
            self._raise_capture_error()

        if self._input_shape is None:
            self.reset_states(tuple(args[-1].shape))

        if not self.capture_on:
            return

        # Transform the tensor to channels-last memory layout (NHWC)
        dim_order = (0, *(i for i in range(2, args[-1].ndim)), 1)
        inputs = args[-1].detach().permute(*dim_order).contiguous()

        # Store the data using a thread. The done-callback surfaces any
        # failure in the logs as soon as the thread finishes.
        future = self._executor.submit(self._thread, inputs)
        future.add_done_callback(self._on_capture_done)
        self._threads.append(future)
