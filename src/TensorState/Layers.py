import abc
import contextlib
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import zarr

import TensorState
import TensorState.States as ts  # noqa: N813 -- deliberate package alias

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

    Layer states are stored in a zarr array, which permits compressed storage of
    data in memory or on disk. Only blosc compression is used to ensure fast
    compression/decompression speeds. By default, data is stored in memory, but
    data can be stored on disk to reduce memory consumption by using the
    disk_path keyword.

    NOTE: This layer currently works with PyTorch ``nn.Module`` and Lightning
    ``LightningModule`` instances.

    """

    capture_on: bool = True
    memory_device: str | int
    raise_on_capture_error: bool = False
    _chunk_size: int = 0
    _state_shape: tuple
    _entropy: float | None = None
    _state_count: int = 0
    _capture_error: BaseException | None = None

    @property
    def state_count(self):
        """Total number of observed states, including repeats."""
        self._executor.submit(self._collect_cache())
        self._wait_for_threads()
        return self._state_count

    @state_count.setter
    def state_count(self, value):
        raise AttributeError("state_count attribute is read-only.")

    @property
    def states(self):
        """Decompressed state data."""
        self._executor.submit(self._collect_cache())
        self._wait_for_threads()
        if not isinstance(self._states, np.ndarray):
            self._states = ts.decompress_states(
                self.raw_states[self._index[self._edges[:-1]], :],
                int(self.max_entropy()),
            )

        return self._states

    @states.setter
    def states(self, value):
        raise AttributeError("states attribute is read-only.")

    @property
    def raw_states(self):
        """Raw state data as stored in memory, bit compressed."""
        self._wait_for_threads()
        return self._raw_states.oindex

    @raw_states.setter
    def raw_states(self, value):
        raise AttributeError("states attribute is read-only.")

    _states = None
    _raw_states = None
    _index = None
    _edges = None
    _counts = None
    _input_shape = None
    _state_ids = None
    _threads: ClassVar[list[Future]] = []
    _zarr_path = None
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
            disk_path: Path on disk to save captured states in zarr format.
                Defaults to None.
            memory_device: ``"cpu"``, ``"gpu"``, or a CUDA device index. When
                set to ``"gpu"`` (or an int) and a CUDA-capable PyTorch is
                available, captured states are accumulated in a GPU-resident
                cache before being flushed to main memory in batches. When
                ``None`` (the default), resolves to ``"gpu"`` if CUDA is
                available, otherwise ``"cpu"``.
            gpu_buffer_size: Size of the GPU-resident compressed-state buffer,
                in megabytes. Larger buffers mean fewer device-to-host
                transfers at the cost of more VRAM. Independent of the zarr
                on-disk chunk size. Defaults to 256 MB. Ignored when
                ``memory_device`` resolves to ``"cpu"``.
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

        # Set up zarr, but don't create anything
        if disk_path is not None:
            if not isinstance(disk_path, Path):
                self._zarr_path = Path(disk_path)
            else:
                self._zarr_path = disk_path
            self._zarr_path = self._zarr_path.joinpath("tensor_states")
            self._zarr_path.mkdir(exist_ok=True)
            self._zarr_path = self._zarr_path.joinpath(name + ".zarr")
            self._zarr_path.mkdir(exist_ok=False)

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

    def _compress_and_store(self, inputs):
        # Calculate the number of states to process
        num_states = int(np.prod(inputs.shape[0:-1]))

        # Reshape using the appropriate library
        if isinstance(inputs, torch.Tensor):
            logger.debug("_compress_and_store: torch.reshape")
            states = inputs.reshape(-1, int(inputs.shape[-1]))
        else:
            logger.debug("_compress_and_store: numpy.reshape")
            states = np.reshape(inputs, (-1, int(inputs.shape[-1])))

        # Compress states. Output backend matches input: numpy for numpy,
        # torch.Tensor for torch input.
        states = ts.compress_states(states)

        # GPU cache path: accumulate compressed bytes on GPU before flushing
        # to main memory in chunks.
        if self.memory_device != "cpu" and isinstance(states, torch.Tensor):
            if self._state_cache_index + states.shape[0] > self._state_cache.shape[0]:
                logger.debug(
                    "_compress_and_store: GPU cache full, collecting and "
                    "sending to main memory."
                )
                if self._state_cache_index > 0:
                    states = torch.vstack(
                        (self._state_cache[: self._state_cache_index], states)
                    )
                num_states += self._state_cache_index
                self._state_cache_index = 0
            else:
                logger.debug("_compress_and_store: Caching states on GPU.")
                self._state_cache[
                    self._state_cache_index : self._state_cache_index + states.shape[0]
                ] = states
                self._state_cache_index += states.shape[0]
                return True

        # Move any torch tensor to numpy before zarr write.
        if isinstance(states, torch.Tensor):
            logger.debug("_compress_and_store: torch -> numpy")
            states = states.cpu().numpy()

        # Resize the zarr array if needed
        if 2 * num_states + self._state_count >= self._raw_states.shape[0]:
            self._state_shape[0] += max(self._chunk_size[0], 2 * num_states)
            self._raw_states.resize(self._state_shape)

        # Store numpy array
        logger.debug("_compress_and_store: zarr storage")
        self._raw_states[self._state_count : self._state_count + num_states] = states
        self._state_count += num_states

        # Reset the _counts and _state_ids so they are recalculated
        logger.debug("_compress_and_store: reset bins")
        self._counts = None
        self._state_ids = None
        self._states = None

        return True

    def _collect_cache(self):
        logger.debug("_collect_cache: torch -> numpy")
        if self._state_cache_index == 0 or self._state_cache is None:
            return True

        num_states = self._state_cache_index
        states = self._state_cache[:num_states].cpu().numpy()

        # Resize the zarr array if needed
        if 2 * num_states + self._state_count >= self._raw_states.shape[0]:
            self._state_shape[0] += self._chunk_size[0]
            self._raw_states.resize(self._state_shape)

        # Store numpy array
        logger.debug("_collect_cache: zarr storage")
        self._raw_states[self._state_count : self._state_count + num_states] = states
        self._state_count += num_states

        # Reset the _counts and _state_ids so they are recalculated
        logger.debug("_collect_cache: reset bins")
        self._counts = None
        self._state_ids = None
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
        """Initialize the state space.

        This method initializes the layer and resets any previously held data.
        The zarr array is initialized in this method.

        Args:
            input_shape (TensorShape,tuple, list): Shape of the input.
        """
        if input_shape is not None:
            self._input_shape = input_shape

        if self._input_shape is None:
            raise ValueError(
                "The input_shape is None, and no previous input shape "
                "information was provided. The first time reset_states is "
                "called, an input_shape must be provided."
            )

        # Try to keep chunks limited to 16MB
        ncols = int(np.ceil(self._input_shape[self._channel_index] / 8))
        nrows = 2**22 // ncols

        # Initialize internal variables related to state space
        self._state_ids = None
        self._edges = None
        self._index = None
        self._counts = None
        self._entropy = None
        self._threads = []
        self._chunk_size = (nrows, ncols)
        self._state_shape = list(self._chunk_size)
        self._state_count = 0
        self._capture_error = None

        if self._raw_states is not None:
            # Zero out states and resize if zarr already open
            self._raw_states.resize(self._state_shape)
            self._raw_states[:] = 0
        else:
            # Initialize the zarr array
            if self._zarr_path is not None:
                if self._zarr_path.is_file():
                    self._zarr_path.unlink()

                self._raw_states = zarr.zeros(
                    shape=self._state_shape,
                    chunks=self._chunk_size,
                    dtype="B",
                    synchronizer=zarr.ThreadSynchronizer(),
                    store=str(self._zarr_path.absolute()),
                )
            else:
                self._raw_states = zarr.zeros(
                    shape=self._state_shape,
                    chunks=self._chunk_size,
                    dtype="B",
                    synchronizer=zarr.ThreadSynchronizer(),
                )

        if (
            self.memory_device != "cpu"
            and input_shape is not None
            and input_shape[0] < self._chunk_size[0]
        ):
            if self._state_cache is None:
                device_idx = (
                    self.memory_device if isinstance(self.memory_device, int) else 0
                )
                # GPU buffer rows from the MB budget, floored at one zarr
                # chunk so we never buffer less than a chunk between flushes.
                buffer_rows = max(
                    self._chunk_size[0],
                    int(self._gpu_buffer_mb * 2**20 // ncols),
                )
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
        if not isinstance(self._state_ids, list):
            self.counts()
            self._state_ids = []
            states = self.raw_states[self._index[self._edges[:-1]], :].tobytes()
            delta = int((self.max_entropy() - 1) // 8 + 1)
            for cindex in range(0, delta * (self._edges.shape[0] - 1), delta):
                self._state_ids.append(states[cindex : cindex + delta])

        return self._state_ids

    def counts(self, index: int | list[int] | np.ndarray | None = None) -> list[int]:
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
        if not isinstance(self._counts, np.ndarray) or index is not None:
            if index is None:
                rows = slice(0, self.state_count)
                count = self.state_count
            else:
                rows = self._instance_indices(index)  # type: ignore
                count = rows.size  # type: ignore

            # Create the index and sort the data to find the bin edges
            _edges, _index = ts.sort_states(self.raw_states[rows, :], count)
            _counts = np.diff(_edges)

            if index is None:
                self._edges, self._index = _edges, _index
                self._counts = _counts

            return _counts

        return self._counts

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

    def _capture(self, *args):
        """Forward-hook callable registered on the watched module.

        Matches both the forward-hook ``(module, inputs, output)`` and
        forward-pre-hook ``(module, inputs)`` signatures by operating on
        the last positional argument (output for post-hooks, inputs for
        pre-hooks), preserving the prior bare-callable behavior.
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
