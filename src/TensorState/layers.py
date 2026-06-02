"""State-capture probes.

A :class:`StateCaptureHook` is registered as a forward (or forward-pre)
hook on a watched module. The hot path: detach -> permute to
channels-last -> bit-pack -> per-batch dedup -> append to the configured
backend store. The store contract lives in :mod:`TensorState.stores`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

import TensorState
import TensorState.states as ts
from TensorState.stores import GPUMemoryStore, HostMemoryStore, _StateStore

if TYPE_CHECKING:
    import pyarrow as pa

logging.basicConfig(
    format="%(asctime)s - %(name)-10s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("TensorState.layers")

_BackendStore = GPUMemoryStore | HostMemoryStore | _StateStore


class Probe(torch.nn.Module):
    """Marker base for observational capture submodules.

    A Probe is an ``nn.Module`` so it is discoverable via
    ``model.named_modules()`` (filter with ``isinstance(m, Probe)``). It
    carries no parameters / buffers — the backing store is a plain
    attribute so checkpoints stay clean.
    """

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            f"{type(self).__name__} is an observational probe and is not "
            "meant to be called directly; its capture runs as a forward "
            "hook on the watched module."
        )


# Kept for back-compat: external code imports AbstractStateCapture for typing.
# In this refactor the single concrete subclass (StateCaptureHook) carries the
# whole implementation; AbstractStateCapture is an alias.
class StateCaptureHook(Probe):
    """State-capture probe for PyTorch.

    Created by :func:`TensorState.attach` / :func:`TensorState.build_efficiency_model`.
    Owns one of three backend stores selected by the ``backend`` kwarg.
    """

    capture_on: bool = True
    _channel_index: int = 1

    def __init__(
        self,
        name: str,
        *,
        backend: str = "host",
        memory_device: torch.device | None = None,
        entropy_window_steps: int | None = None,
        disk_path: str | Path | None = None,
        memory_limit: str | None = None,
        raise_on_capture_error: bool = False,
    ) -> None:
        Probe.__init__(self)

        if backend not in ("gpu", "host", "duckdb"):
            raise ValueError(
                f"backend must be 'gpu' | 'host' | 'duckdb', got {backend!r}"
            )
        if backend != "duckdb" and (disk_path is not None or memory_limit is not None):
            raise ValueError(
                f"disk_path / memory_limit require backend='duckdb' (got {backend!r})"
            )
        if backend == "gpu" and (memory_device is None or memory_device.type != "cuda"):
            raise ValueError(
                f"backend='gpu' requires a CUDA torch.device; got {memory_device!r}"
            )

        self.name = name
        self._backend = backend
        self._device = memory_device
        self._entropy_window_steps = entropy_window_steps
        self._memory_limit = memory_limit
        # raise_on_capture_error kept as an attribute so existing tests that
        # poke it continue to work; it is now informational (capture is sync).
        self._raise_on_capture_error = bool(raise_on_capture_error)

        self._input_shape: tuple[int, ...] | None = None
        self._step_id: int = 0
        self._store: _BackendStore | None = None
        # Read-side caches. Invalidated on every append and every advance_step
        # (the window floor moves). Keyed by an internal append-version so we
        # don't need to wire callbacks from the store.
        self._states_cache: np.ndarray | None = None
        self._counts_cache: tuple[np.ndarray, np.ndarray] | None = None
        # Compatibility shim: lifecycle tests poke probe._capture_error =
        # RuntimeError(...) and then assert it clears on reset. Capture is now
        # synchronous so a real sticky error is impossible -- this attribute
        # exists purely to honor the existing test surface.
        self._capture_error: BaseException | None = None

        self._db_path: str | None = None
        if backend == "duckdb" and disk_path is not None:
            base = (
                disk_path if isinstance(disk_path, Path) else Path(disk_path)
            ) / "tensor_states"
            base.mkdir(parents=True, exist_ok=True)
            self._db_path = str((base / (name + ".duckdb")).absolute())

    # -- lifecycle --------------------------------------------------------

    def reset_states(self, input_shape: tuple[int, ...] | None = None) -> None:
        """(Re)initialize the underlying store and zero counters.

        The first call must provide ``input_shape``. Subsequent calls reuse
        the previously-recorded shape.
        """
        if input_shape is not None:
            self._input_shape = tuple(input_shape)
        if self._input_shape is None:
            raise ValueError("input_shape is required on the first reset_states call.")

        ncols = (int(self._input_shape[self._channel_index]) + 7) // 8

        self._states_cache = None
        self._counts_cache = None
        self._capture_error = None
        self._step_id = 0

        if self._store is not None:
            self._store.close()

        if self._backend == "gpu":
            assert self._device is not None  # guarded in __init__
            self._store = GPUMemoryStore(ncols, device=self._device)
        elif self._backend == "host":
            self._store = HostMemoryStore(ncols)
        else:
            self._store = _StateStore(
                ncols, path=self._db_path, memory_limit=self._memory_limit
            )

    def advance_step(self) -> int:
        """Bump the step counter; evict per the configured window."""
        self._step_id += 1
        self._states_cache = None
        self._counts_cache = None
        if self._entropy_window_steps is not None and self._store is not None:
            floor = self._step_id - self._entropy_window_steps + 1
            if floor > 0:
                self._store.evict_before(floor)
        return self._step_id

    def _window_floor(self) -> int:
        """Smallest step_id (inclusive) that falls in the current window."""
        if self._entropy_window_steps is None:
            return 0
        floor = self._step_id - self._entropy_window_steps + 1
        return floor if floor > 0 else 0

    # -- read-side API ----------------------------------------------------

    @property
    def state_count(self) -> int:
        """Total post-batch-dedup states in the current window."""
        if self._store is None:
            return 0
        return int(self._store.state_count(self._window_floor()))

    @state_count.setter
    def state_count(self, value):
        raise AttributeError("state_count attribute is read-only.")

    def _unique_counts(self) -> tuple[np.ndarray, np.ndarray]:
        if self._counts_cache is None:
            assert self._store is not None
            self._counts_cache = self._store.unique_counts_in_window(
                self._window_floor()
            )
        return self._counts_cache

    @property
    def states(self) -> np.ndarray:
        """Decompressed unique microstates as a boolean ``(K, num_neurons)`` array."""
        if self._states_cache is None:
            u, _c = self._unique_counts()
            self._states_cache = ts.decompress_states(u, int(self.max_entropy()))
        return self._states_cache

    @states.setter
    def states(self, value):
        raise AttributeError("states attribute is read-only.")

    @property
    def raw_states(self) -> np.ndarray:
        """Unique bit-packed microstates in the current window.

        Returns a ``(K, ncols)`` uint8 array. The previous insertion-ordered
        view leaked the DuckDB-blob-table implementation detail; every caller
        in practice fed it into entropy / decompress / state_ids, all of
        which want the unique set.
        """
        return self._unique_counts()[0]

    @raw_states.setter
    def raw_states(self, value):
        raise AttributeError("raw_states attribute is read-only.")

    def state_ids(self) -> list[bytes]:
        """Identity of observed states, aligned with :meth:`counts`."""
        return [bytes(row) for row in self._unique_counts()[0]]

    def counts(self, index: int | list[int] | np.ndarray | None = None) -> np.ndarray:
        """Per-unique-microstate counts in the current window.

        ``index`` selects a subset of rows in the unique-microstate array
        (the same rows :meth:`state_ids` returns); ``None`` returns every
        count.
        """
        _u, c = self._unique_counts()
        if index is None:
            return c
        idx = np.atleast_1d(np.asarray(index)).ravel()
        if idx.dtype == np.bool_:
            idx = np.flatnonzero(idx)
        if c.shape[0] == 0:
            return np.empty((0,), dtype=np.int64)
        return c[idx]

    def states_per_instance(self) -> float:
        """Average number of microstates contributed per input instance."""
        if self._input_shape is None:
            raise RuntimeError("input_shape is not set; run a forward pass first.")
        return float(
            np.prod(self._input_shape[1:]) / self._input_shape[self._channel_index]
        )

    def max_entropy(self) -> float:
        """Theoretical maximum entropy for the layer (== neuron count)."""
        if self._input_shape is None:
            raise RuntimeError("input_shape is not set; run a forward pass first.")
        return float(self._input_shape[self._channel_index])

    def entropy(self, alpha: float | None = 1) -> float:
        """Renyi entropy of order ``alpha`` (``None`` returns the theoretical max)."""
        if alpha is None:
            return self.max_entropy()
        result = TensorState.entropy(self.counts(), alpha)
        assert isinstance(result, (int, float))
        return float(result)

    def efficiency(self, alpha1: float = 1, alpha2: float | None = None) -> float:
        """Layer efficiency: ``entropy(alpha1) / entropy(alpha2)``."""
        assert isinstance(alpha1, (float, int)), "alpha1 must be a float or int"
        assert isinstance(alpha2, (float, int, type(None))), (
            "alpha2 must be a float, int, or None"
        )
        if alpha2 is not None:
            assert alpha1 > alpha2, "alpha1 must be larger than alpha 2"
        return self.entropy(alpha1) / self.entropy(alpha2)

    def to_arrow(self) -> pa.Table:
        """Captured microstates as a pyarrow Table (DuckDB backend only)."""
        if not isinstance(self._store, _StateStore):
            raise NotImplementedError(
                "to_arrow() is only available for backend='duckdb'; this "
                f"probe uses backend={self._backend!r}."
            )
        return self._store.to_arrow()

    # -- hot path ---------------------------------------------------------

    @torch._dynamo.disable
    def _capture(self, *args) -> None:
        """Forward-hook callable.

        Matches both forward-hook ``(module, inputs, output)`` and
        forward-pre-hook ``(module, inputs)`` signatures: the last positional
        argument is the output (post-hook) or the inputs tuple (pre-hook).

        Marked ``torch._dynamo.disable`` so torch.compile triggers a clean
        graph break and the host-side bit-pack runs eagerly (AIQ-22).
        """
        if not self.capture_on:
            return

        # Pre-hook delivers (module, inputs_tuple); pick the first input.
        out = args[-1]
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not isinstance(out, torch.Tensor):
            return  # nothing to capture from non-tensor module outputs.

        if self._input_shape is None:
            self.reset_states(tuple(out.shape))

        # Channels-last reshape: (N, C, ...) -> (N * spatial..., C).
        c = int(out.shape[1]) if out.ndim >= 2 else int(out.shape[-1])
        if out.ndim <= 2:
            x = out.detach().reshape(-1, c)
        else:
            dim_order = (0, *range(2, out.ndim), 1)
            x = out.detach().permute(*dim_order).contiguous().reshape(-1, c)

        with torch.no_grad():
            if self._backend == "gpu":
                # Pass the raw tensor: compress_states' Triton kernel does the
                # threshold internally for float inputs (fused), avoiding an
                # intermediate uint8 tensor.
                packed = ts.compress_states(x)
                assert isinstance(packed, torch.Tensor)
                uniq_t = torch.unique(packed, dim=0)
                assert self._store is not None
                assert isinstance(self._store, GPUMemoryStore)
                self._store.append(self._step_id, uniq_t)
            else:
                # Host / DuckDB: thresholded bool -> Rust _compress_tensor_pi8.
                bits_np = (x > 0).cpu().numpy()
                packed_np = ts.compress_states(bits_np)
                assert isinstance(packed_np, np.ndarray)
                uniq_np = np.unique(packed_np, axis=0)
                assert self._store is not None
                assert isinstance(self._store, (HostMemoryStore, _StateStore))
                self._store.append(self._step_id, uniq_np)
        # Invalidate read caches; the next read recomputes.
        self._states_cache = None
        self._counts_cache = None


# Backwards-compatibility alias. Old code imported AbstractStateCapture for
# typing or isinstance checks; collapse to StateCaptureHook so callers keep
# working without exposing a second class with no abstract methods.
AbstractStateCapture = StateCaptureHook
