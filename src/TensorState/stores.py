"""Step-keyed microstate stores.

Three concrete backends for bit-packed microstates: GPU tensor, host numpy
array, and DuckDB. They share an informal contract enforced at the single
call site that constructs them (`StateCaptureHook.reset_states`):

    ncols: int
    append(step_id: int, packed_unique) -> None
    evict_before(min_step_id: int) -> None
    unique_in_window(min_step_id: int) -> np.ndarray
    unique_counts_in_window(min_step_id: int) -> tuple[np.ndarray, np.ndarray]
    state_count(min_step_id: int) -> int
    close() -> None

Rows arrive already deduped within the batch and in monotonically
non-decreasing step_id order. That lets the in-memory backends use
`searchsorted` for both windowed reads and eviction in O(log N).
"""

from __future__ import annotations

import contextlib
import threading

import duckdb
import numpy as np
import pyarrow as pa
import torch


class GPUMemoryStore:
    """In-VRAM step-keyed store of bit-packed microstates.

    Plain Python object (not nn.Module): the storage tensor is a regular
    attribute, so the probe's state_dict stays empty and a model.to(device)
    after attach does NOT migrate the store. The append-time device check
    surfaces a mismatched .to() loudly.
    """

    def __init__(
        self, ncols: int, *, device: torch.device, initial_rows: int = 4096
    ) -> None:
        self.ncols = int(ncols)
        self.device = device
        self._rows = torch.empty(
            (max(initial_rows, 1), self.ncols), dtype=torch.uint8, device=device
        )
        self._step = torch.empty(
            (max(initial_rows, 1),), dtype=torch.int64, device=device
        )
        self._n = 0

    def _grow_to(self, need: int) -> None:
        cap = max(self._rows.shape[0], 1)
        while cap < need:
            cap *= 2
        old_rows, old_step = self._rows, self._step
        self._rows = torch.empty(
            (cap, self.ncols), dtype=torch.uint8, device=self.device
        )
        self._step = torch.empty((cap,), dtype=torch.int64, device=self.device)
        if self._n > 0:
            self._rows[: self._n].copy_(old_rows[: self._n])
            self._step[: self._n].copy_(old_step[: self._n])
        del old_rows, old_step

    def append(self, step_id: int, packed_unique: torch.Tensor) -> None:
        k = int(packed_unique.shape[0])
        if k == 0:
            return
        if packed_unique.ndim != 2 or packed_unique.shape[1] != self.ncols:
            raise ValueError(
                f"expected (k, {self.ncols}) uint8, got {tuple(packed_unique.shape)}"
            )
        if packed_unique.device != self.device:
            raise RuntimeError(
                f"GPUMemoryStore.append: tensor device {packed_unique.device} "
                f"!= store device {self.device}. Attach the probe AFTER "
                f"model.to(device); the store is fixed at attach time."
            )
        end = self._n + k
        if end > self._rows.shape[0]:
            self._grow_to(end)
        self._rows[self._n : end] = packed_unique
        self._step[self._n : end] = step_id
        self._n = end

    def _cut(self, min_step_id: int) -> int:
        """First index >= min_step_id. step_id is monotonically non-decreasing."""
        if self._n == 0 or min_step_id <= 0:
            return 0
        # searchsorted on a sorted prefix: O(log N), no allocations.
        return int(
            torch.searchsorted(
                self._step[: self._n],
                torch.tensor([min_step_id], device=self.device),
            ).item()
        )

    def evict_before(self, min_step_id: int) -> None:
        if min_step_id <= 0 or self._n == 0:
            return
        cut = self._cut(min_step_id)
        if cut == 0:
            return
        if cut >= self._n:
            self._n = 0
            return
        # Slide the surviving suffix down. torch.copy_ rejects aliased ranges,
        # so we clone the source first; this is one O(keep) allocation, same
        # as a boolean mask + index_select but without their overhead.
        keep = self._n - cut
        self._rows[:keep] = self._rows[cut : self._n].clone()
        self._step[:keep] = self._step[cut : self._n].clone()
        self._n = keep

    def _window_view(self, min_step_id: int) -> torch.Tensor:
        cut = self._cut(min_step_id)
        return self._rows[cut : self._n]

    def unique_in_window(self, min_step_id: int) -> np.ndarray:
        view = self._window_view(min_step_id)
        if view.shape[0] == 0:
            return np.empty((0, self.ncols), dtype=np.uint8)
        return torch.unique(view, dim=0).cpu().numpy()

    def unique_counts_in_window(
        self, min_step_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        view = self._window_view(min_step_id)
        if view.shape[0] == 0:
            return (
                np.empty((0, self.ncols), dtype=np.uint8),
                np.empty((0,), dtype=np.int64),
            )
        u, c = torch.unique(view, dim=0, return_counts=True)
        return u.cpu().numpy(), c.cpu().numpy().astype(np.int64, copy=False)

    def state_count(self, min_step_id: int = 0) -> int:
        if self._n == 0:
            return 0
        if min_step_id <= 0:
            return int(self._n)
        return int(self._n - self._cut(min_step_id))

    def close(self) -> None:
        self._n = 0
        # Free VRAM by dropping references; allow re-use via reset_states which
        # creates a fresh store anyway.
        self._rows = torch.empty((1, self.ncols), dtype=torch.uint8, device=self.device)
        self._step = torch.empty((1,), dtype=torch.int64, device=self.device)


class HostMemoryStore:
    """Host-RAM step-keyed store of bit-packed microstates.

    Mirror of GPUMemoryStore over numpy. Synchronous: a numpy slice
    assignment has nothing to overlap with.
    """

    def __init__(self, ncols: int, *, initial_rows: int = 4096) -> None:
        self.ncols = int(ncols)
        self._rows = np.empty((max(initial_rows, 1), self.ncols), dtype=np.uint8)
        self._step = np.empty((max(initial_rows, 1),), dtype=np.int64)
        self._n = 0

    def _grow_to(self, need: int) -> None:
        cap = max(self._rows.shape[0], 1)
        while cap < need:
            cap *= 2
        new_rows = np.empty((cap, self.ncols), dtype=np.uint8)
        new_step = np.empty((cap,), dtype=np.int64)
        if self._n > 0:
            new_rows[: self._n] = self._rows[: self._n]
            new_step[: self._n] = self._step[: self._n]
        self._rows, self._step = new_rows, new_step

    def append(self, step_id: int, packed_unique: np.ndarray) -> None:
        k = int(packed_unique.shape[0])
        if k == 0:
            return
        if packed_unique.ndim != 2 or packed_unique.shape[1] != self.ncols:
            raise ValueError(
                f"expected (k, {self.ncols}) uint8, got {packed_unique.shape}"
            )
        end = self._n + k
        if end > self._rows.shape[0]:
            self._grow_to(end)
        self._rows[self._n : end] = packed_unique
        self._step[self._n : end] = step_id
        self._n = end

    def _cut(self, min_step_id: int) -> int:
        if self._n == 0 or min_step_id <= 0:
            return 0
        return int(np.searchsorted(self._step[: self._n], min_step_id, side="left"))

    def evict_before(self, min_step_id: int) -> None:
        if min_step_id <= 0 or self._n == 0:
            return
        cut = self._cut(min_step_id)
        if cut == 0:
            return
        if cut >= self._n:
            self._n = 0
            return
        keep = self._n - cut
        self._rows[:keep] = self._rows[cut : self._n]
        self._step[:keep] = self._step[cut : self._n]
        self._n = keep

    def _window_view(self, min_step_id: int) -> np.ndarray:
        cut = self._cut(min_step_id)
        return self._rows[cut : self._n]

    def unique_in_window(self, min_step_id: int) -> np.ndarray:
        view = self._window_view(min_step_id)
        if view.shape[0] == 0:
            return np.empty((0, self.ncols), dtype=np.uint8)
        return np.unique(view, axis=0)

    def unique_counts_in_window(
        self, min_step_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        view = self._window_view(min_step_id)
        if view.shape[0] == 0:
            return (
                np.empty((0, self.ncols), dtype=np.uint8),
                np.empty((0,), dtype=np.int64),
            )
        u, c = np.unique(view, axis=0, return_counts=True)
        return u, c.astype(np.int64, copy=False)

    def state_count(self, min_step_id: int = 0) -> int:
        if self._n == 0:
            return 0
        if min_step_id <= 0:
            return int(self._n)
        return int(self._n - self._cut(min_step_id))

    def close(self) -> None:
        self._n = 0
        self._rows = np.empty((1, self.ncols), dtype=np.uint8)
        self._step = np.empty((1,), dtype=np.int64)


class _StateStore:
    """DuckDB-backed step-keyed store of bit-packed microstates.

    Rows are staged in memory and flushed in Arrow batches because DuckDB is
    slow at many tiny appends. Synchronous: capture failures surface in the
    forward pass directly, matching the in-memory backends. This trades the
    old async-write overlap for simpler error semantics and predictable
    timing.
    """

    _FLUSH_ROWS = 1 << 20

    def __init__(
        self,
        ncols: int,
        *,
        path: str | None = None,
        memory_limit: str | None = None,
        flush_rows: int = _FLUSH_ROWS,
    ) -> None:
        self.ncols = int(ncols)
        self._flush_rows = int(flush_rows)
        self._lock = threading.Lock()
        self._con = duckdb.connect(path if path is not None else ":memory:")
        if memory_limit is not None:
            self._con.execute(f"SET memory_limit='{memory_limit}'")
        self._con.execute(
            "CREATE OR REPLACE TABLE states (step_id BIGINT NOT NULL, s BLOB NOT NULL)"
        )
        # Buffer is list[tuple[step_id, rows_ndarray]] in monotonic step order.
        self._buffer: list[tuple[int, np.ndarray]] = []
        self._buffered = 0
        self._n = 0  # total rows currently in store (buffer + on-disk).

    def append(self, step_id: int, packed_unique: np.ndarray) -> None:
        if packed_unique.ndim != 2 or packed_unique.shape[1] != self.ncols:
            raise ValueError(
                f"expected (k, {self.ncols}) uint8 rows, got "
                f"shape {packed_unique.shape}"
            )
        k = int(packed_unique.shape[0])
        if k == 0:
            return
        rows = np.ascontiguousarray(packed_unique, dtype=np.uint8)
        with self._lock:
            self._buffer.append((int(step_id), rows))
            self._buffered += k
            self._n += k
            if self._buffered >= self._flush_rows:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        # Pre-size to avoid intermediate vstack + concatenate allocations.
        total = self._buffered
        rows_np = np.empty((total, self.ncols), dtype=np.uint8)
        steps_np = np.empty((total,), dtype=np.int64)
        cursor = 0
        for sid, arr in self._buffer:
            m = arr.shape[0]
            rows_np[cursor : cursor + m] = arr
            steps_np[cursor : cursor + m] = sid
            cursor += m
        # Zero-copy wrap: pa.py_buffer accepts buffer-protocol objects.
        fsb = pa.FixedSizeBinaryArray.from_buffers(
            pa.binary(self.ncols), total, [None, pa.py_buffer(rows_np)]
        )
        tbl = pa.table(  # noqa: F841 -- referenced by DuckDB replacement scan
            {"step_id": pa.array(steps_np, type=pa.int64()), "s": fsb}
        )
        self._con.execute("INSERT INTO states SELECT step_id, s FROM tbl")
        self._buffer = []
        self._buffered = 0

    def evict_before(self, min_step_id: int) -> None:
        if min_step_id <= 0:
            return
        with self._lock:
            # Drop in-buffer rows older than the floor (buffer is step-ordered).
            keep_buffer = [(s, r) for s, r in self._buffer if s >= min_step_id]
            dropped_buffer = self._buffered - sum(r.shape[0] for _, r in keep_buffer)
            self._buffer = keep_buffer
            self._buffered -= dropped_buffer
            # Delete on-disk rows and count what we actually removed.
            (n_db_before,) = self._con.execute(
                "SELECT COUNT(*) FROM states WHERE step_id < ?",
                [int(min_step_id)],
            ).fetchone()
            self._con.execute(
                "DELETE FROM states WHERE step_id < ?", [int(min_step_id)]
            )
            self._n -= dropped_buffer + int(n_db_before)

    def unique_in_window(self, min_step_id: int) -> np.ndarray:
        with self._lock:
            self._flush_locked()
            rows = self._con.execute(
                "SELECT s FROM states WHERE step_id >= ? GROUP BY s",
                [int(min_step_id)],
            ).fetchall()
        if not rows:
            return np.empty((0, self.ncols), dtype=np.uint8)
        return np.stack([np.frombuffer(r[0], dtype=np.uint8) for r in rows])

    def unique_counts_in_window(
        self, min_step_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            self._flush_locked()
            rows = self._con.execute(
                "SELECT s, COUNT(*) FROM states WHERE step_id >= ? GROUP BY s",
                [int(min_step_id)],
            ).fetchall()
        if not rows:
            return (
                np.empty((0, self.ncols), dtype=np.uint8),
                np.empty((0,), dtype=np.int64),
            )
        u = np.stack([np.frombuffer(r[0], dtype=np.uint8) for r in rows])
        c = np.array([r[1] for r in rows], dtype=np.int64)
        return u, c

    def state_count(self, min_step_id: int = 0) -> int:
        with self._lock:
            if min_step_id <= 0:
                return int(self._n)
            self._flush_locked()
            (n,) = self._con.execute(
                "SELECT COUNT(*) FROM states WHERE step_id >= ?",
                [int(min_step_id)],
            ).fetchone()
            return int(n)

    def to_arrow(self) -> pa.Table:
        """All stored microstates as a pyarrow Table.

        Convenience for inspection. Returns the (step_id, s) schema -- not
        deduplicated. For grouped counts call :meth:`unique_counts_in_window`.
        """
        with self._lock:
            self._flush_locked()
            return self._con.execute("SELECT step_id, s FROM states").to_arrow_table()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._con.close()


__all__ = ["GPUMemoryStore", "HostMemoryStore", "_StateStore"]
