"""Direct tests for the three step-keyed microstate backends.

Exercises the contract documented in :mod:`TensorState.stores` independently
from the probe glue. The GPU tests are gated on CUDA availability.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from TensorState.stores import GPUMemoryStore, HostMemoryStore, _StateStore


def _pack(rows: list[list[int]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.uint8)


# ---------------------------------------------------------------------------
# HostMemoryStore
# ---------------------------------------------------------------------------


def test_host_append_and_state_count():
    s = HostMemoryStore(ncols=2, initial_rows=2)
    s.append(1, _pack([[1, 2], [3, 4]]))
    s.append(2, _pack([[5, 6]]))
    assert s.state_count() == 3
    assert s.state_count(min_step_id=2) == 1


def test_host_grows_buffer():
    s = HostMemoryStore(ncols=1, initial_rows=2)
    s.append(1, _pack([[1], [2], [3], [4], [5]]))
    assert s.state_count() == 5


def test_host_unique_in_window_dedups():
    s = HostMemoryStore(ncols=1)
    s.append(1, _pack([[1], [2], [1]]))
    s.append(2, _pack([[3]]))
    u = s.unique_in_window(0)
    assert sorted(u.flatten().tolist()) == [1, 2, 3]


def test_host_unique_counts_aligned():
    s = HostMemoryStore(ncols=1)
    s.append(1, _pack([[1], [2], [1]]))
    s.append(2, _pack([[2]]))
    u, c = s.unique_counts_in_window(0)
    pairs = dict(zip(u.flatten().tolist(), c.tolist(), strict=True))
    assert pairs == {1: 2, 2: 2}


def test_host_evict_before_drops_old_rows():
    s = HostMemoryStore(ncols=1)
    s.append(1, _pack([[1], [2]]))
    s.append(5, _pack([[3]]))
    s.evict_before(3)
    assert s.state_count() == 1
    assert s.unique_in_window(0).flatten().tolist() == [3]


def test_host_evict_before_zero_is_noop():
    s = HostMemoryStore(ncols=1)
    s.append(1, _pack([[1]]))
    s.evict_before(0)
    s.evict_before(-1)
    assert s.state_count() == 1


def test_host_evict_before_past_end_clears():
    s = HostMemoryStore(ncols=1)
    s.append(1, _pack([[1]]))
    s.append(2, _pack([[2]]))
    s.evict_before(99)
    assert s.state_count() == 0


def test_host_state_count_with_floor():
    s = HostMemoryStore(ncols=1)
    s.append(1, _pack([[1]]))
    s.append(2, _pack([[2]]))
    s.append(3, _pack([[3]]))
    assert s.state_count(0) == 3
    assert s.state_count(2) == 2
    assert s.state_count(99) == 0


def test_host_append_shape_mismatch_raises():
    s = HostMemoryStore(ncols=2)
    with pytest.raises(ValueError, match="expected"):
        s.append(1, _pack([[1, 2, 3]]))


def test_host_empty_store_returns_empty():
    s = HostMemoryStore(ncols=2)
    assert s.state_count() == 0
    assert s.unique_in_window(0).shape == (0, 2)
    u, c = s.unique_counts_in_window(0)
    assert u.shape == (0, 2)
    assert c.shape == (0,)


def test_host_close_is_idempotent():
    s = HostMemoryStore(ncols=1)
    s.append(1, _pack([[1]]))
    s.close()
    s.close()
    assert s.state_count() == 0


# ---------------------------------------------------------------------------
# _StateStore (DuckDB)
# ---------------------------------------------------------------------------


def test_duckdb_append_and_state_count():
    s = _StateStore(ncols=1)
    try:
        s.append(1, _pack([[1], [2]]))
        s.append(2, _pack([[3]]))
        assert s.state_count() == 3
    finally:
        s.close()


def test_duckdb_unique_in_window():
    s = _StateStore(ncols=1)
    try:
        s.append(1, _pack([[1], [2], [1]]))
        s.append(2, _pack([[3]]))
        u = s.unique_in_window(0)
        assert sorted(u.flatten().tolist()) == [1, 2, 3]
        u2 = s.unique_in_window(2)
        assert u2.flatten().tolist() == [3]
    finally:
        s.close()


def test_duckdb_unique_counts_in_window():
    s = _StateStore(ncols=1)
    try:
        s.append(1, _pack([[1], [2], [1]]))
        s.append(2, _pack([[2]]))
        u, c = s.unique_counts_in_window(0)
        pairs = dict(zip(u.flatten().tolist(), c.tolist(), strict=True))
        assert pairs == {1: 2, 2: 2}
    finally:
        s.close()


def test_duckdb_evict_before_drops_old():
    s = _StateStore(ncols=1)
    try:
        s.append(1, _pack([[1]]))
        s.append(5, _pack([[2]]))
        s.evict_before(3)
        assert s.state_count() == 1
        assert s.unique_in_window(0).flatten().tolist() == [2]
    finally:
        s.close()


def test_duckdb_state_count_with_floor():
    s = _StateStore(ncols=1)
    try:
        s.append(1, _pack([[1]]))
        s.append(2, _pack([[2]]))
        s.append(3, _pack([[3]]))
        assert s.state_count(2) == 2
    finally:
        s.close()


def test_duckdb_to_arrow_returns_step_keyed_table():
    s = _StateStore(ncols=1)
    try:
        s.append(7, _pack([[5]]))
        tbl = s.to_arrow()
        assert tbl.num_rows == 1
        assert {"step_id", "s"} <= set(tbl.schema.names)
    finally:
        s.close()


def test_duckdb_evict_then_append_keeps_counts_consistent():
    s = _StateStore(ncols=1)
    try:
        s.append(1, _pack([[1]]))
        s.append(2, _pack([[2]]))
        s.evict_before(2)
        s.append(3, _pack([[3]]))
        assert s.state_count() == 2
        assert sorted(s.unique_in_window(0).flatten().tolist()) == [2, 3]
    finally:
        s.close()


# ---------------------------------------------------------------------------
# GPUMemoryStore
# ---------------------------------------------------------------------------


gpu_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@gpu_only
def test_gpu_append_and_unique():
    dev = torch.device("cuda", 0)
    s = GPUMemoryStore(ncols=1, device=dev)
    s.append(1, torch.tensor([[1], [2], [1]], dtype=torch.uint8, device=dev))
    assert s.state_count() == 3
    u = s.unique_in_window(0)
    assert sorted(u.flatten().tolist()) == [1, 2]


@gpu_only
def test_gpu_evict_before_drops_old():
    dev = torch.device("cuda", 0)
    s = GPUMemoryStore(ncols=1, device=dev)
    s.append(1, torch.tensor([[1]], dtype=torch.uint8, device=dev))
    s.append(5, torch.tensor([[2], [3]], dtype=torch.uint8, device=dev))
    s.evict_before(5)
    assert s.state_count() == 2


@gpu_only
def test_gpu_device_mismatch_raises():
    dev = torch.device("cuda", 0)
    s = GPUMemoryStore(ncols=1, device=dev)
    cpu_rows = torch.tensor([[1]], dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="device"):
        s.append(1, cpu_rows)
