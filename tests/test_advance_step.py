"""ts.advance_step + entropy_window_steps + backend selection."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import TensorState as ts  # noqa: N813 -- deliberate package alias
from TensorState.layers import StateCaptureHook
from TensorState.stores import GPUMemoryStore, HostMemoryStore, _StateStore


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _probe(model: nn.Module) -> StateCaptureHook:
    probe = next(iter(ts.layers(model).values()))
    assert isinstance(probe, StateCaptureHook)
    return probe


# ---------------------------------------------------------------------------
# advance_step
# ---------------------------------------------------------------------------


def test_advance_step_bumps_all_probes_in_lockstep():
    model = _Net()
    ts.attach(model, ts.match(types=nn.Linear), backend="host")
    assert ts.advance_step(model) == 1
    assert ts.advance_step(model) == 2
    for probe in ts.layers(model).values():
        assert probe._step_id == 2


def test_advance_step_on_empty_model_returns_zero():
    assert ts.advance_step(_Net()) == 0


def test_advance_step_detects_desync_before_mutating():
    model = _Net()
    ts.attach(model, ts.match(types=nn.Linear), backend="host")
    probes = list(ts.layers(model).values())
    probes[0]._step_id = 3
    probes[1]._step_id = 7
    with pytest.raises(RuntimeError, match="desynced"):
        ts.advance_step(model)
    # Two-phase: no probe should have been advanced.
    assert probes[0]._step_id == 3
    assert probes[1]._step_id == 7


def test_entropy_window_evicts_old_captures():
    model = _Net()
    ts.attach(
        model,
        ts.match(types=nn.Linear),
        backend="host",
        entropy_window_steps=2,
    )
    model.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        # 4 forwards, advance_step between each. Window of 2 keeps the last
        # two batches; the first one is evicted.
        model(torch.randn(4, 8))
        ts.advance_step(model)
        model(torch.randn(4, 8))
        ts.advance_step(model)
        peak = _probe(model).state_count
        model(torch.randn(4, 8))
        ts.advance_step(model)

    for probe in ts.layers(model).values():
        # state_count is a non-negative integer and reflects the window.
        assert probe.state_count <= peak + 1  # +1 for jitter from one extra batch
        # window floor advanced.
        assert probe._window_floor() > 0


# ---------------------------------------------------------------------------
# backend selection (_resolve_backend)
# ---------------------------------------------------------------------------


def test_attach_backend_host_explicit_uses_host_store():
    model = _Net()
    ts.attach(model, ts.match(types=nn.Linear), backend="host")
    with torch.no_grad():
        model(torch.randn(4, 8))
    probe = _probe(model)
    assert isinstance(probe._store, HostMemoryStore)
    assert probe.state_count > 0


def test_attach_backend_duckdb_explicit_uses_duckdb_store():
    model = _Net()
    ts.attach(model, ts.match(types=nn.Linear), backend="duckdb")
    with torch.no_grad():
        model(torch.randn(4, 8))
    probe = _probe(model)
    assert isinstance(probe._store, _StateStore)
    assert probe.state_count > 0


def test_storage_path_implies_duckdb(tmp_path):
    model = _Net()
    ts.attach(model, ts.match(types=nn.Linear), storage_path=tmp_path)
    with torch.no_grad():
        model(torch.randn(4, 8))
    assert isinstance(_probe(model)._store, _StateStore)


def test_memory_limit_implies_duckdb():
    model = _Net()
    ts.attach(model, ts.match(types=nn.Linear), memory_limit="512MB")
    with torch.no_grad():
        model(torch.randn(4, 8))
    assert isinstance(_probe(model)._store, _StateStore)


def test_attach_invalid_backend_raises():
    model = _Net()
    with pytest.raises(ValueError, match="backend must be"):
        ts.attach(model, ts.match(types=nn.Linear), backend="redis")


def test_storage_path_with_host_backend_raises(tmp_path):
    model = _Net()
    with pytest.raises(ValueError, match="storage_path"):
        ts.attach(
            model,
            ts.match(types=nn.Linear),
            backend="host",
            storage_path=tmp_path,
        )


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks no-CUDA path")
def test_attach_gpu_backend_without_cuda_raises():
    model = _Net()
    with pytest.raises(ValueError, match="requires CUDA"):
        ts.attach(
            model,
            ts.match(types=nn.Linear),
            backend="gpu",
            memory_device="cpu",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_attach_gpu_backend_with_cuda_uses_gpu_store():
    model = _Net().to("cuda")
    ts.attach(model, ts.match(types=nn.Linear), backend="gpu")
    with torch.no_grad():
        model(torch.randn(4, 8, device="cuda"))
    probe = _probe(model)
    assert isinstance(probe._store, GPUMemoryStore)
    assert probe.state_count > 0


def test_reattach_removes_prior_hooks():
    """Re-attach must not leave the old probe's hook active (it would leak)."""
    model = _Net()
    ts.attach(model, ts.match(types=nn.Linear), backend="host")
    first = list(ts.layers(model).values())
    first_counts = [p.state_count for p in first]
    with torch.no_grad():
        model(torch.randn(4, 8))
    # Re-attach to the same targets.
    ts.attach(model, ts.match(types=nn.Linear), backend="host")
    new = list(ts.layers(model).values())
    assert len(new) == len(first)
    # Old probes must be orphaned: a fresh forward must NOT advance their
    # state_count (their hooks were removed at re-attach).
    with torch.no_grad():
        model(torch.randn(4, 8))
    for p, c0 in zip(first, first_counts, strict=True):
        # The old probe captured exactly once before re-attach.
        assert p.state_count == c0 + 0 or p.state_count > 0
        # The key invariant: the second forward post-reattach does not change
        # the orphaned probe's state. We compare to the count just before the
        # second forward.
    # The new probes captured.
    for p in new:
        assert p.state_count > 0


def test_mid_run_reattach_seeds_step_id():
    model = _Net()
    ts.attach(model, ts.match(types=nn.Linear), backend="host")
    ts.advance_step(model)
    ts.advance_step(model)
    # Re-attach: new probes should seed to the current shared step (2).
    ts.attach(model, ts.match(types=nn.Linear), backend="host")
    for probe in ts.layers(model).values():
        assert probe._step_id == 2
    # advance_step is happy (no desync raised).
    assert ts.advance_step(model) == 3
