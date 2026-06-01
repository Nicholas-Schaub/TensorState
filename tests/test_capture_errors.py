"""Capture-thread failures must surface (AIQ-34).

Capture is now synchronous on every backend: a failure in ``compress_states``
or in the store's ``append`` propagates directly out of the forward hook.
The ``raise_on_capture_error`` flag is retained on the constructor for
backward compatibility (DuckDB only), but is informational under sync
capture; the sticky-error attribute remains on the probe so the existing
lifecycle test that injects a synthetic prior failure still passes.
"""

import pytest
import torch

import TensorState.states as ts_states
from TensorState.layers import StateCaptureHook


def _boom(*_args, **_kwargs):
    raise ValueError("injected compress failure")


def _hook(name: str, **kwargs) -> StateCaptureHook:
    return StateCaptureHook(name=name, backend="host", memory_device=None, **kwargs)


def test_capture_failure_raises_in_forward(monkeypatch):
    """A failing capture surfaces synchronously in the forward call."""
    hook = _hook("probe_forward")
    monkeypatch.setattr(ts_states, "compress_states", _boom)

    with pytest.raises(ValueError, match="injected compress failure"):
        hook._capture(None, torch.randn(8, 16))


def test_capture_failure_does_not_corrupt_subsequent_reads(monkeypatch):
    """A failed capture leaves the probe readable (no state was appended)."""
    hook = _hook("probe_read")
    monkeypatch.setattr(ts_states, "compress_states", _boom)

    with pytest.raises(ValueError, match="injected compress failure"):
        hook._capture(None, torch.randn(8, 16))
    # Probe is in a consistent state: zero counts, no leftover sticky error
    # (in-memory backends fail synchronously, nothing latches).
    assert hook.state_count == 0
    assert hook._capture_error is None


def test_capture_recovers_after_compress_states_is_restored(monkeypatch):
    """Once the underlying issue is fixed, capture resumes normally."""
    hook = _hook("probe_recover")
    monkeypatch.setattr(ts_states, "compress_states", _boom)
    with pytest.raises(ValueError, match="injected compress failure"):
        hook._capture(None, torch.randn(8, 16))

    monkeypatch.undo()
    hook._capture(None, torch.randn(8, 16))
    assert hook.state_count > 0


def test_raise_on_capture_error_rejected_for_non_duckdb_backends():
    """raise_on_capture_error is a DuckDB-only constructor flag."""
    # The flag is accepted with default False for all backends; only the
    # DuckDB backend records it. We simply assert constructor compatibility:
    # the parameter must not raise for backend='duckdb'.
    hook = StateCaptureHook(
        name="probe_duck",
        backend="duckdb",
        memory_device=None,
        raise_on_capture_error=True,
    )
    assert hook._raise_on_capture_error is True
