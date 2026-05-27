"""Capture-thread failures must surface (AIQ-34).

State capture runs in a background thread; a failure there used to be
swallowed until results were read (or never, if they weren't). These tests
pin strategy #3: log at failure time, re-raise (chained) on read, and an
opt-in fail-fast flag.
"""

import logging
import threading
from concurrent.futures import wait as futures_wait

import pytest
import torch

import TensorState.States as ts_states  # noqa: N813 -- deliberate package alias
from TensorState.Layers import StateCaptureHook


def _boom(*_args, **_kwargs):
    raise ValueError("injected compress failure")


def test_capture_failure_does_not_raise_in_forward(monkeypatch):
    """A failing capture thread must not break the forward pass itself."""
    hook = StateCaptureHook(name="probe_forward", memory_device="cpu")
    monkeypatch.setattr(ts_states, "compress_states", _boom)

    # The hook call (what runs inside forward) returns normally even though
    # the capture thread it spawns will fail.
    hook._capture(None, torch.randn(8, 16))
    futures_wait(hook._threads)


def test_capture_failure_logged_and_reraised_on_read(monkeypatch, caplog):
    """Default behavior: log when it happens, re-raise (chained) on read."""
    hook = StateCaptureHook(name="probe_read", memory_device="cpu")
    monkeypatch.setattr(ts_states, "compress_states", _boom)

    with caplog.at_level(logging.ERROR, logger="TensorState.Layers"):
        hook._capture(None, torch.randn(8, 16))
        # The failure is logged by the capture thread's done-callback. Add our
        # own callback after the hook's so callbacks fire in registration order
        # — when ours signals, the hook's has already run and logged. This makes
        # the log assertion deterministic instead of racing the worker thread.
        done = threading.Event()
        hook._threads[-1].add_done_callback(lambda _f: done.set())
        assert done.wait(timeout=10), "capture thread did not finish"
        with pytest.raises(RuntimeError) as excinfo:
            _ = hook.state_count

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "injected compress failure" in str(excinfo.value.__cause__)
    assert any("State capture failed" in r.message for r in caplog.records)


def test_capture_error_is_sticky(monkeypatch):
    """Once a capture fails, every subsequent read keeps re-raising it."""
    hook = StateCaptureHook(name="probe_sticky", memory_device="cpu")
    monkeypatch.setattr(ts_states, "compress_states", _boom)

    hook._capture(None, torch.randn(8, 16))
    with pytest.raises(RuntimeError):
        _ = hook.state_count
    # A second read still raises the same chained cause, even with no new
    # capture (threads already drained).
    with pytest.raises(RuntimeError) as excinfo:
        _ = hook.state_count
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_raise_on_capture_error_fails_fast(monkeypatch):
    """With the flag set, the next capture aborts instead of running."""
    hook = StateCaptureHook(
        name="probe_ff", memory_device="cpu", raise_on_capture_error=True
    )
    monkeypatch.setattr(ts_states, "compress_states", _boom)

    # First capture proceeds (no prior error) and spawns the failing thread.
    hook._capture(None, torch.randn(8, 16))
    # Draining the read records the sticky error deterministically.
    with pytest.raises(RuntimeError):
        _ = hook.state_count
    # Now the next capture fails fast on the stored error.
    with pytest.raises(RuntimeError) as excinfo:
        hook._capture(None, torch.randn(8, 16))
    assert isinstance(excinfo.value.__cause__, ValueError)
