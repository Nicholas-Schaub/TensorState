"""Lifecycle tests for the state-capture hook (AIQ-26).

These tests codify the documented attach / read / reset / detach surface of
``build_efficiency_model`` and friends. They pin current behavior so future
changes to the hook lifecycle are intentional.
"""

from __future__ import annotations

import io

import torch

import TensorState as ts  # noqa: N813 -- deliberate package alias
from TensorState import remove_state_layers
from TensorState import testing as ts_testing
from TensorState.layers import StateCaptureHook

# A single batch shape is reused so the BatchNorm layers inside LeNet's
# Conv2dNormActivation blocks see batch > 1 in train() mode.
_BATCH = (8, 3, 64, 64)


def _make_attached_model() -> torch.nn.Module:
    model = ts_testing.small_model("lenet5", num_classes=10)
    ts.build_efficiency_model(model, attach_to=["Conv2dNormActivation"])
    return model


def _hooks(model: torch.nn.Module) -> list[StateCaptureHook]:
    """Return attached probes narrowed to ``StateCaptureHook`` for ty.

    ``ts.layers`` is typed as ``dict[str, Probe]``; the elements ARE
    ``StateCaptureHook`` instances in practice but ty cannot narrow the
    abstract attributes (``state_count`` etc.) without an explicit
    ``isinstance`` filter, so we centralize the narrowing here.
    """
    out: list[StateCaptureHook] = []
    for probe in ts.layers(model).values():
        assert isinstance(probe, StateCaptureHook)
        out.append(probe)
    return out


def _conv_norm_act_count(model: torch.nn.Module) -> int:
    """Count Conv2dNormActivation modules in the model by class name.

    Using the class name avoids importing torchvision here just to do an
    isinstance check; ``build_efficiency_model`` itself dispatches by class
    name, so this mirrors what the attach path sees.
    """
    return sum(1 for m in model.modules() if type(m).__name__ == "Conv2dNormActivation")


def test_attach_idempotency_double_build_does_not_double_register():
    """Second ``build_efficiency_model`` replaces probes, does not double them.

    NOTE: This pins the *current* documented behavior — calling
    ``build_efficiency_model`` reinitializes ``_tensorstate_probes`` and
    ``state_capture_hooks`` on the model, so calling it twice with the
    same ``attach_to`` ends up with one probe per matched layer (not 2x).
    If this guarantee is ever tightened (e.g. raising on re-attach), this
    test should be tightened accordingly.
    """
    model = ts_testing.small_model("lenet5", num_classes=10)
    matched = _conv_norm_act_count(model)
    assert matched > 0, "test model has no Conv2dNormActivation layers"

    ts.build_efficiency_model(model, attach_to=["Conv2dNormActivation"])
    first_probes = ts.layers(model)
    assert len(first_probes) == matched

    # Re-attach with the same targets.
    ts.build_efficiency_model(model, attach_to=["Conv2dNormActivation"])
    second_probes = ts.layers(model)

    # Final probe count equals the number of matched layers, not 2x.
    assert len(second_probes) == matched
    # And ``state_capture_hooks`` is fresh (one handle per probe with method=
    # "after", the default).
    hooks: list = model.state_capture_hooks  # ty: ignore[invalid-assignment]
    assert len(hooks) == matched


def test_detach_removes_probes_and_hooks():
    """After ``remove_state_layers``, no probes/hooks remain and capture stops."""
    model = _make_attached_model()

    # Prime the hooks once so probes have non-zero state_count before detach.
    model.eval()
    with torch.no_grad():
        model(torch.randn(*_BATCH))
    probes_before = _hooks(model)
    assert probes_before, "no probes attached for detach test"
    counts_before = [p.state_count for p in probes_before]
    assert all(c > 0 for c in counts_before)

    remove_state_layers(model)

    # Container is gone.
    assert not hasattr(model, "_tensorstate_probes")
    # No StateCaptureHook reachable from the module tree.
    assert not any(isinstance(m, StateCaptureHook) for _, m in model.named_modules())
    # ts.layers() reports nothing.
    assert ts.layers(model) == {}

    # The hook handles previously registered have been removed: subsequent
    # forwards must not touch the detached probe objects. We verify by
    # checking each previously-attached probe's state_count is unchanged
    # after another forward pass through the (now hookless) model.
    with torch.no_grad():
        model(torch.randn(*_BATCH))
    counts_after = [p.state_count for p in probes_before]
    assert counts_after == counts_before, (
        f"detached probes still captured: {counts_before} -> {counts_after}"
    )


def test_reset_zeroes_counts_and_clears_sticky_capture_error():
    """``reset_efficiency_model`` returns probes to a pristine state."""
    model = _make_attached_model()
    model.eval()
    with torch.no_grad():
        model(torch.randn(*_BATCH))

    probes = _hooks(model)
    assert probes
    for probe in probes:
        assert probe.state_count > 0
        # Inject a sticky capture error to verify reset clears it.
        probe._capture_error = RuntimeError("synthetic prior failure")

    ts.reset_efficiency_model(model)

    for probe in _hooks(model):
        assert probe.state_count == 0
        assert probe._capture_error is None

    # A subsequent forward must repopulate counts (capture is functional
    # again after reset).
    with torch.no_grad():
        model(torch.randn(*_BATCH))
    for probe in _hooks(model):
        assert probe.state_count > 0


def test_capture_on_false_suppresses_capture_then_resumes():
    """Flipping ``capture_on`` gates the hook without removing it."""
    model = _make_attached_model()
    model.eval()
    with torch.no_grad():
        model(torch.randn(*_BATCH))

    probes = _hooks(model)
    assert probes
    counts_after_one_pass = {p.name: p.state_count for p in probes}
    assert all(c > 0 for c in counts_after_one_pass.values())

    # Disable capture and run a forward; counts must NOT advance.
    for probe in probes:
        probe.capture_on = False
    with torch.no_grad():
        model(torch.randn(*_BATCH))
    for probe in probes:
        assert probe.state_count == counts_after_one_pass[probe.name], (
            f"probe {probe.name} captured while capture_on=False"
        )

    # Re-enable and run another forward; counts must advance again.
    for probe in probes:
        probe.capture_on = True
    with torch.no_grad():
        model(torch.randn(*_BATCH))
    for probe in probes:
        assert probe.state_count > counts_after_one_pass[probe.name], (
            f"probe {probe.name} did not resume capture when capture_on=True"
        )


def test_state_dict_does_not_carry_observational_state():
    """Probe observational state must not round-trip through ``state_dict``.

    The store is a plain Python attribute (not a registered buffer), so a
    checkpoint cannot smuggle per-batch observations between training runs.
    """
    model = _make_attached_model()
    model.eval()
    with torch.no_grad():
        model(torch.randn(*_BATCH))

    sd = model.state_dict()
    # No probe-owned observational key may appear in the checkpoint.
    bad = [
        k
        for k in sd
        if "_state_cache" in k or "_tensorstate_probes" in k or "_store" in k
    ]
    assert bad == [], f"observational state leaked into state_dict: {bad}"

    # Round-trip via torch.save / torch.load into a fresh (un-attached) model.
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    reloaded_sd = torch.load(buf, weights_only=True)

    fresh = ts_testing.small_model("lenet5", num_classes=10)
    fresh.load_state_dict(reloaded_sd)
    # The fresh model isn't attached, so there's nothing to read; the
    # explicit check is that it has no probes at all.
    assert ts.layers(fresh) == {}
    assert not hasattr(fresh, "_tensorstate_probes")


def test_capture_mode_neutral_train_vs_eval():
    """Capture is not gated on ``module.training``: both modes record states.

    The hook does not gate on ``module.training``; this test pins that so a
    future change is intentional. Under the windowed-store contract,
    ``state_count`` is post-batch-dedup so the exact numbers differ with
    random inputs; we assert both modes record nonzero counts on every
    probe.
    """
    model = _make_attached_model()

    # eval() pass.
    model.eval()
    with torch.no_grad():
        model(torch.randn(*_BATCH))
    eval_counts = {p.name: p.state_count for p in _hooks(model)}
    assert all(c > 0 for c in eval_counts.values())

    ts.reset_efficiency_model(model)
    for probe in _hooks(model):
        assert probe.state_count == 0

    # train() pass at the same batch shape (BatchNorm needs N > 1, which
    # the chosen _BATCH satisfies).
    model.train()
    model(torch.randn(*_BATCH))
    train_counts = {p.name: p.state_count for p in _hooks(model)}

    # Both modes captured for every probe.
    assert eval_counts.keys() == train_counts.keys()
    assert all(c > 0 for c in train_counts.values())
