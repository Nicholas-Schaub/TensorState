"""Tests for the predicate-based ``ts.attach`` / ``ts.match`` API (AIQ-23)."""

import re

import pytest
import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation

import TensorState as ts  # noqa: N813 -- deliberate package alias
from TensorState import testing as ts_testing
from TensorState.Layers import StateCaptureHook


def _input():
    return torch.randn(8, 3, 64, 64)


def _model():
    return ts_testing.small_model("lenet5", num_classes=10)


# ---------------------------------------------------------------------------
# match() — structured kwargs
# ---------------------------------------------------------------------------


def test_match_types_filters_by_isinstance():
    m = _model()
    matcher = ts.match(types=Conv2dNormActivation)
    targets = [n for n, mod in m.named_modules() if matcher(n, mod)]
    assert all(isinstance(m.get_submodule(n), Conv2dNormActivation) for n in targets)
    assert len(targets) >= 1


def test_match_name_regex_string_uses_search():
    m = _model()
    matcher = ts.match(name=r"^features\.\d+$")
    targets = [n for n, mod in m.named_modules() if matcher(n, mod)]
    # LeNet5's wrapped conv blocks: features.0, features.1, features.2.
    assert {"features.0", "features.1", "features.2"}.issubset(set(targets))


def test_match_name_compiled_pattern_works():
    m = _model()
    matcher = ts.match(name=re.compile(r"classifier"))
    assert any(matcher(n, mod) for n, mod in m.named_modules() if "classifier" in n)


def test_match_name_iterable_does_exact_match():
    m = _model()
    matcher = ts.match(name=["features.0", "features.2"])
    targets = {n for n, mod in m.named_modules() if matcher(n, mod)}
    assert targets == {"features.0", "features.2"}


def test_match_predicate_can_inspect_module():
    m = _model()
    matcher = ts.match(predicate=lambda _n, mod: isinstance(mod, nn.Linear))
    targets = [mod for n, mod in m.named_modules() if matcher(n, mod)]
    assert targets and all(isinstance(t, nn.Linear) for t in targets)


def test_match_combines_kwargs_with_and():
    m = _model()
    # Conv2dNormActivation AND name ending in "1" -> only features.1.
    matcher = ts.match(types=Conv2dNormActivation, name=r"\.1$")
    targets = {n for n, mod in m.named_modules() if matcher(n, mod)}
    assert targets == {"features.1"}


# ---------------------------------------------------------------------------
# Matcher composition — | & ~
# ---------------------------------------------------------------------------


def test_match_or_unions():
    m = _model()
    a = ts.match(types=nn.Linear)
    b = ts.match(name=r"^features\.\d+$")
    combined = a | b
    only_a = {n for n, mod in m.named_modules() if a(n, mod)}
    only_b = {n for n, mod in m.named_modules() if b(n, mod)}
    union = {n for n, mod in m.named_modules() if combined(n, mod)}
    assert union == only_a | only_b


def test_match_and_intersects():
    m = _model()
    a = ts.match(types=nn.Conv2d)
    b = ts.match(name=r"\.0$")  # ends with ".0"
    combined = a & b
    union_a = {n for n, mod in m.named_modules() if a(n, mod)}
    inter = {n for n, mod in m.named_modules() if combined(n, mod)}
    assert inter <= union_a  # intersection is a subset of each operand
    assert all(n.endswith(".0") for n in inter)


def test_match_invert_negates():
    m = _model()
    a = ts.match(types=nn.Linear)
    not_a = ~a
    linears = {n for n, mod in m.named_modules() if a(n, mod)}
    non_linears = {n for n, mod in m.named_modules() if not_a(n, mod)}
    assert linears.isdisjoint(non_linears)


# ---------------------------------------------------------------------------
# attach() — behavior and parity with build_efficiency_model
# ---------------------------------------------------------------------------


def test_attach_matches_build_efficiency_model_for_class_name():
    """ts.attach(types=X) and build_efficiency_model(attach_to=[X]) match the same set."""
    a = _model()
    ts.attach(a, where=ts.match(types=Conv2dNormActivation))

    b = _model()
    ts.build_efficiency_model(b, attach_to=["Conv2dNormActivation"])

    assert list(ts.layers(a).keys()) == list(ts.layers(b).keys())


def test_attach_when_before_uses_pre_hooks():
    m = _model()
    ts.attach(m, where=ts.match(types=Conv2dNormActivation), when="before")
    probes = ts.layers(m)
    assert all(name.endswith("_pre") for name in probes)


def test_attach_when_both_attaches_pre_and_post():
    m = _model()
    ts.attach(m, where=ts.match(types=Conv2dNormActivation), when="both")
    keys = set(ts.layers(m).keys())
    assert any(k.endswith("_pre") for k in keys)
    assert any(k.endswith("_post") for k in keys)


def test_attach_invalid_when_raises():
    m = _model()
    with pytest.raises(ValueError, match="when must be"):
        ts.attach(m, where=ts.match(types=nn.Linear), when="bogus")


def test_attach_captures_under_forward():
    """End-to-end: attached probes capture state across a forward pass."""
    m = _model()
    ts.attach(m, where=ts.match(types=Conv2dNormActivation))
    m.eval()
    with torch.no_grad():
        m(_input())
    for probe in ts.layers(m).values():
        assert isinstance(probe, StateCaptureHook)
        assert probe.state_count > 0
