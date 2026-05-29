"""Public inspection API over attached probes (AIQ-25)."""

import numpy as np
import pytest
import torch

import TensorState as ts  # noqa: N813 -- deliberate package alias


class _TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def _model_with_states():
    """A tiny model with probes attached and one forward pass captured.

    Uses a custom module (not a top-level ``nn.Sequential``, whose ``forward``
    would iterate the probe container) so probes live off the forward path.
    """
    model = _TinyNet()
    ts.build_efficiency_model(model, attach_to=["Linear"])
    model.eval()
    model(torch.randn(64, 16))
    return model


def test_layers_returns_name_to_probe_dict():
    model = _model_with_states()
    probes = ts.layers(model)

    assert isinstance(probes, dict)
    assert len(probes) == 2  # two Linear layers, post hooks
    assert all(isinstance(p, ts.Probe) for p in probes.values())
    # Clean names: container suffix reduced to _post, no _states.
    assert all(k.endswith("_post") for k in probes)


def test_layer_single_lookup_and_keyerror():
    model = _model_with_states()
    probes = ts.layers(model)
    name = next(iter(probes))

    assert ts.layer(model, name) is probes[name]
    with pytest.raises(KeyError):
        ts.layer(model, "does.not.exist")


def test_entropy_model_form_dict_and_named():
    model = _model_with_states()
    ent = ts.entropy(model)

    assert isinstance(ent, dict)
    assert ent.keys() == ts.layers(model).keys()
    assert all(isinstance(v, float) for v in ent.values())

    name = next(iter(ent))
    scalar = ts.entropy(model, name=name)
    assert isinstance(scalar, float)
    assert scalar == pytest.approx(ent[name])


def test_entropy_count_form_unchanged():
    counts = np.array([3, 1])
    # Shannon entropy of a 3:1 split.
    expected = -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))
    assert ts.entropy(counts) == pytest.approx(expected)
    # name= is meaningless for the count form.
    with pytest.raises(TypeError):
        ts.entropy(counts, name="x")


def test_efficiency_dict_and_geomean_reduce():
    model = _model_with_states()
    eff = ts.efficiency(model)

    assert isinstance(eff, dict)
    assert eff.keys() == ts.layers(model).keys()

    reduced = ts.efficiency(model, reduce="geomean")
    assert isinstance(reduced, float)
    assert reduced == pytest.approx(ts.network_efficiency(model))

    with pytest.raises(ValueError, match="reduce"):
        ts.efficiency(model, reduce="mean")


def test_efficiency_layers_is_deprecated():
    model = _model_with_states()
    with pytest.warns(DeprecationWarning, match="ts.layers|TensorState.layers"):
        probes = list(model.efficiency_layers)
    assert len(probes) == 2
