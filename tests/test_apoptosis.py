"""Unit tests for the merge / destroy primitives in Dependency.py.

Validates per-node merge_outputs / merge_inputs / destroy_outputs /
destroy_inputs for each layer type, plus candidate-generation and
graph-level apoptose orchestration.
"""

import numpy as np
import pytest
import torch

from TensorState.Dependency import (
    ApoptosisType,
    BatchNormNode,
    ConvNode,
    GroupNormNode,
    LayerNormNode,
    LinearNode,
    ModuleData,
    correlated_weight_groups,
    zero_info_groups,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_module_data(module: torch.nn.Module) -> ModuleData:
    """Build a ModuleData carrier with a dummy grad_fn so a node can wrap it."""
    # Use a real grad_fn from a small operation so the BaseModel validator
    # is happy. The graph traversal path is not exercised in these unit tests.
    y = (torch.ones(1, requires_grad=True) * 2).clone()
    return ModuleData(name=y.grad_fn.name(), grad_fn=y.grad_fn, module=module)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def test_zero_info_groups_separates_always_on_off_and_synced():
    rng = np.random.default_rng(0)
    N = 1000
    states = rng.choice([True, False], size=(N, 12), p=[0.5, 0.5])
    states[:, 0] = False  # always off
    states[:, 1] = False  # always off
    states[:, 2] = True  # always on
    states[:, 3] = True  # always on
    states[:, 5] = states[:, 4]  # synced pair

    groups = zero_info_groups(states)

    assert set(groups[0]) == {0, 1}, "always-off group"
    assert set(groups[1]) == {2, 3}, "always-on group"
    # The synced pair must appear together in one of the remaining groups
    found = False
    for g in groups[2:]:
        if set(g) == {4, 5}:
            found = True
            break
    assert found, f"synced pair {{4, 5}} not detected; got {groups[2:]}"


def test_correlated_weight_groups_drops_uncorrelated():
    # Build a weight matrix where rows 0 and 1 are identical and 2 is different
    w = torch.zeros(3, 4)
    w[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    w[1] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    w[2] = torch.tensor([4.0, 3.0, 2.0, 1.0])

    refined = correlated_weight_groups(w, candidates=[[0, 1, 2]], threshold=0.99)
    # The cluster should contain only {0, 1} (perfectly correlated)
    assert len(refined) == 1
    assert set(refined[0]) == {0, 1}


# ---------------------------------------------------------------------------
# Linear merge / destroy
# ---------------------------------------------------------------------------


def test_linear_merge_outputs_averages_weights():
    layer = torch.nn.Linear(4, 8, bias=True)
    layer.weight.data = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    layer.bias.data = torch.arange(8, dtype=torch.float32)

    node = LinearNode(_fake_module_data(layer))
    redundant = node.merge_outputs(groups=[[0, 1, 2]])

    # Output index 0 (the survivor) should equal the mean of rows 0, 1, 2
    expected_weight_0 = (
        torch.arange(32, dtype=torch.float32).reshape(8, 4)[[0, 1, 2]].mean(dim=0)
    )
    assert torch.allclose(layer.weight.data[0], expected_weight_0)

    expected_bias_0 = torch.tensor([0.0, 1.0, 2.0]).mean()
    assert layer.bias.data[0].item() == pytest.approx(expected_bias_0.item())

    assert redundant == [1, 2]


def test_linear_merge_inputs_sums_columns():
    layer = torch.nn.Linear(4, 8)
    layer.weight.data = torch.arange(32, dtype=torch.float32).reshape(8, 4)

    node = LinearNode(_fake_module_data(layer))
    redundant = node.merge_inputs(groups=[[0, 1]])

    # Column 0 should be sum of columns 0 and 1
    original = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    expected_col_0 = original[:, 0] + original[:, 1]
    assert torch.allclose(layer.weight.data[:, 0], expected_col_0)

    assert redundant == [1]


def test_linear_destroy_outputs_removes_rows():
    layer = torch.nn.Linear(4, 8)
    original = layer.weight.data.clone()
    node = LinearNode(_fake_module_data(layer))
    keep, _idxs = node.destroy_outputs([1, 3, 5])
    assert keep == [0, 2, 4, 6, 7]
    assert layer.weight.shape == (5, 4)
    assert layer.out_features == 5
    assert torch.equal(layer.weight.data, original[[0, 2, 4, 6, 7]])


def test_linear_destroy_inputs_removes_columns():
    layer = torch.nn.Linear(8, 4)
    original = layer.weight.data.clone()
    node = LinearNode(_fake_module_data(layer))
    keep, _ = node.destroy_inputs([1, 3, 5])
    assert keep == [0, 2, 4, 6, 7]
    assert layer.weight.shape == (4, 5)
    assert layer.in_features == 5
    assert torch.equal(layer.weight.data, original[:, [0, 2, 4, 6, 7]])


# ---------------------------------------------------------------------------
# Conv merge / destroy
# ---------------------------------------------------------------------------


def test_conv_merge_outputs_averages_4d_weight():
    layer = torch.nn.Conv2d(3, 8, kernel_size=3, bias=True)
    original = layer.weight.data.clone()
    original_bias = layer.bias.data.clone()
    node = ConvNode(_fake_module_data(layer))
    redundant = node.merge_outputs(groups=[[2, 4, 7]])

    expected = original[[2, 4, 7]].mean(dim=0)
    assert torch.allclose(layer.weight.data[2], expected)
    assert layer.bias.data[2].item() == pytest.approx(
        original_bias[[2, 4, 7]].mean().item()
    )
    assert redundant == [4, 7]


def test_conv_merge_inputs_sums_columns():
    layer = torch.nn.Conv2d(8, 16, kernel_size=3, bias=False)
    original = layer.weight.data.clone()
    node = ConvNode(_fake_module_data(layer))
    redundant = node.merge_inputs(groups=[[1, 5]])

    expected = original[:, 1] + original[:, 5]
    assert torch.allclose(layer.weight.data[:, 1], expected)
    assert redundant == [5]


# ---------------------------------------------------------------------------
# BatchNorm
# ---------------------------------------------------------------------------


def test_batchnorm_merge_outputs_averages_all_four():
    bn = torch.nn.BatchNorm2d(8)
    bn.weight.data = torch.arange(8, dtype=torch.float32)
    bn.bias.data = torch.arange(8, dtype=torch.float32) * 2
    bn.running_mean.data = torch.arange(8, dtype=torch.float32) * 3
    bn.running_var.data = torch.arange(8, dtype=torch.float32) * 4 + 1

    node = BatchNormNode(_fake_module_data(bn))
    redundant = node.merge_outputs(groups=[[1, 3, 5]])

    assert bn.weight.data[1].item() == pytest.approx(3.0)  # mean of 1, 3, 5
    assert bn.bias.data[1].item() == pytest.approx(6.0)  # mean of 2, 6, 10
    assert bn.running_mean.data[1].item() == pytest.approx(9.0)  # mean of 3, 9, 15
    assert redundant == [3, 5]


# ---------------------------------------------------------------------------
# GroupNorm
# ---------------------------------------------------------------------------


def test_groupnorm_merge_outputs_averages_affine():
    gn = torch.nn.GroupNorm(num_groups=2, num_channels=8, affine=True)
    gn.weight.data = torch.arange(8, dtype=torch.float32)
    gn.bias.data = torch.arange(8, dtype=torch.float32) * 10

    node = GroupNormNode(_fake_module_data(gn))
    node.merge_outputs(groups=[[0, 1]])

    assert gn.weight.data[0].item() == pytest.approx(0.5)
    assert gn.bias.data[0].item() == pytest.approx(5.0)


def test_groupnorm_destroy_outputs_respects_num_groups_divisibility():
    gn = torch.nn.GroupNorm(num_groups=2, num_channels=8)
    node = GroupNormNode(_fake_module_data(gn))
    # Removing 4 channels from a num_groups=2 layer leaves 4 channels = 2 per group ✓
    node.destroy_outputs([0, 1, 2, 3])
    assert gn.num_channels == 4

    gn2 = torch.nn.GroupNorm(num_groups=2, num_channels=8)
    node2 = GroupNormNode(_fake_module_data(gn2))
    with pytest.raises(AssertionError):
        # Removing 3 channels leaves 5 — not divisible by 2 groups.
        node2.destroy_outputs([0, 1, 2])


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------


def test_layernorm_merge_outputs_averages_affine():
    ln = torch.nn.LayerNorm(normalized_shape=8, elementwise_affine=True)
    ln.weight.data = torch.arange(8, dtype=torch.float32)
    ln.bias.data = torch.arange(8, dtype=torch.float32) * 10

    node = LayerNormNode(_fake_module_data(ln))
    node.merge_outputs(groups=[[2, 4, 6]])

    assert ln.weight.data[2].item() == pytest.approx(4.0)  # mean of 2, 4, 6
    assert ln.bias.data[2].item() == pytest.approx(40.0)  # mean of 20, 40, 60


def test_layernorm_destroy_outputs():
    ln = torch.nn.LayerNorm(8)
    node = LayerNormNode(_fake_module_data(ln))
    node.destroy_outputs([0, 1])
    # normalized_shape becomes a tuple of len 1 with the new size, or an int.
    new_shape = (
        ln.normalized_shape
        if isinstance(ln.normalized_shape, int)
        else ln.normalized_shape[0]
    )
    assert new_shape == 6
    assert ln.weight.shape == (6,)


# ---------------------------------------------------------------------------
# Apoptose linearity check (end-to-end on a small model)
# ---------------------------------------------------------------------------


def test_apoptose_preserves_output_for_identical_neurons():
    """If two output channels of a Linear layer have identical weights,
    apoptose-merging them should leave the downstream output unchanged
    when the consuming layer's input weights are summed."""
    torch.manual_seed(0)

    layer1 = torch.nn.Linear(4, 8)
    layer2 = torch.nn.Linear(8, 2)

    # Force channels 3 and 5 to have identical weights in layer1
    layer1.weight.data[5] = layer1.weight.data[3].clone()
    layer1.bias.data[5] = layer1.bias.data[3].clone()

    model = torch.nn.Sequential(layer1, layer2)
    x = torch.randn(1, 4)
    y_before = model(x).detach().clone()

    # Manual apoptose-style: merge layer1's outputs 3 and 5,
    # then sum layer2's input columns 3 and 5,
    # then destroy the redundant rows/cols.
    node1 = LinearNode(_fake_module_data(layer1))
    node2 = LinearNode(_fake_module_data(layer2))

    node1.merge_outputs(groups=[[3, 5]])
    node2.merge_inputs(groups=[[3, 5]])
    node1.destroy_outputs([5])
    node2.destroy_inputs([5])

    y_after = model(x).detach()
    assert torch.allclose(y_before, y_after, atol=1e-5), (
        f"output diverged after merge/destroy: max diff = "
        f"{(y_before - y_after).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# ApoptosisType enum
# ---------------------------------------------------------------------------


def test_apoptosis_type_flag_arithmetic():
    assert int(ApoptosisType.states) == 0
    assert ApoptosisType.weights | ApoptosisType.connections == ApoptosisType.wc
    assert ApoptosisType.weights & ApoptosisType.wc == ApoptosisType.weights
