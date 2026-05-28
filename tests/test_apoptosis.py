"""Unit tests for the merge / destroy primitives in Dependency.py.

Validates per-node merge_outputs / merge_inputs / destroy_outputs /
destroy_inputs for each layer type, plus candidate-generation and
graph-level apoptose orchestration.
"""

import numpy as np
import pytest
import torch

from TensorState import testing as ts_testing
from TensorState.dependency import (
    ApoptosisType,
    AttentionNode,
    BatchNormNode,
    ConvGroupNode,
    ConvNode,
    GroupNormNode,
    LayerNormNode,
    LinearNode,
    ModuleData,
    ModuleGraph,
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
# Conv forward-invariant under real (N,C,H,W) input (AIQ-16 cases 1, 3, 5)
#
# The existing mean-on-producer / sum-on-consumer surgery is spatially correct
# because the channel ops never touch the spatial dims. These tests pin the
# *forward* invariant: merging two identical producer channels and summing the
# consumer's matching input columns must leave the downstream output unchanged,
# across kernel/stride/padding/dilation/bias and the transposed weight layout.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "conv_kwargs",
    [
        {"kernel_size": 3, "padding": 1},
        {"kernel_size": 3, "padding": 1, "stride": 2},
        {"kernel_size": 3, "padding": 1, "bias": False},
        {"kernel_size": 5, "padding": 2, "dilation": 2},
    ],
)
def test_conv_forward_invariant_under_merge_destroy(conv_kwargs):
    torch.manual_seed(0)
    prod = torch.nn.Conv2d(3, 8, **conv_kwargs)
    cons = torch.nn.Conv2d(8, 4, kernel_size=3, padding=1)

    # Force producer output channels 1 and 4 to be identical neurons.
    prod.weight.data[4] = prod.weight.data[1].clone()
    if prod.bias is not None:
        prod.bias.data[4] = prod.bias.data[1].clone()

    x = torch.randn(2, 3, 16, 16)
    before = cons(prod(x)).detach().clone()

    prod_node = ConvNode(_fake_module_data(prod))
    cn = ConvNode(_fake_module_data(cons))
    prod_node.merge_outputs([[1, 4]])
    cn.merge_inputs([[1, 4]])
    prod_node.destroy_outputs([4])
    cn.destroy_inputs([4])

    after = cons(prod(x)).detach()
    assert torch.allclose(before, after, atol=1e-5), (
        f"max diff {(before - after).abs().max().item()}"
    )
    assert prod.out_channels == 7
    assert cons.in_channels == 7


def test_convgroup_depthwise_forward_invariant_and_structure():
    """Depthwise conv: a channel and its producer are one linked neuron.

    The whole linked group (producer output + depthwise filter) must be
    merged together; destroy keeps in_channels == out_channels == groups.
    """
    torch.manual_seed(0)
    prod = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
    dw = torch.nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8)
    cons = torch.nn.Conv2d(8, 4, kernel_size=3, padding=1)

    # Channels 1 and 4 identical through the full linked group.
    prod.weight.data[4] = prod.weight.data[1].clone()
    prod.bias.data[4] = prod.bias.data[1].clone()
    dw.weight.data[4] = dw.weight.data[1].clone()
    dw.bias.data[4] = dw.bias.data[1].clone()

    x = torch.randn(2, 3, 16, 16)
    before = cons(dw(prod(x))).detach().clone()

    prod_node = ConvNode(_fake_module_data(prod))
    gn = ConvGroupNode(_fake_module_data(dw))
    cn = ConvNode(_fake_module_data(cons))
    prod_node.merge_outputs([[1, 4]])
    gn.merge_outputs([[1, 4]])
    cn.merge_inputs([[1, 4]])
    prod_node.destroy_outputs([4])
    gn.destroy_outputs([4])
    cn.destroy_inputs([4])

    after = cons(dw(prod(x))).detach()
    assert torch.allclose(before, after, atol=1e-5), (
        f"max diff {(before - after).abs().max().item()}"
    )
    assert dw.in_channels == dw.out_channels == dw.groups == 7


@pytest.mark.parametrize(
    "tconv_kwargs",
    [
        {"kernel_size": 3, "padding": 1},
        {"kernel_size": 4, "stride": 2, "padding": 1},
        {"kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    ],
)
def test_convtranspose_forward_invariant(tconv_kwargs):
    """Transposed conv weight is (C_in, C_out, ...): out-channels on dim 1.

    Exercises the dead ``layer.transposed`` branches and confirms
    output_padding (which touches spatial extent, not channels) is invariant.
    """
    torch.manual_seed(0)
    prod = torch.nn.ConvTranspose2d(3, 8, **tconv_kwargs)
    cons = torch.nn.Conv2d(8, 4, kernel_size=3, padding=1)

    # Out-channel axis is dim 1 for transposed weights.
    prod.weight.data[:, 4] = prod.weight.data[:, 1].clone()
    prod.bias.data[4] = prod.bias.data[1].clone()

    x = torch.randn(2, 3, 8, 8)
    before = cons(prod(x)).detach().clone()

    prod_node = ConvNode(_fake_module_data(prod))
    cn = ConvNode(_fake_module_data(cons))
    prod_node.merge_outputs([[1, 4]])
    cn.merge_inputs([[1, 4]])
    prod_node.destroy_outputs([4])
    cn.destroy_inputs([4])

    after = cons(prod(x)).detach()
    assert torch.allclose(before, after, atol=1e-5), (
        f"max diff {(before - after).abs().max().item()}"
    )
    assert prod.out_channels == 7


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


# ---------------------------------------------------------------------------
# MultiheadAttention — AttentionNode (head-level, EXACT-ONLY)
#
# The merge unit is a HEAD. in_proj_weight is packed (3*E, E) = [W_q; W_k; W_v]
# stacked row-blocks; head h owns rows h*hd:(h+1)*hd within each block (offsets
# 0, E, 2E) plus out_proj.weight COLUMNS h*hd:(h+1)*hd. Because softmax(QKᵀ) is
# non-linear in the projection weights, the exact forward invariant only holds
# for destroying dead/duplicate heads and merging IDENTICAL heads.
# ---------------------------------------------------------------------------


def _make_heads_identical(
    mha: torch.nn.MultiheadAttention, src: int, dst: int
) -> slice:
    """Make heads ``src`` and ``dst`` produce IDENTICAL per-head outputs.

    Destroying the merged duplicate shrinks ``embed_dim`` by ``head_dim``,
    which drops ``dst``'s embed slice from the input columns. So we first
    zero that input slice across all projections (no head reads it), then
    copy ``src``'s Q/K/V row-blocks + bias into ``dst`` so both heads apply
    the same projection to the same (remaining) input -> identical outputs.
    Returns the ``dst`` embed slice that will be removed on destroy.
    """
    e = mha.embed_dim
    hd = e // mha.num_heads
    w = mha.in_proj_weight.data
    b = mha.in_proj_bias.data
    dst_sl = slice(dst * hd, (dst + 1) * hd)
    with torch.no_grad():
        for block in range(3):
            off = block * e
            # No projection reads dst's input slice (it gets dropped on destroy).
            w[:, dst_sl] = 0.0
            # dst's row block == src's row block -> identical per-head output.
            w[off + dst * hd : off + (dst + 1) * hd] = w[
                off + src * hd : off + (src + 1) * hd
            ].clone()
            b[off + dst * hd : off + (dst + 1) * hd] = b[
                off + src * hd : off + (src + 1) * hd
            ].clone()
    return dst_sl


def _dead_head_slice(mha: torch.nn.MultiheadAttention, head: int) -> slice:
    """Fully isolate ``head``'s embed slice so removing it is lossless.

    Removing a head shrinks ``embed_dim`` (PyTorch requires
    ``embed_dim == head_dim * num_heads``), which drops the input columns
    and out_proj rows in that head's slice. The forward stays invariant on
    the surviving dimensions only if no surviving head reads that input
    slice and no surviving output dim is produced from it. So we zero the
    slice's input columns across all projections, the head's own output
    rows, and out_proj's coupling into/out of that slice.
    """
    e = mha.embed_dim
    hd = e // mha.num_heads
    sl = slice(head * hd, (head + 1) * hd)
    with torch.no_grad():
        for block in range(3):
            off = block * e
            # Zero this head's output rows (it produces nothing).
            mha.in_proj_weight.data[off + head * hd : off + (head + 1) * hd] = 0.0
            mha.in_proj_bias.data[off + head * hd : off + (head + 1) * hd] = 0.0
            # Zero this slice's INPUT columns so survivors don't read it.
            mha.in_proj_weight.data[:, sl] = 0.0
        # out_proj: zero this head's input columns and its output rows.
        mha.out_proj.weight.data[:, sl] = 0.0
        mha.out_proj.weight.data[sl, :] = 0.0
        if mha.out_proj.bias is not None:
            mha.out_proj.bias.data[sl] = 0.0
    return sl


@pytest.mark.parametrize("batch_first", [True, False])
def test_attention_destroy_zeroed_head_forward_invariant(batch_first):
    torch.manual_seed(0)
    e, h = 8, 4
    mha = torch.nn.MultiheadAttention(e, h, batch_first=batch_first)
    mha.eval()
    hd = e // h
    sl = _dead_head_slice(mha, head=2)
    keep = [i for i in range(e) if not (sl.start <= i < sl.stop)]

    x = torch.randn(2, 5, e) if batch_first else torch.randn(5, 2, e)
    before, _ = mha(x, x, x)
    before = before.detach().clone()

    node = AttentionNode(_fake_module_data(mha))
    node.destroy_outputs([2])

    # embed_dim shrank: feed the narrowed input, compare on kept dims.
    x_narrow = x[..., keep]
    after, _ = mha(x_narrow, x_narrow, x_narrow)
    after = after.detach()
    assert mha.num_heads == 3
    assert mha.embed_dim == e - hd
    assert torch.allclose(before[..., keep], after, atol=1e-4), (
        f"max diff {(before[..., keep] - after).abs().max().item()}"
    )


@pytest.mark.parametrize("batch_first", [True, False])
def test_attention_merge_identical_heads_forward_invariant(batch_first):
    torch.manual_seed(1)
    e, h = 8, 4
    mha = torch.nn.MultiheadAttention(e, h, batch_first=batch_first)
    mha.eval()
    hd = e // h
    # Make heads 0 and 1 produce identical per-head outputs, then merge {0,1}
    # (mean Q/K/V producer, sum out_proj consumer) and destroy the duplicate.
    # Survivor is head 0, the destroyed duplicate is head 1, so make head 1's
    # embed slice the disposable one and head 1 identical to head 0.
    sl = _make_heads_identical(mha, src=0, dst=1)
    keep = [i for i in range(e) if not (sl.start <= i < sl.stop)]

    x = torch.randn(2, 5, e) if batch_first else torch.randn(5, 2, e)
    before, _ = mha(x, x, x)
    before = before.detach().clone()

    node = AttentionNode(_fake_module_data(mha))
    redundant = node.merge_outputs(groups=[[0, 1]])
    assert redundant == [1]
    node.destroy_outputs(redundant)

    # embed_dim shrank by one head; feed narrowed input, compare kept dims.
    x_narrow = x[..., keep]
    after, _ = mha(x_narrow, x_narrow, x_narrow)
    after = after.detach()
    assert mha.num_heads == 3
    assert mha.embed_dim == e - hd
    assert torch.allclose(before[..., keep], after, atol=1e-4), (
        f"max diff {(before[..., keep] - after).abs().max().item()}"
    )


def test_attention_destroy_head_bookkeeping_shapes():
    e, h = 12, 4
    mha = torch.nn.MultiheadAttention(e, h)
    hd = e // h
    node = AttentionNode(_fake_module_data(mha))
    node.destroy_outputs([1])
    new_e = e - hd
    assert mha.embed_dim == new_e
    assert mha.num_heads == h - 1
    # in_proj_weight stays packed (3*E', E') and bias (3*E',).
    assert mha.in_proj_weight.shape == (3 * new_e, new_e)
    assert mha.in_proj_bias.shape == (3 * new_e,)
    # out_proj: Linear(E', E') -> weight (E', E').
    assert mha.out_proj.weight.shape == (new_e, new_e)
    assert mha.out_proj.in_features == new_e
    assert mha.out_proj.out_features == new_e


def test_attention_destroy_indivisible_raises():
    # embed_dim 8, 4 heads (hd=2). Removing one head -> embed_dim 6, heads 3 -> ok.
    # Construct a case where the leftover is NOT divisible: a model where heads
    # do not evenly tile is impossible for nn.MultiheadAttention construction,
    # so we assert the divisibility guard by directly forcing an inconsistent
    # state: embed_dim that leaves a non-divisible remainder is checked after.
    e, h = 8, 4
    mha = torch.nn.MultiheadAttention(e, h)
    node = AttentionNode(_fake_module_data(mha))
    # Monkey-force num_heads so that after removing one head, embed_dim %
    # num_heads != 0. With hd=2, destroy 1 head -> embed_dim 6. If we pretend
    # there are 4 heads remaining the guard must fire.
    mha.num_heads = 5  # 8 % 5 != 0 to begin; destroying yields indivisible
    with pytest.raises(AssertionError):
        node.destroy_outputs([0])


def test_attention_surfaces_as_single_node_in_graph():
    """A model containing MHA must produce exactly one AttentionNode whose
    data.module is the MHA instance, and out_proj must not surface separately."""
    torch.manual_seed(0)
    m = ts_testing.small_model("tiny_transformer")
    m.eval()
    inp = torch.randint(0, 64, (1, 8))
    graph = ModuleGraph(m, inp_tensor=inp)

    attention_nodes = []
    mha_modules = [
        mod for mod in m.modules() if isinstance(mod, torch.nn.MultiheadAttention)
    ]
    for grp in graph.groups:
        for v in grp.V():
            if isinstance(v, AttentionNode):
                attention_nodes.append(v)

    assert len(attention_nodes) == len(mha_modules), (
        f"expected {len(mha_modules)} AttentionNodes, got {len(attention_nodes)}"
    )
    surfaced_mhas = {id(v.data.module) for v in attention_nodes}
    assert surfaced_mhas == {id(mod) for mod in mha_modules}

    # out_proj children must NOT surface as their own module nodes.
    out_projs = {id(mod.out_proj) for mod in mha_modules}
    for grp in graph.groups:
        for v in grp.V():
            if isinstance(v.data, ModuleData):
                assert id(v.data.module) not in out_projs, (
                    "out_proj surfaced as its own node"
                )
