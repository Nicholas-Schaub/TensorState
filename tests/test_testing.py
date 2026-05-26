"""Tests for the tensorstate.testing fixture package."""

import numpy as np
import pytest
import torch

from TensorState import testing as tst


def test_tiny_dataset_deterministic_and_shape():
    d1 = tst.tiny_dataset(n=32, channels=3, size=8, num_classes=5, seed=1)
    d2 = tst.tiny_dataset(n=32, channels=3, size=8, num_classes=5, seed=1)
    assert len(d1) == 32
    x1, y1 = d1[0]
    x2, y2 = d2[0]
    assert x1.shape == (3, 8, 8)
    assert x1.dtype == torch.float32
    assert y1.dtype == torch.int64
    assert torch.equal(x1, x2)
    assert int(y1) == int(y2)
    # Values are clamped to [0, 1].
    assert float(x1.min()) >= 0.0
    assert float(x1.max()) <= 1.0


def test_tiny_dataset_seed_changes_output():
    a = tst.tiny_dataset(n=8, seed=0)[0][0]
    b = tst.tiny_dataset(n=8, seed=1)[0][0]
    assert not torch.equal(a, b)


def test_tiny_text_dataset_shapes():
    d = tst.tiny_text_dataset(n=16, seq_len=12, vocab_size=32, num_classes=3, seed=0)
    assert len(d) == 16
    inp, tgt = d[0]
    assert inp.shape == (11,)
    assert tgt.shape == (11,)
    assert inp.dtype == torch.int64
    assert int(inp.max()) < 32


def test_tiny_loader_batches():
    loader = tst.tiny_loader(batch_size=8, n=24, size=8)
    xb, yb = next(iter(loader))
    assert xb.shape == (8, 3, 8, 8)
    assert yb.shape == (8,)


def test_random_states_density_and_dtype():
    s = tst.random_states(n=2000, neurons=16, density=0.3, seed=0)
    assert s.shape == (2000, 16)
    assert s.dtype == torch.bool
    # Empirical density should land near the requested density.
    assert abs(float(s.float().mean()) - 0.3) < 0.05


def test_random_states_deterministic():
    a = tst.random_states(seed=7)
    b = tst.random_states(seed=7)
    assert torch.equal(a, b)


@pytest.mark.parametrize(
    ("kind", "expected_unique"),
    [
        ("all_zero", 1),
        ("all_one", 1),
        ("singleton", 2),
        ("duplicates", 2),
    ],
)
def test_degenerate_states(kind, expected_unique):
    s = tst.degenerate_states(kind, n=20, neurons=17)
    assert s.shape == (20, 17)
    assert s.dtype == torch.bool
    unique = {tuple(row) for row in s.tolist()}
    assert len(unique) == expected_unique


def test_degenerate_states_rejects_unknown():
    with pytest.raises(ValueError, match="unknown degenerate-state kind"):
        tst.degenerate_states("nonsense")


def test_seed_all_reproducible():
    tst.seed_all(123)
    a = torch.randn(4)
    n_a = np.random.rand(4)
    tst.seed_all(123)
    b = torch.randn(4)
    n_b = np.random.rand(4)
    assert torch.equal(a, b)
    assert np.array_equal(n_a, n_b)


@pytest.mark.parametrize("arch", ["mlp", "groupnorm_conv"])
def test_small_model_vision_forward(arch):
    model = tst.small_model(arch, in_channels=3, num_classes=7)
    x = tst.tiny_dataset(n=8, channels=3, size=8)[:][0]
    out = model(x)
    assert out.shape == (8, 7)


def test_small_model_lenet5_forward():
    # LeNet-5's conv/pool stack needs a larger spatial input than the 8x8
    # default; 64x64 matches its design size.
    model = tst.small_model("lenet5", num_classes=10)
    x = tst.tiny_dataset(n=4, channels=3, size=64)[:][0]
    out = model(x)
    assert out.shape[0] == 4


def test_small_model_transformer_forward():
    model = tst.small_model("tiny_transformer", vocab_size=32, max_len=16)
    d = tst.tiny_text_dataset(n=4, seq_len=12, vocab_size=32)
    inp = torch.stack([d[i][0] for i in range(4)])
    out = model(inp)
    assert out.shape == (4, 11, 32)


def test_small_model_rejects_unknown():
    with pytest.raises(ValueError, match="unknown arch"):
        tst.small_model("nonsense")
