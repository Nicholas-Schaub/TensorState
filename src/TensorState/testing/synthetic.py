"""Deterministic synthetic data for tests, examples, and benchmarks.

Nothing here downloads anything. Every generator is seeded by a local
``torch.Generator`` so the same kwargs produce bit-identical output and
no global RNG state is mutated.

The vision and text datasets are trivially learnable, so a small model
reaches high accuracy in a few CPU epochs — enough to assert that a
loss / regularizer / capture pipeline is wired correctly end-to-end.
The state generators (``random_states`` / ``degenerate_states``) produce
boolean ``(observations, neurons)`` matrices that exercise the
bit-pack and lex-sort paths directly, without needing a model at all.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


def tiny_dataset(
    n: int = 64,
    channels: int = 3,
    size: int = 8,
    num_classes: int = 10,
    seed: int = 0,
) -> Dataset:
    """Synthetic image classification dataset.

    Each class has a deterministic per-class mean image; samples are
    ``class_mean + small Gaussian noise`` clamped to ``[0, 1]``.

    Args:
        n: Number of samples.
        channels: Image channel count.
        size: Spatial height = width.
        num_classes: Number of classes.
        seed: RNG seed.

    Returns:
        A ``TensorDataset`` of ``(images, labels)`` where images are
        ``float32`` of shape ``(n, channels, size, size)`` and labels are
        ``int64`` in ``[0, num_classes)``.
    """
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(num_classes, channels, size, size, generator=g)
    labels = torch.randint(0, num_classes, (n,), generator=g)
    noise = 0.1 * torch.randn(n, channels, size, size, generator=g)
    images = (means[labels] + noise).clamp(0.0, 1.0)
    return TensorDataset(images, labels)


def tiny_text_dataset(
    n: int = 256,
    seq_len: int = 32,
    vocab_size: int = 64,
    num_classes: int = 4,
    seed: int = 0,
) -> Dataset:
    """Synthetic next-token-prediction dataset.

    Each sample's token sequence is generated from a class-specific
    Markov chain (a per-class transition matrix), so a small transformer
    can fit it. The LM target is the input shifted by one position.

    Args:
        n: Number of sequences.
        seq_len: Sequence length (inputs and targets are ``seq_len - 1``).
        vocab_size: Token vocabulary size.
        num_classes: Number of distinct Markov chains.
        seed: RNG seed.

    Returns:
        A ``TensorDataset`` of ``(inputs, targets)``, both ``int64`` of
        shape ``(n, seq_len - 1)``.
    """
    g = torch.Generator().manual_seed(seed)
    trans = torch.rand(num_classes, vocab_size, vocab_size, generator=g)
    trans = trans / trans.sum(dim=-1, keepdim=True)

    classes = torch.randint(0, num_classes, (n,), generator=g)
    seqs = torch.zeros(n, seq_len, dtype=torch.long)
    seqs[:, 0] = torch.randint(0, vocab_size, (n,), generator=g)
    for t in range(1, seq_len):
        probs = trans[classes, seqs[:, t - 1]]
        seqs[:, t] = torch.multinomial(probs, 1, generator=g).squeeze(1)

    return TensorDataset(seqs[:, :-1].contiguous(), seqs[:, 1:].contiguous())


def tiny_loader(batch_size: int = 16, **dataset_kwargs) -> DataLoader:
    """Wrap :func:`tiny_dataset` in a ``DataLoader``.

    Args:
        batch_size: Batch size.
        **dataset_kwargs: Forwarded to :func:`tiny_dataset`.

    Returns:
        A ``DataLoader`` over a fresh :func:`tiny_dataset`.
    """
    return DataLoader(tiny_dataset(**dataset_kwargs), batch_size=batch_size)


def random_states(
    n: int = 128,
    neurons: int = 17,
    density: float = 0.3,
    seed: int = 0,
) -> torch.Tensor:
    """Random boolean firing-state matrix.

    Args:
        n: Number of observations (rows).
        neurons: Number of neurons (columns). The default 17 is
            deliberately not a multiple of 8 so the bit-pack partial-byte
            tail is exercised.
        density: Probability that any given neuron fires.
        seed: RNG seed.

    Returns:
        A boolean tensor of shape ``(n, neurons)``.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, neurons, generator=g) < density


def degenerate_states(
    kind: Literal["all_zero", "all_one", "singleton", "duplicates"],
    n: int = 64,
    neurons: int = 17,
) -> torch.Tensor:
    """Edge-case boolean firing-state matrix.

    Args:
        kind: One of:
            - ``"all_zero"``: no neuron ever fires (one unique state).
            - ``"all_one"``: every neuron always fires (one unique state).
            - ``"singleton"``: all rows share one state except a single
              unique row (tests rare-state handling).
            - ``"duplicates"``: two alternating states, heavily repeated.
        n: Number of observations (rows).
        neurons: Number of neurons (columns).

    Returns:
        A boolean tensor of shape ``(n, neurons)``.
    """
    if kind == "all_zero":
        return torch.zeros(n, neurons, dtype=torch.bool)
    if kind == "all_one":
        return torch.ones(n, neurons, dtype=torch.bool)
    if kind == "singleton":
        out = torch.zeros(n, neurons, dtype=torch.bool)
        # One unique row that differs from the all-zero majority.
        out[0, :] = True
        return out
    if kind == "duplicates":
        a = torch.zeros(neurons, dtype=torch.bool)
        a[::2] = True
        b = ~a
        rows = [a if i % 2 == 0 else b for i in range(n)]
        return torch.stack(rows)
    raise ValueError(f"unknown degenerate-state kind: {kind!r}")
