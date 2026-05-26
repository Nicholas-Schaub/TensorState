"""Small canonical reference models for tests, examples, and benchmarks.

Each architecture is sized to train on :func:`~TensorState.testing.synthetic.tiny_dataset`
(or ``tiny_text_dataset``) in seconds on CPU. They exist to give the
capture and apoptosis machinery concrete, lightweight targets:

- ``lenet5`` — the existing LeNet-5 variant (Conv + BatchNorm chain).
- ``mlp`` — a plain Linear stack.
- ``groupnorm_conv`` — Conv2d -> GroupNorm -> ReLU blocks, giving the
  apoptosis ``GroupNormNode`` a non-torchvision target.
- ``tiny_transformer`` — a small decoder-style transformer, giving
  attention apoptosis a real ``MultiheadAttention`` surface without
  pulling in HuggingFace.
"""

from __future__ import annotations

import torch
from torch import nn

from TensorState.models import LeNet_5


class _MLP(nn.Module):
    """Flatten -> Linear -> ReLU -> Dropout -> Linear."""

    def __init__(self, num_classes: int = 10, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GroupNormConv(nn.Module):
    """Conv2d -> GroupNorm -> ReLU blocks + global-pool head.

    Channel counts are multiples of ``groups`` so ``GroupNorm`` is valid.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        channels: tuple[int, ...] = (16, 16, 16),
        groups: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c = in_channels
        for out in channels:
            if out % groups != 0:
                raise ValueError(
                    f"channel count {out} not divisible by groups {groups}"
                )
            layers += [
                nn.Conv2d(c, out, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(groups, out),
                nn.ReLU(),
            ]
            c = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class _TinyTransformer(nn.Module):
    """Decoder-style transformer for next-token prediction.

    Two ``TransformerEncoderLayer`` blocks (batch-first) over a learned
    token + position embedding, with a tied-width vocab projection head.
    Small enough to train on ``tiny_text_dataset`` quickly, but it
    contains real ``MultiheadAttention`` modules for attention-apoptosis
    testing.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 128,
        num_layers: int = 2,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.size(1)
        h = self.embed(x) + self.pos[:, :seq]
        # Causal mask so position t only attends to <= t.
        mask = torch.triu(
            torch.full((seq, seq), float("-inf"), device=x.device), diagonal=1
        )
        h = self.encoder(h, mask=mask)
        return self.head(h)


def small_model(
    arch: str = "lenet5",
    in_channels: int = 3,
    num_classes: int = 10,
    **kwargs,
) -> nn.Module:
    """Construct a small reference model by name.

    Args:
        arch: One of ``"lenet5"``, ``"mlp"``, ``"groupnorm_conv"``,
            ``"tiny_transformer"``.
        in_channels: Input channel count (vision archs only).
        num_classes: Output class count (vision archs). For
            ``tiny_transformer`` the vocab size is controlled by the
            ``vocab_size`` kwarg instead.
        **kwargs: Architecture-specific overrides forwarded to the
            constructor (e.g. ``hidden``, ``channels``, ``groups``,
            ``vocab_size``, ``d_model``, ``nhead``).

    Returns:
        An ``nn.Module``.
    """
    if arch == "lenet5":
        return LeNet_5(num_classes=num_classes, **kwargs)
    if arch == "mlp":
        return _MLP(num_classes=num_classes, **kwargs)
    if arch == "groupnorm_conv":
        return _GroupNormConv(
            in_channels=in_channels, num_classes=num_classes, **kwargs
        )
    if arch == "tiny_transformer":
        return _TinyTransformer(**kwargs)
    raise ValueError(f"unknown arch: {arch!r}")
