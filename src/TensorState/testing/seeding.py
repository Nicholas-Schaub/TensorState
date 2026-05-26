"""Reproducible RNG seeding for tests, examples, and benchmarks."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger("TensorState.testing")


def seed_all(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and torch RNGs for reproducible runs.

    Args:
        seed: The seed value applied to all RNGs.
        deterministic: When True, additionally enables
            ``torch.use_deterministic_algorithms(True)`` and the cuDNN
            deterministic flags. This makes results bit-reproducible but
            errors on ops that lack a deterministic implementation
            (e.g. ``F.interpolate`` backward, ``scatter_add_``,
            ``CTCLoss`` backward) and requires the environment variable
            ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` to be set *before* the
            process starts for cuBLAS GEMMs. Defaults to False — seeds
            initial conditions without forcing algorithm determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(
            "seed_all: deterministic algorithms enabled. Ensure "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 is set in the environment "
            "for cuBLAS determinism."
        )
