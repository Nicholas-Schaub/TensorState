"""Pytest plugin re-exporting the testing fixtures.

Registered via the ``pytest11`` entry point in ``pyproject.toml``, so any
project that installs tensorstate can request these fixtures by name
without importing anything.
"""

from __future__ import annotations

import pytest
import torch

from TensorState.testing import synthetic


@pytest.fixture
def tiny_loader() -> torch.utils.data.DataLoader:
    """A small download-free image DataLoader (see ``synthetic.tiny_loader``)."""
    return synthetic.tiny_loader()


@pytest.fixture
def tiny_dataset() -> torch.utils.data.Dataset:
    """A small download-free image Dataset (see ``synthetic.tiny_dataset``)."""
    return synthetic.tiny_dataset()


@pytest.fixture
def tiny_text_dataset() -> torch.utils.data.Dataset:
    """A small synthetic token-sequence Dataset for transformer tests."""
    return synthetic.tiny_text_dataset()


@pytest.fixture
def random_states() -> torch.Tensor:
    """A random boolean firing-state matrix (see ``synthetic.random_states``)."""
    return synthetic.random_states()
