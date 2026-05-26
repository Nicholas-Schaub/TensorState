"""Deterministic, download-free fixtures for tests, examples, and benchmarks.

Importable as ``from TensorState.testing import ...``. A pytest plugin
(registered via the ``pytest11`` entry point) re-exports the most-used
generators as fixtures so downstream repositories get them for free.
"""

from TensorState.testing.models import small_model
from TensorState.testing.seeding import seed_all
from TensorState.testing.synthetic import (
    degenerate_states,
    random_states,
    tiny_dataset,
    tiny_loader,
    tiny_text_dataset,
)

__all__ = [
    "degenerate_states",
    "random_states",
    "seed_all",
    "small_model",
    "tiny_dataset",
    "tiny_loader",
    "tiny_text_dataset",
]
