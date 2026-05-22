import numpy as np
import pytest
import torch

from TensorState import States


def _make_input(compression, num_neurons):
    """Build the test input for the requested backend."""
    rng = np.random.default_rng(0)
    a = (rng.random((10000, num_neurons)) - 0.5).astype(np.float32)
    if compression == "numpy":
        return a, a
    if compression == "torch":
        return a, torch.from_numpy(a)
    if compression == "torch_cuda":
        if not torch.cuda.is_available():
            pytest.skip("torch.cuda not available")
        return a, torch.from_numpy(a).cuda()
    raise ValueError(f"Unknown compression backend: {compression}")


@pytest.mark.parametrize("num_neurons", [2**n for n in range(10, 16)])
def test_roundtrip(num_neurons, compression):
    """Bit-pack then decompress; the result must match the > 0 mask of the input."""
    a, b = _make_input(compression, num_neurons)
    compressed = States.compress_states(b)

    if isinstance(compressed, torch.Tensor):
        compressed = compressed.cpu().numpy()

    decompressed = States.decompress_states(compressed, num_neurons=num_neurons)

    assert np.all((a > 0) == decompressed)


@pytest.mark.parametrize("num_neurons", [2**n for n in range(10, 16)])
def test_benchmark_compress(num_neurons, compression, benchmark):
    """Benchmark compress_states for the requested backend."""
    _, b = _make_input(compression, num_neurons)
    benchmark(States.compress_states, b)
