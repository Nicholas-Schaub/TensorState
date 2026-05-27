import numpy as np
import pytest
import torch

from TensorState import States


def _make_input(compression, num_neurons, dtype="float32"):
    """Build the test input for the requested backend and dtype.

    Returns ``(mask, payload)`` where ``mask`` is the ground-truth boolean
    firing mask (``> 0``) and ``payload`` is what gets fed to
    ``compress_states`` in the requested dtype/backend.
    """
    rng = np.random.default_rng(0)
    a = (rng.random((2000, num_neurons)) - 0.5).astype(np.float32)
    mask = a > 0

    if dtype == "float32":
        payload_np = a
    elif dtype == "bool":
        payload_np = mask
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    if compression == "numpy":
        return mask, payload_np
    if compression == "torch":
        return mask, torch.from_numpy(payload_np)
    if compression == "torch_cuda":
        if not torch.cuda.is_available():
            pytest.skip("torch.cuda not available")
        return mask, torch.from_numpy(payload_np).cuda()
    raise ValueError(f"Unknown compression backend: {compression}")


@pytest.mark.parametrize("dtype", ["float32", "bool"])
@pytest.mark.parametrize("num_neurons", [2**n for n in range(10, 16)])
def test_roundtrip(num_neurons, compression, dtype):
    """Bit-pack then decompress; the result must match the > 0 mask of the input."""
    mask, payload = _make_input(compression, num_neurons, dtype=dtype)
    compressed = States.compress_states(payload)

    if isinstance(compressed, torch.Tensor):
        compressed = compressed.cpu().numpy()

    decompressed = States.decompress_states(compressed, num_neurons=num_neurons)

    assert np.all(mask == decompressed)


@pytest.mark.parametrize("dtype", ["float32", "bool"])
@pytest.mark.parametrize("num_neurons", [1, 7, 9, 13, 17, 1000, 1023])
def test_roundtrip_partial_byte(num_neurons, dtype):
    """Roundtrip with neuron counts that are not multiples of 8.

    Exercises the partial-byte tail of the bit-pack / unpack on both the
    float32 and bool numpy paths.
    """
    mask, payload = _make_input("numpy", num_neurons, dtype=dtype)
    compressed = States.compress_states(payload)
    decompressed = States.decompress_states(compressed, num_neurons=num_neurons)
    assert np.all(mask == decompressed)


@pytest.mark.parametrize("num_neurons", [8, 17, 64, 1023, 1024])
def test_compress_dtype_equivalence(num_neurons):
    """float32 and its bool > 0 mask must compress to identical bytes.

    Guards the bool numpy path (``_compress_tensor_pi8`` via uint8 view)
    against drifting from the float32 path (``_compress_tensor_ps``).
    """
    rng = np.random.default_rng(1)
    a = (rng.random((512, num_neurons)) - 0.5).astype(np.float32)
    mask = a > 0

    packed_f32 = States.compress_states(a)
    packed_bool = States.compress_states(mask)

    assert np.array_equal(packed_f32, packed_bool)


@pytest.mark.parametrize("num_neurons", [2**n for n in range(10, 16)])
def test_benchmark_compress(num_neurons, compression, benchmark):
    """Benchmark compress_states for the requested backend."""
    _, b = _make_input(compression, num_neurons)
    benchmark(States.compress_states, b)
