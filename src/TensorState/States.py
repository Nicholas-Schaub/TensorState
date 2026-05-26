"""State compression, decompression, and sorting primitives.

The CPU bit-packing and lex-sort routines are implemented in the Cython
extension ``_TensorState``. The GPU bit-packing path is a Triton kernel
that matches the throughput of the previous CuPy ``ElementwiseKernel``
while requiring no CuPy dependency (Triton ships with PyTorch). For
torch tensors on CPU there is a slower fallback using native torch ops.

The previous implementation routed GPU data through CuPy with a custom
``ElementwiseKernel`` and a CuPy ``lexsort``. CuPy has been removed; CPU
sorting always uses the Cython ``_lex_sort`` (round-tripping CPU data
through GPU just for sorting is not worthwhile).
"""

import logging
import os

import numpy as np
import torch
import triton
import triton.language as tl

# CPU-side bit-packing + lex-sort is implemented in a Rust extension by
# default (TensorState._TensorState_rs). The original Cython extension
# (TensorState._TensorState) is preserved in-tree for regression testing
# and selected via the TENSORSTATE_USE_CYTHON=1 environment variable.
_USE_CYTHON = os.environ.get("TENSORSTATE_USE_CYTHON", "0") == "1"
if _USE_CYTHON:
    import TensorState._TensorState as _ts  # type: ignore
else:
    try:
        import TensorState._TensorState_rs as _ts  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorState's Rust extension `_TensorState_rs` is not built. "
            "Install the package via `uv sync` / `pip install .` to build it, "
            "or set TENSORSTATE_USE_CYTHON=1 and build the Cython extension "
            "via `scripts/build-cython.sh` for regression testing."
        ) from exc

logging.basicConfig(
    format="%(asctime)s - %(name)-10s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("TensorState.States")


def _check_extension_functional() -> bool:
    """Smoke-test the CPU extension once at import time.

    Compresses a known float32 input and verifies the packed output. Guards
    against an importable-but-broken extension (e.g., a stale build or an
    ABI/signature mismatch). The result gates whether CPU torch tensors are
    routed through the extension or the slower torch-native fallback.
    """
    try:
        # Neurons 0 and 2 fire -> bit 0 and bit 2 set -> 0b00000101 = 5.
        probe = np.array([[1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        packed = _ts._compress_tensor_ps(probe)
        return tuple(packed.shape) == (1, 1) and int(packed[0, 0]) == 5
    except Exception:  # noqa: BLE001 -- any failure means "not functional"
        logger.warning(
            "CPU extension smoke test failed; CPU torch tensors will use the "
            "slower torch-native bit-pack fallback.",
            exc_info=True,
        )
        return False


_EXTENSION_FUNCTIONAL = _check_extension_functional()


@triton.jit
def _packbits_kernel_fused(
    input_ptr,
    out_ptr,
    n_rows,
    n_cols,
    out_cols,
    BLOCK_N: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Pack 8 input values (>0 firing) into each output uint8 byte.

    Fused variant: the thresholding (>0) happens inside the kernel, so
    the input can be float and no intermediate uint8 tensor is needed.
    Layout:
        input  shape (n_rows, n_cols)      float32
        out    shape (n_rows, out_cols)    uint8
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bytes_ = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)

    row_mask = rows < n_rows
    byte_mask = bytes_ < out_cols

    packed = tl.zeros((BLOCK_N, BLOCK_B), dtype=tl.uint8)

    for j in tl.static_range(8):
        cols = bytes_[None, :] * 8 + j
        load_mask = row_mask[:, None] & byte_mask[None, :] & (cols < n_cols)
        ptrs = input_ptr + rows[:, None] * n_cols + cols
        vals = tl.load(ptrs, mask=load_mask, other=0.0)
        bit = (vals > 0).to(tl.uint8)
        packed = packed | (bit << j)

    out_ptrs = out_ptr + rows[:, None] * out_cols + bytes_[None, :]
    tl.store(out_ptrs, packed, mask=row_mask[:, None] & byte_mask[None, :])


@triton.jit
def _packbits_kernel(
    bits_ptr,
    out_ptr,
    n_rows,
    n_cols,
    out_cols,
    BLOCK_N: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Pack 8 input bits into each output uint8 byte.

    Layout:
        bits  shape (n_rows, n_cols)        dtype uint8 (values 0 or 1)
        out   shape (n_rows, out_cols)      dtype uint8

    The thresholding (>0) is done in PyTorch before this kernel runs. This
    variant is used when the input is already pre-thresholded uint8 (e.g.,
    when a caller passes ``(states > 0).to(torch.uint8)`` directly), since
    1-byte loads are 4x cheaper than 4-byte float loads.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bytes_ = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)

    row_mask = rows < n_rows
    byte_mask = bytes_ < out_cols

    packed = tl.zeros((BLOCK_N, BLOCK_B), dtype=tl.uint8)

    for j in tl.static_range(8):
        cols = bytes_[None, :] * 8 + j
        load_mask = row_mask[:, None] & byte_mask[None, :] & (cols < n_cols)
        ptrs = bits_ptr + rows[:, None] * n_cols + cols
        vals = tl.load(ptrs, mask=load_mask, other=0).to(tl.uint8)
        packed = packed | (vals << j)

    out_ptrs = out_ptr + rows[:, None] * out_cols + bytes_[None, :]
    tl.store(out_ptrs, packed, mask=row_mask[:, None] & byte_mask[None, :])


def _compress_states_triton(states: torch.Tensor) -> torch.Tensor:
    """Bit-pack a 2D CUDA torch tensor using a Triton kernel.

    Picks the fused-threshold kernel for float dtypes (avoids an
    intermediate uint8 tensor) and the pre-thresholded kernel when the
    input is already uint8 / bool (saves bandwidth).

    Args:
        states: 2D CUDA tensor. Values > 0 are firing.

    Returns:
        2D uint8 CUDA tensor with shape ``(N, ceil(C / 8))``.
    """
    states = states.contiguous()
    n_rows, n_cols = states.shape
    out_cols = (n_cols + 7) // 8
    out = torch.empty((n_rows, out_cols), dtype=torch.uint8, device=states.device)

    BLOCK_N = 16
    BLOCK_B = 128
    grid = (triton.cdiv(n_rows, BLOCK_N), triton.cdiv(out_cols, BLOCK_B))

    if states.dtype in (torch.uint8, torch.bool):
        kernel = _packbits_kernel
        if states.dtype == torch.bool:
            states = states.to(torch.uint8)
    else:
        kernel = _packbits_kernel_fused

    kernel[grid](
        states,
        out,
        n_rows,
        n_cols,
        out_cols,
        BLOCK_N=BLOCK_N,
        BLOCK_B=BLOCK_B,
    )
    return out


def _compress_states_torch_cpu(states: torch.Tensor) -> torch.Tensor:
    """Bit-pack a CPU torch tensor using native torch ops.

    Used when the input is a torch tensor but on CPU. The Triton path is
    GPU-only; this is the CPU fallback when a torch tensor happens to be
    passed in.
    """
    bits = (states > 0).to(torch.uint8)
    n_rows, n_cols = bits.shape
    pad = (-n_cols) % 8
    if pad:
        bits = torch.nn.functional.pad(bits, (0, pad))
    bits = bits.reshape(n_rows, -1, 8)
    shifts = torch.arange(8, dtype=torch.uint8, device=bits.device)
    return (bits << shifts).sum(dim=-1).to(torch.uint8)


def _compress_states_torch(states: torch.Tensor) -> torch.Tensor:
    """Dispatch a torch tensor to the fastest available bit-pack path.

    - CUDA tensor: the Triton kernel.
    - CPU tensor, extension functional: threshold in torch (uniform across
      dtypes), then pack via the Rust/Cython extension (much faster than
      torch-native packing — see the AIQ-32 benchmarks).
    - CPU tensor, extension broken/missing: the torch-native fallback.
    """
    if states.is_cuda:
        return _compress_states_triton(states)
    if _EXTENSION_FUNCTIONAL:
        # `> 0` works for any numeric dtype; `.to(uint8)` makes a contiguous
        # 0/1 array the extension's pi8 path accepts. `.numpy()` is zero-copy
        # for a contiguous CPU tensor.
        bits = (states > 0).to(torch.uint8).detach().cpu().numpy()
        return torch.from_numpy(_ts._compress_tensor_pi8(bits))
    return _compress_states_torch_cpu(states)


def compress_states(states):
    """Compress a state space tensor.

    This function quantizes neurons into firing (>0) or non-firing (<=0), then
    compresses the bits into uint8 values. Thus, if a layer has 8 neurons in it,
    then the output is a raw byte ranging in value from 0-255. This compresses
    the statespace by 32x relative to holding values as floats, or by 8x
    relative to holding values as boolean. This is an important consideration
    since state space is large and grows exponentially with the number of
    neurons in the layer.

    Args:
        states: A 2d array of neuron outputs as ``numpy.float32`` /
            ``numpy.bool_`` values, or a ``torch.Tensor`` (on CPU or GPU).
            Rows are state observations, columns are neurons.

    Returns:
        A 2d array of uint8 values, where each value is the compressed
        representation of the state. The return type matches the input
        backend: ``numpy.ndarray`` for numpy input, ``torch.Tensor`` for
        torch input.
    """
    logger.debug("compress_states")

    if isinstance(states, np.ndarray):
        if states.dtype == np.float32:
            logger.debug("compress_states: _compress_tensor_ps")
            return _ts._compress_tensor_ps(states)
        if states.dtype == np.bool_:
            logger.debug("compress_states: _compress_tensor_pi8")
            return _ts._compress_tensor_pi8(states)
        raise TypeError("states must be numpy.float32 or numpy.bool_")
    if isinstance(states, torch.Tensor):
        logger.debug("compress_states: _compress_states_torch")
        return _compress_states_torch(states)
    raise TypeError(
        "states must be a numpy.ndarray (float32 or bool_) or a torch.Tensor"
    )


def sort_states(states, state_count):
    """Sort the states to place identical states next to each other.

    This function sorts the states stored in a 2d numpy.ndarray so that
    identical states are placed next to each other. To increase speed, the
    states are not actually sorted since moving data around in memory can be
    time consuming, and usually not useful. What is returned is a sorted index
    and the location of unique states in the sorted index.

    The CPU Cython lex-sort is used regardless of where the data originated.
    Sorting via GPU is not worth the round-trip cost for the buffer sizes
    typical of state capture workflows.

    Args:
        states: A 2d array of compressed states (numpy or torch). See
            ``compress_states`` function.
        state_count: The number of states (or number of rows to sort).

    Returns:
        A tuple containing bin edges, or locations of unique states, and a
        sorted index, which can be used to sort the input states using
        ``states[index]``.
    """
    logger.debug("sort_states: tensorstate._lex_sort")
    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()
    bin_edges, index = _ts._lex_sort(states, state_count)
    return bin_edges, index


def decompress_states(states, num_neurons):
    """Decompress states to numpy array of booleans.

    This functions takes a 2d numpy array of compressed neuron states and
    returns a boolean array of states, where each column of values represents
    the state of an individual neuron (firing=True, non-firing=False).

    For example, take a neuron layer with 5 neurons. The compressed state will
    be represented by a single byte, and if all but the first neuron is firing
    then the bits will be set as follows:

    ``'00011110'``

    To decompress this, the number of neurons needs to be input to know how many
    of the bits are actual neuron representations. When this state is
    decompressed, the numpy array will be:

    [False, True, True, True, True]

    Args:
        states: A 2d array of compressed states. See ``compress_states``
            function.
        num_neurons: The number of neurons in the layer.

    Returns:
        Boolean numpy array of neuron states
    """
    logger.debug("_decompress_tensor")
    return _ts._decompress_tensor(states, num_neurons)
