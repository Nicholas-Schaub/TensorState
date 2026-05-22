"""State compression, decompression, and sorting primitives.

The CPU bit-packing and lex-sort routines are implemented in the Cython
extension ``_TensorState``. The GPU bit-packing path is a small pure-torch
implementation (no custom CUDA kernel) that uses standard tensor operations
to pack 8 bits into each uint8 byte.

The previous implementation routed GPU data through CuPy with a custom
``ElementwiseKernel`` and a CuPy ``lexsort``. CuPy has been removed; the
GPU path now uses PyTorch tensor ops natively, and CPU sorting always uses
the Cython ``_lex_sort`` (which is fast enough that round-tripping CPU
data through GPU just for sorting is not worthwhile).
"""

import logging

import numpy as np
import torch

import TensorState._TensorState as _ts

logging.basicConfig(
    format="%(asctime)s - %(name)-10s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("TensorState.States")


def _compress_states_torch(states: torch.Tensor) -> torch.Tensor:
    """Bit-pack a 2D torch tensor of neuron states into uint8 bytes.

    Neurons fire if their value is > 0. Each contiguous group of 8 neurons
    along the last dimension is packed into one uint8 byte. If the number of
    neurons is not a multiple of 8, the final byte is padded with zeros.

    Args:
        states: A 2D tensor where rows are state observations and columns
            are neurons. Any numeric dtype; values > 0 are treated as firing.

    Returns:
        A 2D ``torch.uint8`` tensor on the same device as ``states`` with
        shape ``(N, ceil(C / 8))``.
    """
    bits = (states > 0).to(torch.uint8)
    n_rows, n_cols = bits.shape
    pad = (-n_cols) % 8
    if pad:
        bits = torch.nn.functional.pad(bits, (0, pad))
    # Reshape into groups of 8 bits and combine via shift+sum.
    # For distinct bit positions sum is equivalent to bitwise OR, and uint8
    # cannot overflow since the per-byte maximum is 255.
    bits = bits.reshape(n_rows, -1, 8)
    shifts = torch.arange(8, dtype=torch.uint8, device=bits.device)
    packed = (bits << shifts).sum(dim=-1).to(torch.uint8)
    return packed


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
        elif states.dtype == np.bool_:
            logger.debug("compress_states: _compress_tensor_pi8")
            return _ts._compress_tensor_pi8(states)
        else:
            raise TypeError("states must be numpy.float32 or numpy.bool_")
    elif isinstance(states, torch.Tensor):
        logger.debug("compress_states: _compress_states_torch")
        return _compress_states_torch(states)
    else:
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
