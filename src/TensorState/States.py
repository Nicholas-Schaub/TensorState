import logging

import numpy as np

import TensorState._TensorState as _ts
from TensorState import has_cupy

logging.basicConfig(
    format="%(asctime)s - %(name)-10s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("TensorState.States")

if has_cupy:
    import cupy

    # modified from cupy source
    # https://github.com/cupy/cupy/blob/v8.1.0/cupy/_binary/packing.py#L16
    _compress_kernel = cupy.ElementwiseKernel(
        "raw T myarray, raw int64 myarray_size, raw int64 in_cols, raw int64 out_cols, raw int64 stride",
        "uint8 packed",
        """
        long row = i / out_cols;
        long col = (i % out_cols) * stride;
        long k = row * in_cols + col;
        long nvals = (col + stride - 1 < in_cols) ? stride : in_cols - col;
        for (long j = 0; j < nvals; ++j) {
            int bit = myarray[k+j] != 0;
            packed |= bit << j;
        }""",
        "packbits_kernel",
    )

    # modified from cupy source
    # https://github.com/cupy/cupy/blob/v8.1.0/cupy/_binary/packing.py#L16
    def _compress_states_cuda(states):
        myarray = (states > 0).ravel()
        nrows = states.shape[0]
        ncols = (states.shape[1] + 7) // 8
        packed_size = nrows * ncols
        packed = cupy.zeros((packed_size,), dtype=cupy.uint8)
        stride = min([8, states.shape[1]])
        return _compress_kernel(
            myarray, myarray.size, states.shape[1], ncols, stride, packed
        ).reshape(nrows, ncols)


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
        states: A 2d array of neuron outputs as numpy.float32 or np.bool_
            values, where columns are a particular neuron's value, and rows are
            states.

    Returns:
        A 2d array of uint8 values, where each value is the compressed
            representation of the state.
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
    elif has_cupy and isinstance(states, cupy.ndarray):
        logger.debug("compress_states: _compress_states_cuda")
        return _compress_states_cuda(states)
    else:
        raise TypeError("states must be a numpy.ndarray")


def sort_states(states, state_count):
    """Sort the states to place identical states next to each other.

    This function sorts the states stored in a 2d numpy.ndarray so that
    identical states are placed next to each other. To increase speed, the
    states are not actually sorted since moving data around in memory can be
    time consuming, and usually not useful. What is returned is a sorted index
    and the location of unique states in the sorted index.

    Args:
        states: A 2d array of compressed states. See ``compress_states``
            function.
        state_count: The number of states (or number of rows to sort).

    Returns:
        A tuple containing bin edges, or locations of unique states, and a
            sorted index, which can be used to sort the input states using
            ``states[index]``
    """
    logger.debug("sort_states")
    if has_cupy:
        logger.debug("sort_states: cupy.lexsort")
        states = cupy.asarray(states[:state_count]).T
        index = cupy.lexsort(states)
        states = states[:, index]
        uniques = cupy.argwhere(cupy.any(states[:, :-1] != states[:, 1:], axis=0)) + 1
        bin_edges = cupy.zeros((uniques.size + 2,), dtype=np.int64)
        bin_edges[1:-1] = uniques.squeeze()
        bin_edges[-1] = states.shape[1]
        bin_edges = cupy.asnumpy(bin_edges)
        index = cupy.asnumpy(index)
    else:
        logger.debug("sort_states: tensorstate._lex_sort")
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
