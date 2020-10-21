import TensorState._TensorState as ts

def compress_states(states):
    """Compress a state space tensors
    
    This function quantizes neurons into firing (>0) or non-firing (<=0), then
    compresses the bits into uint8 values. Thus, if a layer has 8 neurons in it,
    then the output is a raw byte ranging in value from 0-255. This compresses
    the statespace by 32x relative to holding values as floats, or by 8x
    relative to holding values as boolean. This is an important consideration
    considering the immense size of state space, which grows exponentially with
    the number of neurons in the layer.

    Args:
        states ([numpy.ndarray]): A 2d array of neuron outputs as float32
            values, where columns are a particular neuron's value, and rows are
            states.
    
    Returns:
        numpy.ndarray: A 2d array of uint8 values, where each value is the
            compressed representation of the state.
    """
    
    return ts._compress_tensor(states)

def sort_states(states,state_count):
    """Sort the states to place identical states next to each other
    
    This function sorts the states stored in a 2d numpy.ndarray so that
    identical states are placed next to each other. To increase speed, the
    states are not actually sorted since moving data around in memory can be
    time consuming, and usually not useful. What is returned is a sorted index
    and the location of unique states in the sorted index.

    Args:
        states ([numpy.ndarray]): A 2d array of compressed states.
            See ``compress_states`` function.
        state_count ([int]): The number of states (or number of rows to sort).
    
    Returns:
        edges ([np.ndarray]): Bin edges, or locations of unique states
        index ([np.ndarray]): Sorted index. This output can be used to actually
            sort the input states by doing ``states[index]``
    """
    
    return ts._lex_sort(states,state_count)