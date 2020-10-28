import TensorState._TensorState as ts
import numpy as np

def compress_states(states):
    """Compress a state space tensors
    
    This function quantizes neurons into firing (>0) or non-firing (<=0), then
    compresses the bits into uint8 values. Thus, if a layer has 8 neurons in it,
    then the output is a raw byte ranging in value from 0-255. This compresses
    the statespace by 32x relative to holding values as floats, or by 8x
    relative to holding values as boolean. This is an important consideration
    since state space is large and grows exponentially with the number of
    neurons in the layer.

    Args:
        states ([numpy.ndarray]): A 2d array of neuron outputs as numpy.float32
            or np.bool_ values, where columns are a particular neuron's value,
            and rows are states.
    
    Returns:
        numpy.ndarray: A 2d array of uint8 values, where each value is the
            compressed representation of the state.
    """
    
    if isinstance(states,np.ndarray):
        if states.dtype == np.float32:
            return ts._compress_tensor_ps(states)
        elif states.dtype == np.bool_:
            return ts._compress_tensor_pi8(states)
        else:
            raise TypeError('states must be numpy.float32 or numpy.bool_')
    else:
        raise TypeError('states must be a numpy.ndarray')

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

def decompress_states(states,num_neurons):
    """Decompress states to numpy array of booleans
    
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
        states ([numpy.ndarray]): A 2d array of compressed states.
            See ``compress_states`` function.
        num_neurons ([int]): The number of neurons in the layer.
    
    Returns:
        decompressed_states ([np.ndarray]): Boolean numpy array of neuron states
        
    """
    
    return ts._decompress_tensor(states,num_neurons)