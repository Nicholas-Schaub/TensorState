import os, queue, subprocess, argparse, sys, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow.keras as keras
import TensorState
import numpy as np
from pathlib import Path
import zarr
import TensorState._TensorState as ts
from concurrent.futures import ThreadPoolExecutor, wait

logging.basicConfig(format='%(asctime)s - %(name)-10s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger('TensorState')
logger.setLevel(logging.WARNING)

def network_efficiency(efficiencies):
    """Calculate the network efficiency

    This method calculates the neural network efficiency, defined
    as the geometric mean of the efficiency values calculated for
    the network.

    Args:
        efficiencies ([list,keras.Model]): A list of efficiency
            values (floats) or a keras.Model

    Returns:
        [float]: The network efficiency
    """
    # Get the efficiency values for the keras model
    if isinstance(efficiencies,keras.Model):
        efficiencies = [eff.efficiency() for eff in efficiencies.efficiency_layers]
    assert isinstance(efficiencies,list), 'Input must be list or keras.Model'
    
    # If the length of efficiencies is 0, return None and warn the user
    if len(efficiencies) == 0:
        logger.warning('List of efficiency values is empty. Verify input or model input to network_efficiency.')
        return None
    
    # Calculate the geometric mean of efficiencies
    net_efficiency = np.exp(sum(np.log(eff) for eff in efficiencies)/len(efficiencies)) # geometric mean
    return net_efficiency

def aIQ(net_efficiency,accuracy,weight):
    """Calculate the artificial intelligence quotient

    The artificial intelligence quotient (aIQ) is a simple metric to
    report a balance of neural network efficiency and task performance.
    Although not required, it is assumed that the accuracy argument
    is a float ranging from 0.0-1.0, with 1.0 meaning more accurate.
    
    aIQ = (net_efficiency * accuracy ** weight) ** (1/(weight+1))
    
    The weight argument is an integer, with higher values giving more
    weight to the accuracy of the model.

    Args:
        net_efficiency ([float]): A float ranging from 0.0-1.0
        accuracy ([float]): A float ranging from 0.0-1.0
        weight ([int]): An integer with value >=1

    Raises:
        ValueError: Raised if weight <= 0

    Returns:
        [float]: The artificial intelligence quotient
    """
    if weight <= 0 or not isinstance(weight,int):
        raise ValueError('aIQ weight must be an integer greater than 0.')
    aIQ = np.power(accuracy**weight * net_efficiency,1/(weight+1))
    return aIQ

def entropy(counts,alpha=1):
    """Calculate the Renyi entropy

    The Renyi entropy is a general definition of entropy that encompasses
    Shannon's entropy, Hartley (maximum) entropy, and min-entropy. It is
    defined as:
    
    ``(1-alpha)**-1 * log2( sum(p**alpha) )``
    
    By default, this method sets alpha=1, which is Shannon's entropy.

    Args:
        counts (list of ints): Count of each time a state is observed.
        alpha ([int,float], optional): Entropy order. Defaults to 1.

    Returns:
        [float]: The entropy of the count data.
    """
    num_microstates = counts.sum()
    frequencies = counts/num_microstates
    if alpha==1:
        entropy = (-frequencies * np.log2(frequencies)).sum()
    else:
        defaults = np.seterr(all='warn')
        try:
            entropy = 1/(1-alpha) * np.log2((frequencies**alpha).sum())
        except Warning:
            logger.warning('A warning was generated, like due to alpha being too large. Returning min entropy (alpha=inf)')
            entropy = -np.log2(np.max(frequencies))
        np.seterr(**defaults)
        
    return entropy

class StateCapture(keras.layers.Layer):
    """Tensorflow 2 layer to capture layer states.

    This layer quantizes inputs as firing or not firing based on
    whether the input values are >0 or <=0 respectively. Although
    this layer is intended to be attached before or after a neural
    layer, this can actually be attached to any layer type. After
    recording the firing state of all neurons, the original input
    is returned unaltered. Thus, this layer can be thought of as a
    "probe", since it does not add or subtract from the function of
    a network.
    
    Layer states are stored in a zarr array, which permits
    compressed storage of data in memory or on disk. Only blosc
    compression is used to ensure fast compression/decompression
    speeds. By default, data is stored in memory, but data can be
    stored on disk to reduce memory consumption by using the
    disk_path keyword.
    
    NOTE: This layer currently only works within Keras models.

    Args:
        name (string): Name of the state capture layer.
        disk_path ([imglib.Path,str], optional): Path on disk to save
            captured states in zarr format.
            Defaults to None.
        
    """
    
    _chunk_size = 0
    _state_shape = tuple()
    _entropy = None
    _state_count = 0
    
    @property
    def state_count(self):
        """The total number of observed states, including repeats."""
        self._wait_for_threads()
        return self._state_count
    
    @state_count.setter
    def state_count(self,count):
        self._state_count = count
    
    _states = None
    _index = None
    _edges = None
    _counts = None
    _input_shape = [None]
    _state_ids = None
    _threads = []
    _zarr_path = None
    
    def __init__(self,name,disk_path=None,**kwargs):
        super().__init__(name=name,**kwargs)
        self._executor = ThreadPoolExecutor(4)
        if disk_path != None:
            if not isinstance(disk_path,Path):
                self._zarr_path = Path(disk_path)
            else:
                self._zarr_path = disk_path
            self._zarr_path = self._zarr_path.joinpath('tensor_states')
            self._zarr_path.mkdir(exist_ok=True)
            self._zarr_path = self._zarr_path.joinpath(name + '.zarr')
            self._zarr_path.mkdir(exist_ok=False)
            
    def _wait_for_threads(self):
        wait(self._threads)
        self._threads = []

    def build(self,input_shape):
        """Build the StateCapture Keras Layer

        This method initializes the layer and resets any previously held
        data. The zarr array is initialized in this method.

        Args:
            input_shape (TensorShape): Either a TensorShape or list of
                TensorShape instances.
        """
        
        # Try to keep chunks limited to 16MB
        ncols = int(np.ceil(input_shape[-1]/8))
        nrows = 2**22 // ncols
        self._input_shape = input_shape
        
        self._state_ids = None
        self._edges = None
        self._index = None
        self._counts = None
        self._entropy = None
        self._threads = []
        
        self._chunk_size = (nrows,ncols)
        self._state_shape = list(self._chunk_size)
        self.state_count = 0
        if self._zarr_path != None:
            if self._zarr_path.is_file():
                self._zarr_path.unlink()
                
            self._states = zarr.zeros(shape=self._state_shape,chunks=self._chunk_size,dtype='B',
                                      synchronizer=zarr.ThreadSynchronizer(),
                                      store=str(self._zarr_path.absolute()))
        else:
            self._states = zarr.zeros(shape=self._state_shape,chunks=self._chunk_size,dtype='B',
                                      synchronizer=zarr.ThreadSynchronizer())
        
    def _compress_and_store(self,inputs):
        num_states = inputs.shape[0] * int(np.prod(inputs.shape[1:-1]))
        if 2*num_states + self._state_count >= self._states.shape[0]:
            self._state_shape[0] += self._chunk_size[0]
            self._states.resize(self._state_shape)
        self._states[self._state_count:self._state_count+num_states] = ts.compress_tensor(np.reshape(inputs,(-1,int(inputs.shape[-1]))))
        self._state_count += num_states
        
        return True

    def call(self, inputs):
        if inputs.shape[0] == None:
            return inputs

        # self._compress_and_store(inputs)
        self._threads.append(self._executor.submit(self._compress_and_store,inputs))
        
        return inputs
            
    def state_ids(self):
        """Identity of observed states

        This method returns a list of byte arrays. Each byte array
        corresponds to a unique observed state, where each bit in
        the byte array corresponds to a neuron. The list returned
        by this method matches the list returned by ``counts``, so
        that the value in ``state_ids`` at position i is associated
        with the ``counts`` value at position i.
        
        For example, if the StateCapture layer is attached to a
        convolutional layer with 8 neurons, then each item in the
        list will be a byte array of length 1. If one of the bytes
        is ``\\x00`` (a null byte), then the state is one where none
        of the neurons fire.
        
        NOTE: Only observed states are contained in the list.

        Returns:
            [list of Bytes]: Unique states observed by the layer
        """
        
        if self._state_ids == None:
            self.counts()
            self._state_ids = []
            states = self._states[:self.state_count,:].tobytes()
            for cindex in range(self._edges.shape[0]-1):
                start_index = int(self._index[self._edges[cindex]]*self._states.shape[1])
                end_index = start_index + self._states.shape[1]
                self._state_ids.append(states[start_index:end_index])
            
        return self._state_ids
    
    def counts(self):
        """Layer state counts

        This method returns a list of integers, where each integer is the
        number of times a state is observed. The identity of the states can
        be obtained by calling the ``state_ids`` method.
        
        NOTE: The list only contains counts for observed states, so all values
        will be >0

        Returns:
            [list of ``int``]: Counts of stat occurences
        """
        if not isinstance(self._counts,np.ndarray):
            # First make sure all storage threads are finished
            self._wait_for_threads()
            
            # Create the index and sort the data to find the bin edges
            self._edges,self._index = ts.lex_sort(self._states[:self.state_count,:],self.state_count)
            self._counts = np.diff(self._edges)
        
        return self._counts
    
    def max_entropy(self):
        """Theoretical maximum entropy for the layer

        The maximum entropy for the layer is equal to the number of neurons
        in the layer. This is different than the maximum entropy value that
        would be returned from the ``TensorState.entropy`` method with
        ``alpha=0``, which is a count of the observed states.

        Returns:
            [int]: Theoretical maximum entropy value
        """
        return self._input_shape[-1]
    
    def entropy(self,alpha=1):
        """Calculate the entropy of the layer

        Calculate the entropy from the observed states. The alpha value is
        the order of entropy calculated using the formula for Renyi entropy.
        
        Args:
            alpha (int): Order of entropy to calculate

        Returns:
            [float]: The entropy of the layer
        """
        self._entropy = entropy(self.counts(),alpha)
        return self._entropy

    def efficiency(self,alpha=1):
        """Calculate the efficiency of the layer

        This method returns the efficiency of the layer. Originally, the 
        efficiency was defined as the ratio of Shannon's entropy to the
        theoretical maximum entropy based on the number of neurons in the
        layer. This method with no inputs will return that value. However,
        this method will also now permit defining the alpha value for the
        Renyi entropy, so that the efficiency will be calculated as the
        Renyi entropy of order alpha divided by the maximum theoretical
        entropy.

        Args:
            alpha (float, optional): Order of Renyi entropy

        Returns:
            [float]: The efficiency of the layer
        """
        return self.entropy()/self.max_entropy()
        
def reset_efficiency_model(model):
    for layer in model.efficiency_layers:
        layer.build(layer._input_shape)

def build_efficiency_model(model,attach_to=['Conv2D','Dense'],method='after',storage_path=None):
    """Attach StateCapture layers a Keras model.

    This method takes an existing keras model in Tensorflow and attaches
    StateCapture layers to it, returning a new model. This is a simplified
    way to attach StateCapture layers to an existing model without affecting
    the function of the network.

    Args:
        model (keras.Model): A Keras model
        attach_to (list, optional): List of strings indicating the types of
            layers to attach to. Names of layers can also be specified
            to attach StateCapture to specific layers
            Defaults to ['Conv2D','Dense'].
        method (str, optional): The location to attach the StateCapture layer
            to. Must be one of ['before','after','both'].
            Defaults to 'after'.
        storage_path ([str,pathlib.Path], optional): Path on disk to store states
            in zarr format. If None, states are stored in memory.
            Defaults to None.

    Returns:
        keras.Model: A Keras Model with StateCapture layers attached to it.
    """
    
    # Check to make sure method is valid
    assert method in ['before','after','both'], 'Method must be one of [before,after,both]'

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    efficiency_layers = []
    for layer in model.layers[1:]:
    
        # Determine input tensorss
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Add layer before if requested
        if method in ['before','both']\
            and (layer.__class__.__name__ in attach_to or layer.name in attach_to):
                
            efficiency_layer = StateCapture(name=network_dict['input_layers_of'][layer.name][0]+'_states',disk_path=storage_path)
            efficiency_layers.append(efficiency_layer)
            layer_input = efficiency_layer(layer_input)
            
            network_dict['new_output_tensor_of'].update({network_dict['input_layers_of'][layer.name][0]: layer_input})

        x = layer(layer_input)
        
        if method in ['after','both']\
            and (layer.__class__.__name__ in attach_to or layer.name in attach_to):
                
            efficiency_layer = StateCapture(name=layer.name+'_states',disk_path=storage_path)
            efficiency_layers.append(efficiency_layer)
            x = efficiency_layer(x)
            
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer.name in model.output_names:
            model_outputs.append(x)

    new_model = keras.Model(inputs=model.inputs, outputs=model_outputs)
    new_model._is_graph_network = True
    new_model._init_graph_network(inputs=model.inputs, outputs=model_outputs)
    new_model._run_eagerly = True
    new_model.efficiency_layers = efficiency_layers

    return new_model