import zarr, abc, logging

from concurrent.futures import ThreadPoolExecutor, wait
import TensorState.States as ts
import TensorState
from pathlib import Path
import numpy as np

class AbstractStateCapture(abc.ABC):
    """Base class for capturing state space information in a neural network.

    This class implements the infrastructure used to capture, quantize, and
    process state space information. For Tensorflow, a subclass is constructed
    to inherit these methods as a layer to be inserted into the network. For
    PyTorch, a subclass is constructed to implement these methods as layer
    hooks.

    This class captures state information and quantizes layer outputs as firing
    or not firing based on whether the values are >0 or <=0 respectively.
    Although this layer is intended to be attached before or after a neural
    layer, this can actually be attached to any layer type. After recording the
    firing state of all neurons, the original input is returned unaltered. Thus,
    this layer can be thought of as a "probe", since it does not add or subtract
    from the function of a network.
    
    Layer states are stored in a zarr array, which permits compressed storage of
    data in memory or on disk. Only blosc compression is used to ensure fast
    compression/decompression speeds. By default, data is stored in memory, but
    data can be stored on disk to reduce memory consumption by using the
    disk_path keyword.
    
    NOTE: This layer currently only works within Tensorflow Keras models and
    PyTorch models.
        
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
    def state_count(self,value):
        raise AttributeError('state_count attribute is read-only.')
    
    @property
    def states(self):
        """Decompressed state data"""
        self._wait_for_threads()
        if not isinstance(self._states,np.ndarray):
            self._states = ts.decompress_states(self.raw_states[self._index[self._edges[:-1]],:],
                                                int(self.max_entropy()))
        
        return self._states
    
    @states.setter
    def states(self,value):
        raise AttributeError('states attribute is read-only.')
    
    @property
    def raw_states(self):
        """Raw state data as stored in memory, bit compressed"""
        self._wait_for_threads()
        return self._raw_states.oindex
    
    @raw_states.setter
    def raw_states(self,value):
        raise AttributeError('states attribute is read-only.')
    
    _states = None
    _raw_states = None
    _index = None
    _edges = None
    _counts = None
    _input_shape = None
    _state_ids = None
    _threads = []
    _zarr_path = None
    _channel_index = -1
    
    def __init__(self,name,disk_path=None,**kwargs):
        """ Default initialization

        Args:
            name (string): Name of the state capture layer.
            disk_path ([imglib.Path,str], optional): Path on disk to save captured
                states in zarr format.
                Defaults to None.

            **kwargs: Keyword arguments. Used for passing arguments to other
                classes that inerit from AbstractStateCapture.
        """
        
        self._executor = ThreadPoolExecutor(2)
        
        # Assign a name to the layer. Some inheriting classes make name
        # protected, so catch the error just in case.
        try:
            self.name = name
        except AttributeError:
            pass

        # Set up zarr, but don't create anything
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
        
    def _compress_and_store(self,inputs):
        # Calculate the number of states to process
        num_states = inputs.shape[0] * int(np.prod(inputs.shape[1:-1]))
        
        # Resize the zarr array if needed
        if 2*num_states + self._state_count >= self._raw_states.shape[0]:
            self._state_shape[0] += self._chunk_size[0]
            self._raw_states.resize(self._state_shape)
        
        # Compress and store the states
        self._raw_states[self._state_count:self._state_count+num_states] = ts.compress_states(np.reshape(inputs,(-1,int(inputs.shape[-1]))))
        self._state_count += num_states
        
        # Reset the _counts and _state_ids so they are recalculated
        self._counts = None
        self._state_ids = None
        self._states = None
        
        return True

    def reset_states(self,input_shape=None):
        """Initialize the state space

        This method initializes the layer and resets any previously held data.
        The zarr array is initialized in this method.

        Args:
            input_shape (TensorShape,tuple, list): Shape of the input.
        """
        
        if not isinstance(input_shape, type(None)):
            self._input_shape = input_shape
            
        if self._input_shape == None:
            raise ValueError('The input_shape is None, and no previous input ' +
                             'shape information was provided. The first time ' +
                             'reset_states is called, an input_shape must be ' +
                             'provided.')

        # Try to keep chunks limited to 16MB
        ncols = int(np.ceil(self._input_shape[self._channel_index]/8))
        nrows = 2**22 // ncols
        
        # Initialize internal variables related to state space
        self._state_ids = None
        self._edges = None
        self._index = None
        self._counts = None
        self._entropy = None
        self._threads = []
        self._chunk_size = (nrows,ncols)
        self._state_shape = list(self._chunk_size)
        self._state_count = 0
        
        if self._raw_states != None:
            # Zero out states and resize if zarr already open
            self._raw_states.resize(self._state_shape)
            self._raw_states[:] = 0
        else:
            # Initialize the zarr array
            if self._zarr_path != None:
                if self._zarr_path.is_file():
                    self._zarr_path.unlink()
                    
                self._raw_states = zarr.zeros(shape=self._state_shape,chunks=self._chunk_size,dtype='B',
                                              synchronizer=zarr.ThreadSynchronizer(),
                                              store=str(self._zarr_path.absolute()))
            else:
                self._raw_states = zarr.zeros(shape=self._state_shape,chunks=self._chunk_size,dtype='B',
                                              synchronizer=zarr.ThreadSynchronizer())
            
    def state_ids(self):
        """Identity of observed states

        This method returns a list of byte arrays. Each byte array corresponds
        to a unique observed state, where each bit in the byte array corresponds
        to a neuron. The list returned by this method matches the list returned
        by ``counts``, so that the value in ``state_ids`` at position i is 
        associated with the ``counts`` value at position i.
        
        For example, if the StateCapture layer is attached to a convolutional
        layer with 8 neurons, then each item in the list will be a byte array of
        length 1. If one of the bytes is ``\\x00`` (a null byte), then the state
        has no firing neurons.
        
        NOTE: Only observed states are contained in the list.

        Returns:
            [list of Bytes]: Unique states observed by the layer
        """
        
        if not isinstance(self._state_ids,list):
            self.counts()
            self._state_ids = []
            states = self.raw_states[self._index[self._edges[:-1]],:].tobytes()
            delta = int((self.max_entropy()-1)//8 + 1)
            for cindex in range(0,delta * (self._edges.shape[0]-1),delta):
                self._state_ids.append(states[cindex:cindex+delta])
            
        return self._state_ids
    
    def counts(self):
        """Layer state counts

        This method returns a numpy.array of integers, where each integer is the
        number of times a state is observed. The identity of the states can be
        obtained by calling the ``state_ids`` method.
        
        NOTE: The list only contains counts for observed states, so all values
        will be >0

        Returns:
            [list of ``int``]: Counts of stat occurences
        """
        
        if not isinstance(self._counts,np.ndarray):
                        
            # Create the index and sort the data to find the bin edges
            self._edges,self._index = ts.sort_states(self.raw_states[:self.state_count,:],
                                                     self.state_count)
            self._counts = np.diff(self._edges)
        
        return self._counts
    
    def max_entropy(self):
        """Theoretical maximum entropy for the layer

        The maximum entropy for the layer is equal to the number of neurons in
        the layer. This is different than the maximum entropy value that would
        be returned from the ``TensorState.entropy`` method with ``alpha=0``,
        which is a count of the observed states.

        Returns:
            [float]: Theoretical maximum entropy value
        """
        return float(self._input_shape[self._channel_index])
    
    def entropy(self,alpha=1):
        """Calculate the entropy of the layer

        Calculate the entropy from the observed states. The alpha value is the
        order of entropy calculated using the formula for Renyi entropy. When
        alpha=1, this returns Shannon's entropy.
        
        Args:
            alpha (int, None): Order of entropy to calculate. If ``None``, then
                use ``max_entropy()``

        Returns:
            [float]: The entropy of the layer
        """
        
        if alpha==None:
            return self.max_entropy()
        else:
            return TensorState.entropy(self.counts(),alpha)

    def efficiency(self,alpha1=1,alpha2=None):
        """Calculate the efficiency of the layer

        This method returns the efficiency of the layer. Originally, the 
        efficiency was defined as the ratio of Shannon's entropy to the
        theoretical maximum entropy based on the number of neurons in the layer.
        This method with no inputs will return that value. However, this method
        will also now permit defining the alpha value for the Renyi entropy, so
        that the efficiency will be calculated as the Renyi entropy of order
        alpha1 divided by the maximum theoretical entropy.

        Args:
            alpha1 ([float, int], optional): Order of Renyi entropy in numerator
            alpha2 ([float, int, None], optional): Order of Renyi entropy in
                denominator

        Returns:
            [float]: The efficiency of the layer
        """
        
        assert isinstance(alpha1,(float,int)), 'alpha1 must be a float or int'
        assert isinstance(alpha2,(float,int,None.__class__)), 'alpha2 must be a\
                                                           float, int, or None'
        if alpha2 != None:
            assert alpha1>alpha2, 'alpha1 must be larger than alpha 2'

        return self.entropy(alpha1)/self.entropy(alpha2)

try:
    import tensorflow.keras as keras
    
    class StateCapture(keras.layers.Layer,AbstractStateCapture):
        """Tensorflow keras layer to capture states in keras models
        
        This class is designed to be used in a Tensorflow keras model to
        automate the capturing of neurons states as data is passed through the
        network.
            
        """
        def __init__(self,name,disk_path=None,**kwargs):
            # Use both parent class initializers
            keras.layers.Layer.__init__(self,name=name,**kwargs)
            AbstractStateCapture.__init__(self,name,disk_path=disk_path,**kwargs)
            
        def call(self, inputs):
            if inputs.shape[0] == None:
                return inputs

            self._threads.append(self._executor.submit(self._compress_and_store,inputs))
            
            return inputs
            
        def build(self,input_shape):
            """Build the StateCapture Keras Layer

            This method initializes the layer and resets any previously held
            data. The zarr array is initialized in this method.

            Args:
                input_shape (TensorShape): Either a TensorShape or list of
                    TensorShape instances.
            """
            
            self.reset_states(input_shape)

except ModuleNotFoundError:
    
    class StateCapture(AbstractStateCapture):
        """Tensorflow keras layer to capture states in keras models

        This class is designed to be used in a Tensorflow keras model to
        automate the capturing of neurons states as data is passed through the
        network.
            
        """
            
        def __init__(self,name,disk_path=None,**kwargs):
            raise ModuleNotFoundError('StateCapture class is unavailable since'+
                                      ' tensorflow was not found.')
            
        def call(self, inputs):
            if inputs.shape[0] == None:
                return inputs

            self._threads.append(self._executor.submit(self._compress_and_store,inputs))
            
            return inputs
            
        def build(self,input_shape):
            """Build the StateCapture Keras Layer

            This method initializes the layer and resets any previously held
            data. The zarr array is initialized in this method.

            Args:
                input_shape (TensorShape): Either a TensorShape or list of
                    TensorShape instances.
            """
            
            self.reset_states(input_shape)

try:
    import torch
    
    class StateCaptureHook(AbstractStateCapture):
        """StateCapture hook for PyTorch

        This class implements all methods in AbstractStateCapture, but is
        designed to be a pre or post hook for a layer.
        
        """

        def __init__(self,name,disk_path=None,**kwargs):
            # Use both parent class initializers
            super().__init__(name,disk_path,**kwargs)
            
            self._channel_index = 1

        def __call__(self,*args):
            
            if self._input_shape == None:
                self.reset_states(tuple(args[-1].shape))

            # Transform the tensor to have similar memory format as Tensorflow
            inputs = args[-1].permute(0,2,3,1).contiguous()
            
            # Store the data using a thread
            self._threads.append(self._executor.submit(self._compress_and_store,inputs.detach().cpu().numpy()))

except ModuleNotFoundError:
    
    class StateCaptureHook(AbstractStateCapture):
        """StateCapture hook for PyTorch

        This class implements all methods in AbstractStateCapture, but is
        designed to be a pre or post hook for a layer.
        
        """
            
        def __init__(self,name,disk_path=None,**kwargs):
            raise ModuleNotFoundError('StateCaptureHook class is unavailable' +
                                      ' since torch was not found.')