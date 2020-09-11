import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow.keras as keras
import TensorState
import numpy as np
from pathlib import Path
import zarr
import TensorState._TensorState as ts

def network_efficiency(efficiencies):
    """network_efficiency Calculate the network efficiency

    [extended_summary]

    Args:
        efficiencies ([type]): [description]

    Returns:
        [type]: [description]
    """
    if isinstance(efficiencies,keras.Model):
        efficiencies = [eff.efficiency() for eff in efficiencies.efficiency_layers]
    assert isinstance(efficiencies,list), 'Input must be list or keras.Model'
    net_efficiency = np.exp(sum(np.log(eff) for eff in efficiencies)/len(efficiencies)) # geometric mean
    return net_efficiency

def aIQ(net_efficiency,accuracy,weight):
    if weight <= 0:
        raise ValueError('aIQ weight must be greater than 0.')
    aIQ = np.power(accuracy**weight * net_efficiency,1/(weight+1))
    return aIQ

def layer_entropy(counts):
    num_microstates = counts.sum()
    frequencies = counts/num_microstates
    entropy = (-frequencies * np.log2(frequencies)).sum()
    return entropy

class TensorState(keras.layers.Layer):
    
    _chunk_size = 0
    _state_shape = tuple()
    _entropy = None
    state_count = 0
    _states = None
    _index = None
    _edges = None
    _counts = None
    _input_shape = [None]
    _bins = None
    
    def __init__(self,name=None,disk_path=None,**kwargs):
        super(TensorState, self).__init__(name=name,**kwargs)

    def build(self,input_shape):
        # Try to keep chunks limited to 16MB
        ncols = int(np.ceil(input_shape[-1]/8))
        nrows = 2**22 // ncols
        self._input_shape = input_shape
        
        self._bins = None
        self._edges = None
        self._index = None
        self._counts = None
        self._entropy = None
        
        self._chunk_size = (nrows,ncols)
        self._state_shape = list(self._chunk_size)
        self.state_count = 0
        self._states = zarr.zeros(self._state_shape,chunks=self._chunk_size,dtype='B')

    def call(self, inputs):
        if inputs.shape[0] == None:
            instances = 0
        else:
            instances = inputs.shape[0]
        num_states = instances * int(np.prod(inputs.shape[1:-1]))
        if instances > 0:
            if num_states + self.state_count >= self._states.shape[0]:
                self._state_shape[0] += 101*num_states
                self._states.resize(self._state_shape)
            self._states[self.state_count:self.state_count+num_states] = ts.compress_tensor(np.reshape(inputs,(-1,int(inputs.shape[-1]))))
            self.state_count += num_states
        return inputs
    
    def iterate_states(self,num_states=None):
        if num_states == None:
            num_states = self.chunk_size
        else:
            assert num_states>0, "num_states must be greater than 0"
        
        for index in range(0,self.state_count,num_states):
            end_index = max([index+num_states,self.state_count])
            yield self._states[index:end_index]
            
    def bins(self):
        if self._bins == None:
            # Create a dictionary to hold the bin counts
            self._bins = {}
            states = self._states[:self.state_count,:].tobytes()
            counts = self.counts()
            for cindex in range(len(counts)):
                start_index = int(self._index[self._edges[cindex]]*self._states.shape[1])
                end_index = start_index + self._states.shape[1]
                self._bins[states[start_index:end_index]] = counts[cindex]
            
        return self._bins
    
    def counts(self):
        if not isinstance(self._counts,np.ndarray):
            # Create the index and sort the data to find the bin edges
            start = time.time()
            self._edges,self._index = ts.lex_sort(self._states[:self.state_count,:],self.state_count)
            start = time.time()
            self._counts = np.diff(self._edges)
        
        return self._counts
    
    def max_entropy(self):
        return self._input_shape[-1]
    
    def entropy(self):
        if self._entropy == None:
            self._entropy = layer_entropy(self.counts())
        return self._entropy
    
    def efficiency(self):
        return self.entropy()/self.max_entropy()
        
def reset_efficiency_model(model):
    for layer in model.efficiency_layers:
        layer.build(layer._input_shape)

def build_efficiency_model(model,attach_to=['Conv2D','Dense'],method='after'):
    
    # Check to make sure method is valid
    assert method in ['before','after','both'], 'Method must be one of [before,after,both]'

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        # Skip input layers
        if (layer.__class__.__name__ == 'InputLayer'):
            continue
        
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)
        for node in layer._inbound_nodes:
            if node.inbound_layers.__class__.__name__ != 'InputLayer':
                continue
            layer_name = node.inbound_layers.name
            if layer.name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer.name: [layer_name]})
            else:
                network_dict['input_layers_of'][layer.name].append(layer_name)
            network_dict['new_output_tensor_of'][layer_name] = node.input_tensors

    # Iterate over all layers after the input
    model_outputs = []
    efficiency_layers = []
    for layer in model.layers:
        
        # Skip input layers
        if (layer.__class__.__name__ == 'InputLayer'):
            continue
    
        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Add layer before if requested
        if method in ['before','both']\
            and (layer.__class__.__name__ in attach_to or layer.name in attach_to):
                
            efficiency_layer = TensorState(name=network_dict['input_layers_of'][layer.name][0]+'_states')
            efficiency_layers.append(efficiency_layer)
            layer_input = efficiency_layer(layer_input)
            
            network_dict['new_output_tensor_of'].update({network_dict['input_layers_of'][layer.name][0]: layer_input})

        x = layer(layer_input)
        
        if method in ['after','both']\
            and (layer.__class__.__name__ in attach_to or layer.name in attach_to):
                
            efficiency_layer = TensorState(name=layer.name+'_states')
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