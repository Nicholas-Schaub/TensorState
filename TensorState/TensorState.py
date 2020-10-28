import os, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from TensorState.Layers import StateCapture, StateCaptureHook
import numpy as np

logging.basicConfig(format='%(asctime)s - %(name)-10s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger('TensorState')
logger.setLevel(logging.WARNING)

def network_efficiency(efficiencies):
    """Calculate the network efficiency

    This method calculates the neural network efficiency, defined as the
    geometric mean of the efficiency values calculated for the network.

    Args:
        efficiencies ([list,keras.Model,torch.nn.Module]): A list of efficiency
            values (floats) or a keras.Model

    Returns:
        [float]: The network efficiency
    """
    # Get the efficiency values for the keras model
    if hasattr(efficiencies,'efficiency_layers'):
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

    The artificial intelligence quotient (aIQ) is a simple metric to report a
    balance of neural network efficiency and task performance. Although not
    required, it is assumed that the accuracy argument is a float ranging from
    0.0-1.0, with 1.0 meaning more accurate.
    
    aIQ = (net_efficiency * accuracy ** weight) ** (1/(weight+1))
    
    The weight argument is an integer, with higher values giving more weight to
    the accuracy of the model.

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
    Shannon's entropy, Hartley (maximum) entropy, and min-entropy. It is defined
    as:
    
    ``(1-alpha)**-1 * log2( sum(p**alpha) )``
    
    By default, this method sets alpha=1, which is Shannon's entropy.

    Args:
        counts (numpy.ndarray): Array of counts representing number of times a
            state is observed.
        alpha ([int,float], optional): Entropy order. Defaults to 1.

    Returns:
        [float]: The entropy of the count data.
    """
    num_microstates = counts.sum()
    frequencies = counts/num_microstates
    if alpha==1:
        entropy = (-frequencies * np.log2(frequencies)).sum()
    else:
        entropy = 1/(1-alpha) * np.log2((frequencies**alpha).sum())
        
    return entropy
        
def reset_efficiency_model(model):
    """Reset all efficiency layers/hooks in a model
    
    This method resets all efficiency layers or hooks in a model, setting the
    ``state_count=0``. This is useful for repeated evaluation of a model
    during a single session.

    Args:
        model ([keras.Model, torch.nn.Module]): Model to reset
    """
    
    for layer in model.efficiency_layers:
        layer.reset_states()

def _pt_efficiency_model(model,attach_to,exclude,method,storage_path):

    model.efficiency_layers = []
    
    layer_ids = {id(module):(module.__class__.__name__, None, module) for module in model.modules()}
    layer_ids.update({id(module):(module.__class__.__name__, name, module) for name,module in model.named_modules()})

    for cls_name,mod_name,module in layer_ids.values():
        
        if (cls_name not in attach_to or mod_name in exclude) and mod_name not in attach_to:
            continue
        
        # Add pre-hook if requested
        if method in ['before','both']:
                        
            efficiency_layer = StateCaptureHook(name=str(mod_name)+'_pre_states',disk_path=storage_path)
            model.efficiency_layers.append(efficiency_layer)
            
            module.register_forward_pre_hook(efficiency_layer)
        
        if method in ['after','both']:
            
            efficiency_layer = StateCaptureHook(name=str(mod_name)+'_post_states',disk_path=storage_path)
            model.efficiency_layers.append(efficiency_layer)
            
            module.register_forward_hook(efficiency_layer)

    return model

def _tf_efficiency_model(model,attach_to,exclude,method,storage_path):
    import tensorflow.keras as keras

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
            and (layer.__class__.__name__ in attach_to or layer.name in attach_to) \
            and (layer.__class__.__name__ not in exclude or layer.name in exclude):
                
            efficiency_layer = StateCapture(name=network_dict['input_layers_of'][layer.name][0]+'_pre_states',disk_path=storage_path)
            efficiency_layers.append(efficiency_layer)
            layer_input = efficiency_layer(layer_input)
            
            network_dict['new_output_tensor_of'].update({network_dict['input_layers_of'][layer.name][0]: layer_input})

        # Process layer
        x = layer(layer_input)
        
        # Add layer after if requested
        if method in ['after','both']\
            and (layer.__class__.__name__ in attach_to or layer.name in attach_to) \
            and (layer.__class__.__name__ not in exclude or layer.name in exclude):
                
            efficiency_layer = StateCapture(name=layer.name+'_post_states',disk_path=storage_path)
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

def build_efficiency_model(model,attach_to,exclude=[],method='after',storage_path=None):
    """Attach state capture methods to a neural network

    This method takes an existing neural network model and attaches either
    layers or hooks to the model to capture the states of neural network layers.

    For Tensorflow, only keras.Model networks can serve as inputs to this
    function. When a Tensorflow model is fed into this function, a new network
    is returned where StateCapture layers are inserted into the network at the
    designated locations.

    For PyTorch, a neural network that implements the Module class will have
    hooks added to the layers. A new network is not generated, but for
    consistency the model is returned from this function.

    Args:
        model ([keras.Model, torch.nn.Module]): A Keras model
        attach_to (list, optional): List of strings indicating the types of
            layers to attach to. Names of layers can also be specified
            to attach StateCapture to specific layers
        exclude (list, optional): List of strings indicating the names of layers
            to not attach StateCapture layers to. This will override the
            attach_to keyword, so that a Conv2D layer with the name specified
            by exclude will not have a StateCapture layer attached to it.
            Defaults to [].
        method (str, optional): The location to attach the StateCapture layer
            to. Must be one of ['before','after','both'].
            Defaults to 'after'.
        storage_path ([str, pathlib.Path], optional): Path on disk to store
            states in zarr format. If None, states are stored in memory.
            Defaults to None.

    Returns:
        model: A model of the same type as the input model
    """
    
    class_module = {cls.__module__:cls.__name__ for cls in model.__class__.__bases__}
    
    # Validate input arguments
    assert isinstance(attach_to,(list,str)) and len(attach_to) > 0
    assert method in ['before','after','both'], 'Method must be one of [before,after,both]'

    if isinstance(exclude,str):
        exclude = [exclude]
    assert isinstance(exclude,list)
    
    if class_module.get('tensorflow.python.keras.engine.network') == 'Network':
        new_model = _tf_efficiency_model(model,attach_to,exclude,method,storage_path)
    elif class_module.get('torch.nn.modules.module') == 'Module':
        new_model = _pt_efficiency_model(model,attach_to,exclude,method,storage_path)

    return new_model