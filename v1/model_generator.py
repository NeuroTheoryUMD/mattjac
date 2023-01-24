# Wrapper model
import model as m
import hyperparameter_explorer as hpexplorer

# NDN model
import NDNT.NDNT as NDN
from NDNT.modules.layers import *
from NDNT.networks import *

from enum import Enum


# enums to make things easier to remember
class Explorer(Enum):
    # assign enums to functions
    sequential = hpexplorer.sequential
    grid = hpexplorer.grid


# private methods
# prefix with __ to make it private
####### Model --> NDN converters
def __NetworkParams_to_Network(template_network, prev_created_network, model_params, verbose):
    node = template_network
    
    if isinstance(node, m.Network):
        # get params for this network given the node index
        network_params = model_params[node.index]

        # make its layers from the provided configuration
        layers = []
        for li, (layer, layer_params) in enumerate(zip(node.layers, network_params)):
            # get params to pass
            layer_params_map = {k:v for k,v in layer_params}
            # create new layer of the given type
            layer = type(layer)()
            layer.params = layer_params_map
            layers.append(layer)

        # copy the network properties
        new_network = m.Network(layers=layers, name=node.name, network_type=node.network_type)
        new_network.index = node.index
        new_network.input_covariate = node.input_covariate
        
        if verbose:
            print('Network', new_network)
        # Network only has one input, so this is OK
        new_network.inputs = [_traverse_and_build(node.inputs[0], new_network, model_params, verbose)]
        desired_network = new_network

    elif isinstance(node, m.Add):
        # get params for this network given the node index
        network_params = model_params[node.index]
        
        # create the Add network
        add_network = m.Add()
        add_network.index = node.index
        # make its layers from the provided configuration
        layers = []
        for li, (layer, layer_params) in enumerate(zip(node.layers, network_params)):
            # get params to pass
            layer_params_map = {k:v for k,v in layer_params}
            # create new layer of the given type
            layer = type(layer)()
            layer.params = layer_params_map
            layers.append(layer)
        add_network.layers = layers # copy the layers over

        if verbose:
            print('Add', add_network)
        add_network.inputs = [_traverse_and_build(prev_in, add_network, model_params, verbose) for prev_in in node.inputs]
        add_network.name = '+'.join(inp.name for inp in add_network.inputs) # TODO: this is kind of a hack
        desired_network = add_network

    elif isinstance(node, m.Mult):
        # get params for this network given the node index
        network_params = model_params[node.index]
        
        # create the Mult network
        mult_network = m.Mult()
        mult_network.index = node.index
        # make its layers from the provided configuration
        layers = []
        for li, (layer, layer_params) in enumerate(zip(node.layers, network_params)):
            # get params to pass
            layer_params_map = {k:v for k,v in layer_params}
            # create new layer of the given type
            layer = type(layer)()
            layer.params = layer_params_map
            layers.append(layer)
        mult_network.layers = layers # copy the layers over

        if verbose:
            print('Mult', mult_network)
        mult_network.inputs = [_traverse_and_build(prev_in, mult_network, model_params, verbose) for prev_in in node.inputs]
        mult_network.name = '+'.join(inp.name for inp in mult_network.inputs) # TODO: this is kind of a hack
        desired_network = mult_network

    elif isinstance(node, m.Input):
        # create the Input network
        input_network = m.Input(node.covariate, node.input_dims)
        if verbose:
            print('Input', input_network)
        # input doesn't have any inputs to build, it is the base case
        desired_network = input_network

    else: # isinstance(node, m.Output):
        # create the Output network
        output_network = m.Output(node.num_neurons)
        if verbose:
            print('Output', output_network)
        # Output only has one input, so this is OK
        output_network.inputs = [_traverse_and_build(node.inputs[0], output_network, model_params, verbose)]
        desired_network = output_network

    if prev_created_network is not None:
        desired_network.output = prev_created_network # set the prev_layer's output
    return desired_network


# traverse the template_model networks and build new networks
# this will recursively build the expression tree from output to inputs
def _traverse_and_build(cur_template_network, prev_created_network, model_params, verbose):
    if verbose:
        if prev_created_network is None:
            print(cur_template_network.name, '-->', 'None')
        elif cur_template_network is None:
            print('None', '-->', prev_created_network.name)
        else:
            print(cur_template_network.name, '-->', prev_created_network.name)

    # base case
    return __NetworkParams_to_Network(cur_template_network, prev_created_network, model_params, verbose)


def __ModelParams_to_Model(template_model, model_params, verbose=False):
    configured_output = _traverse_and_build(template_model.output, None, model_params, verbose)
    configured_model = m.Model(configured_output)
    # set the params to keep track of the exact things going into this model
    configured_model.model_configuration = model_params
    return configured_model
    
    

####### Model --> NDN converters
def __Network_to_FFnetwork(network):
    # add layers to the network
    NDNLayers = []
    for li, layer in enumerate(network.layers):
        # get the layer_type
        layer_type = layer.params['internal_layer_type']
        
        # get params to pass
        sanitized_layer_params = {}
        for k,v in layer.params.items():
            # skip internal params
            if not 'internal' in k:
                sanitized_layer_params[k] = v

        NDNLayers.append(layer_type.layer_dict(**sanitized_layer_params))

    # if the network gets input from an Input (e.g. has input_covariate)
    if network.input_covariate is not None:
        return FFnetwork.ffnet_dict(
            xstim_n=network.input_covariate,
            ffnet_n=None,
            layer_list=NDNLayers,
            ffnet_type=network.ffnet_type)
    else: # if the network gets inputs from other Networks
        return FFnetwork.ffnet_dict(
            xstim_n=None,
            ffnet_n=[inp.index for inp in network.inputs],
            layer_list=NDNLayers,
            ffnet_type=network.ffnet_type)
        

def __Model_to_NDN(model, verbose):
    ffnets = []
    for network in model.networks:
        ffnets.append(__Network_to_FFnetwork(network))
    import pprint
    if verbose:
        print('====FF====')
        for i in range(len(model.networks)):
            print('---', model.networks[i].name, '---')
            pprint.pprint(ffnets[i])
    return NDN.NDN(ffnet_list=ffnets)


####### public methods
# TODO: use the sequential explorer by default
def generate(model_template, explorer=Explorer.grid, verbose=False):
    models = []
    
    models_params = explorer(model_template)
    
    for model_params in models_params:
        model =__ModelParams_to_Model(model_template, model_params, verbose)
        NDN = __Model_to_NDN(model, verbose)
        model.NDN = NDN
        models.append(model)
    
    return models