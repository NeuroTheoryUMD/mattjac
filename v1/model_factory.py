# Wrapper model
import model as m

# NDN model
import NDNT.NDNT as NDN
from NDNT.modules.layers import *
from NDNT.networks import *

# stuff to do work
import copy
import itertools as it
from collections import deque


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
            layers.append(type(layer)(layer_params_map))

        # copy the network properties
        new_network = m.Network(layers=layers, name=node.name)
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
            layers.append(type(layer)(layer_params_map))
        add_network.layers = layers # copy the layers over

        if verbose:
            print('Add', add_network)
        add_network.inputs = [_traverse_and_build(prev_in, add_network, model_params, verbose) for prev_in in node.inputs]
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
            layers.append(type(layer)(layer_params_map))
        mult_network.layers = layers # copy the layers over

        if verbose:
            print('Mult', mult_network)
        mult_network.inputs = [_traverse_and_build(prev_in, mult_network, model_params, verbose) for prev_in in node.inputs]
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
    #print('len modelparams', len(model_params))
    configured_output = _traverse_and_build(template_model.output, None, model_params, verbose)
    #print('output', configured_output)
    configured_model = m.Model(configured_output)
    # set the params to keep track of the exact things going into this model
    configured_model.model_configuration = model_params
    return configured_model
    
    

####### Model --> NDN converters
def __Network_to_FFnetwork(network):
    # add layers to the network
    NDNLayers = []
    for li, layer in enumerate(network.layers):
        # get the layer_type (the first element of the list)
        layer_type = layer.params['internal_layer_type'][0]

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


####### Params Exploder
def __explode_params(model):
    networks_with_exploded_layers = []
    for network in model.networks:
        # break apart the layer definitions into
        # a list of each possible combination of parameters
        # so each layer definition like:
        #  [ [subs=[1,2], filter=[3,4],  [subs=[a,b], filter=[c,d] ]
        #  -->  [
        #       [[subs=1, filter=3], [subs=1, filter=4],
        #        [subs=2, filter=3], [subs=2, filter=4]],
        #
        #       [[subs=a, filter=b], [subs=a, filter=d],
        #        [subs=b, filter=e], [subs=b, filter=d]] 
        exploded_layers = [list(it.product(*layer.build())) for layer in network.layers]

        # break apart the list of each possible combination of parameters
        # 
        # so each layer definition like:
        #  [ [subs=[1,2], filter=[3,4],  [subs=[a,b], filter=[c,d] ]
        #  -->  [
        #       [[subs=1, filter=3], [subs=1, filter=4],
        #        [subs=2, filter=3], [subs=2, filter=4]],
        #
        #       [[subs=a, filter=b], [subs=a, filter=d],
        #        [subs=b, filter=c], [subs=b, filter=d]]
        #    ]
        #        
        #    --> [
        #         [[subs=1, filter=3], [subs=a, filter=b]],
        #         [[subs=1, filter=3], [subs=a, filter=d]],
        #         [[subs=1, filter=3], [subs=b, filter=c]],
        #         [[subs=1, filter=3], [subs=b, filter=d]],
        #         ...
        #         [[subs=2, filter=4], [subs=a, filter=b]],
        #         [[subs=2, filter=4], [subs=a, filter=d]],
        #         [[subs=2, filter=4], [subs=b, filter=c]],
        #         [[subs=2, filter=4], [subs=b, filter=d]],
        #        ]
        #           
        layer_groups = list(it.product(*exploded_layers))
        networks_with_exploded_layers.append(layer_groups)

    # each network has a (list of layer.configuration_lists)
    # so we need to cross-product the possible layers in the networks
    # each 
    #                 [network.(list of layer.configuration_lists)]
    #     Network[ 
    #               Layer[
    #                     [subs=1, filter=3], [subs=a, filter=c]],
    #                     [subs=1, filter=3], [subs=a, filter=d]],
    #                    ]
    #             ],
    #     Network[
    #               Layer[
    #                     [subs=1, filter=3], [subs=a, filter=c]],
    #                     [subs=1, filter=3], [subs=a, filter=d]],
    #                    ]
    #             ]
    #     ...
    #         ]
    model_configurations = list(it.product(*networks_with_exploded_layers))
    return model_configurations



####### public methods
def create_models(model_template, verbose=False):
    models = []
    
    models_params = __explode_params(model_template)
    print('--->', len(models_params))
    
    for model_params in models_params:
        model =__ModelParams_to_Model(model_template, model_params, verbose)
        NDN = __Model_to_NDN(model, verbose)
        model.NDN = NDN
        models.append(model)
    
    return models