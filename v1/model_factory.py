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
def __LayerParams_to_Layer(template_layer, layer_params):
    ...


def __NetworkParams_to_Network(template_network, output_network, network_params):
    # configured_network = Network([])
    # # TODO: support Sum and Mult
    # for li, layer in enumerate(template_node.layers):
    #     layer_configuration = network_configuration[li]
    #     layer_params = {k:v for k,v in layer_configuration}
    # 
    #     # set input_dims on the first layer if they are provided
    #     if li == 0 and input_dims is not None:
    #         layer_params['input_dims'] = input_dims
    #     # set output_dims on the laster layer if they are provided
    #     if li == len(template_node.layers)-1 and output_dims is not None:
    #         layer_params['num_filters'] = output_dims
    # 
    #     # make the Layer
    #     configured_layer = Layer(layer_params)
    #     configured_network.add_layer(configured_layer)
    # return configured_network
    ...


# traverse the template_model networks and build new networks
# this will recursively build the expression tree from output to inputs
def _traverse_and_build(cur_template_network, prev_created_network, model_params):
    node = cur_template_network # node to use
    
    if prev_created_network is None:
        print(cur_template_network.name, '-->', 'None')
    elif cur_template_network is None:
        print('None', '-->', prev_created_network.name)
    else:
        print(cur_template_network.name, '-->', prev_created_network.name)

    # base case
    desired_network = None
    
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
            
        # copy the network
        new_network = m.Network(layers=layers, name=node.name)
        new_network.index = node.index

        print('Network', new_network)
                
        # Network only has one input, so this is OK
        new_network.inputs = [_traverse_and_build(node.inputs[0], new_network, model_params)]
        desired_network = new_network

    elif isinstance(node, m.Sum):
        # create the Sum network
        sum_network = m.Sum()
        sum_network.index = node.index
        print('Sum', sum_network)
        sum_network.inputs = [_traverse_and_build(prev_in, sum_network, model_params) for prev_in in node.inputs]
        desired_network = sum_network

    elif isinstance(node, m.Mult):
        # create the Mult network
        mult_network = m.Mult()
        mult_network.index = node.index
        print('Mult', mult_network)
        mult_network.inputs = [_traverse_and_build(prev_in, mult_network, model_params) for prev_in in node.inputs]
        desired_network = mult_network

    elif isinstance(node, m.Input):
        # create the Input network
        input_network = m.Input(node.covariate, node.input_dims)
        print('Input', input_network)
        # input doesn't have any inputs to build, it is the base case
        desired_network = input_network

    else: # isinstance(node, m.Output):
        # create the Output network
        output_network = m.Output(node.num_neurons)
        print('Output', output_network)
        # Output only has one input, so this is OK
        output_network.inputs = [_traverse_and_build(node.inputs[0], output_network, model_params)]
        desired_network = output_network
    
    return desired_network



def __ModelParams_to_Model(template_model, model_params):
    print('len modelparams', len(model_params))
    configured_output = _traverse_and_build(template_model.output, None, model_params)
    print('output', configured_output)
    configured_model = m.Model(configured_output)
    # set the params to keep track of the exact things going into this model
    configured_model.model_configuration = model_params
    return configured_model
    
    

####### Model --> NDN converters
def __Layer_to_FFlayer(network):
    ...


def __Network_to_FFnetwork(network):
    assert not (parents is not None and input_name is not None), "only parents xor input name can be specified"

    NDNLayers = []
    # TODO: support Sum and Mult
    for li, layer in enumerate(network.layers):
        # get the layer_type (the first element of the list)
        layer_type = layer.params['internal_layer_type'][0]

        # get params to pass
        layer_params = {}
        for k,v in layer.params.items():
            # skip internal params
            if not 'internal' in k:
                layer_params[k] = v

        # set input_dims on the first layer if they are provided
        if li == 0 and input_dims is not None:
            layer_params['input_dims'] = input_dims
        # set output_dims on the laster layer if they are provided
        if li == len(network.layers)-1 and output_dims is not None:
            layer_params['num_filters'] = output_dims

        NDNLayers.append(layer_type.layer_dict(**layer_params))


    ffnet_in = None
    if parents is not None:
        ffnet_in = [parent for parent in network.parents if parent.index >= 0]

    # add desired children based on the original Model children (and parents)
    return FFnetwork.ffnet_dict(
        xstim_n=input_name,
        ffnet_n=ffnet_in,
        layer_list=NDNLayers,
        ffnet_type=network.ffnet_type)


def __Model_to_NDN(model, model_params):
    ...


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
def create_models(model_template):
    models = []
    
    models_params = __explode_params(model_template)
    
    for model_params in models_params:
        model =__ModelParams_to_Model(model_template, model_params)
        #NDN = __Model_to_NDN(model, model_params)
        #model.NDN = NDN
        models.append(model)
    
    return models