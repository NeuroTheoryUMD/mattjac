import sys
sys.path.insert(0, '../') # to have access to NDNT

import copy # needed to handle annoying python pass by reference
import networkx as nx
from enum import Enum

# NDN imports
import NDNT.NDNT as NDN
from NDNT.modules.layers import *
from NDNT.networks import *


# enums to make things easier to remember
class Norm(Enum):
    none = 0
    unit = 1
    max = 2

class NL(Enum):
    linear = 'lin'
    relu = 'relu'
    elu = 'elu'
    square = 'square'
    softplus = 'softplus'
    tanh = 'tanh'
    sigmoid = 'sigmoid'

class LayerType(Enum):
    normal = 'normal'
    conv = 'conv'
    divnorm = 'divnorm'
    tconv = 'tconv'
    stconv = 'stconv'
    biconv = 'biconv'
    readout = 'readout'
    fixation = 'fixation'
    lag = 'lag'
    time = 'time' 
    dim0 = 'dim0'
    dimSP = 'dimSP'
    channel = 'channel'
    LVlayer = 'LVlayer'
    
class NetworkType(Enum):
    normal = 'normal'
    add = 'add'
    mult = 'mult'
    readout = 'readout'
    scaffold = 'scaffold'
    
# TODO: make output_norm an enum
# TODO: make window an enum
# TODO: make reg_vals enums as well



# layer
# need to deepcopy the params
# to get around weird python inheritance junk
# where subclasses overwrite the superclass state
# https://stackoverflow.com/questions/15469579/when-i-instantiate-a-python-subclass-it-overwrites-base-class-attribute
# defines superset of all possible params
# converts params that are provided to the value required by the NDN
def _convert_params(internal_layer_type,
                    filter_dims=None,
                    window=None,
                    padding=None,
                    num_filters=None,
                    num_inh=None,
                    bias=None,
                    norm_type=None,
                    NLtype=None,
                    initialize_center=None,
                    reg_vals=None,
                    output_norm=None,
                    pos_constraint=None,
                    temporal_tent_spacing=None):
    params = {
        'internal_layer_type': internal_layer_type
    }

    if filter_dims is not None: params['filter_dims'] = filter_dims
    if window is not None: params['window'] = window
    if padding is not None: params['padding'] = padding
    if num_filters is not None: params['num_filters'] = num_filters
    if num_inh is not None: params['num_inh'] = num_inh
    if bias is not None: params['bias'] = bias
    if norm_type is not None: params['norm_type'] = norm_type.value
    if NLtype is not None: params['NLtype'] = NLtype.value
    if initialize_center is not None: params['initialize_center'] = initialize_center
    if reg_vals is not None: params['reg_vals'] = reg_vals
    if output_norm is not None: params['output_norm'] = output_norm
    if pos_constraint is not None: params['pos_constraint'] = pos_constraint
    if temporal_tent_spacing is not None: params['temporal_tent_spacing'] = temporal_tent_spacing
    
    return params
    

class Layer:
    def __init__(self,
                 num_filters:int=None,
                 num_inh:int=None,
                 bias:bool=None,
                 norm_type:Norm=None,
                 NLtype:NL=None,
                 initialize_center:bool=None,
                 reg_vals:dict=None,
                 output_norm:bool=None,
                 pos_constraint:bool=None,
                 temporal_tent_spacing:int=None):
        self.params = _convert_params(internal_layer_type=NDNLayer,
                                      num_filters=num_filters,
                                      num_inh=num_inh,
                                      bias=bias,
                                      norm_type=norm_type,
                                      NLtype=NLtype,
                                      initialize_center=initialize_center,
                                      reg_vals=reg_vals,
                                      output_norm=output_norm,
                                      pos_constraint=pos_constraint,
                                      temporal_tent_spacing=temporal_tent_spacing)
        self.network = None # to be able to point to the network we are a part of
        self.index = None # to be able to point to the layer we are a part of in the Network
        # TODO: be able to set this layer's weights as the 
        #       weights of a previous layer that maybe we can access by name
        #       from the Model API
        # like Layer().use_weights(prev_layer.get_layer('drift'))
        # or something like this...
    
    # make this layer like another layer
    def like(self, layer):
        # copy the other layer params into this layer's params
        self.params = copy.deepcopy(layer.params)
        return self
    
    def _get_weights(self):
        return self.network.model.NDN.networks[self.network.index].layers[self.index].get_weights()

    # define property to make it easier to remember
    weights = property(_get_weights)


# layer subclasses
class PassthroughLayer:
    def __init__(self, 
                 num_filters=None,
                 bias=None,
                 NLtype=None):
        self.params = _convert_params(internal_layer_type = ChannelLayer,
                                      num_filters = num_filters,
                                      bias = bias,
                                      NLtype = NLtype)
        
        self.network = None # to be able to point to the network we are a part of
        self.index = 0 # to be able to point to the layer we are a part of in the NDN

    def _get_weights(self):
        return self.network.model.NDN.networks[self.network.index].layers[self.index].get_weights()

    # define property to make it easier to remember
    weights = property(_get_weights)


class ConvolutionalLayer:
    def __init__(self,
                 filter_dims=None,
                 window=None,
                 padding='same', # default in the NDN
                 num_filters=None,
                 num_inh=0,
                 bias=False,
                 norm_type=Norm.none,
                 NLtype=NL.linear,
                 initialize_center=False,
                 reg_vals=None,
                 output_norm=None,
                 pos_constraint=False,
                 temporal_tent_spacing:int=None):
        self.params = _convert_params(internal_layer_type=ConvLayer,
                                      filter_dims=filter_dims,
                                      window=window,
                                      padding=padding,
                                      num_filters=num_filters,
                                      num_inh=num_inh,
                                      bias=bias,
                                      norm_type=norm_type,
                                      NLtype=NLtype,
                                      initialize_center=initialize_center,
                                      reg_vals=reg_vals,
                                      output_norm=output_norm,
                                      pos_constraint=pos_constraint,
                                      temporal_tent_spacing=temporal_tent_spacing)
        self.network = None # to be able to point to the network we are a part of
        self.index = None # to be able to point to the layer we are a part of in the NDN

    # make this layer like another layer
    def like(self, layer):
        # copy the other layer params into this layer's params
        self.params = copy.deepcopy(layer.params)
        return self

    def _get_weights(self):
        return self.network.model.NDN.networks[self.network.index].layers[self.index].get_weights()

    # define property to make it easier to remember
    weights = property(_get_weights)

class TemporalConvolutionalLayer:
    def __init__(self,
                 filter_dims=None,
                 window=None,
                 padding='valid', # default in the NDN
                 num_filters=None,
                 num_inh=0,
                 bias=False,
                 norm_type=Norm.none,
                 NLtype=NL.linear,
                 initialize_center=False,
                 reg_vals=None,
                 output_norm=None,
                 pos_constraint=False,
                 temporal_tent_spacing:int=None):
        self.params = _convert_params(internal_layer_type=TconvLayer,
                                      filter_dims=filter_dims,
                                      window=window,
                                      padding=padding,
                                      num_filters=num_filters,
                                      num_inh=num_inh,
                                      bias=bias,
                                      norm_type=norm_type,
                                      NLtype=NLtype,
                                      initialize_center=initialize_center,
                                      reg_vals=reg_vals,
                                      output_norm=output_norm,
                                      pos_constraint=pos_constraint,
                                      temporal_tent_spacing=temporal_tent_spacing)
        self.network = None # to be able to point to the network we are a part of
        self.index = None # to be able to point to the layer we are a part of in the NDN

    # make this layer like another layer
    def like(self, layer):
        # copy the other layer params into this layer's params
        self.params = copy.deepcopy(layer.params)
        return self

    def _get_weights(self):
        return self.network.model.NDN.networks[self.network.index].layers[self.index].get_weights()

    # define property to make it easier to remember
    weights = property(_get_weights)


# utility "networks"
class Input: # holds the input info
    def __init__(self, covariate, input_dims):
        self.name = covariate
        self.index = -1 # index of network in the list of networks
        self.covariate = covariate
        self.input_dims = input_dims
        self.inputs = [] # this should always be empty for an Input
        self.output = None

    def to(self, network):
        # set the input_dims of the network to be the desired Input.input_dims
        network.input_covariate = self.covariate
        network.layers[0].params['input_dims'] = self.input_dims
        self.output = network
        network.inputs.append(self)

    def __str__(self):
        return 'Input name='+self.name+', covariate='+self.covariate+', input_dims='+','.join([str(d) for d in self.input_dims])


class Output: # holds the output info
    def __init__(self, num_neurons):
        self.name = 'output'
        self.index = -1 # index of network in the list of networks
        self.num_neurons = num_neurons
        self.inputs = []
        self.output = None
    
    def update_num_neurons(self, num_neurons):
        # this will iterate over its inputs,
        # and update their last layer.num_filters to be num_neurons
        assert len(self.inputs) == 1, 'Output can only have one input'
        self.num_neurons = num_neurons
        self.inputs[0].layers[-1].params['num_filters'] = num_neurons
    
    def __str__(self):
        return 'Output name='+self.name+', num_neurons='+str(self.num_neurons)
    

class Add:
    def __init__(self, 
                 networks=None, 
                 NLtype=NL.softplus, 
                 bias=False):
        # TODO: this is a little hacky
        num_filters = None
        if networks is not None:
            num_filters = networks[0].layers[-1].params['num_filters']
            # make sure that the networks all have the same num_filters in their output
            for network in networks:
                assert network.layers[-1].params['num_filters'] == num_filters, "input networks must all have the same num_filters"

        self.name = '+'
        if networks is not None: # TODO: kind of a hack
            self.name = '+'.join(network.name for network in networks)
        
        self.index = -1 # index of network in the list of networks
        self.inputs = networks
        self.output = None

        # NDN params
        self.input_covariate = None # this should always be None for an Operator
        self.ffnet_type = NetworkType.add.value
        self.layers = [PassthroughLayer(num_filters=num_filters, NLtype=NLtype, bias=bias)]

        # points its parents (the things to be summed) to this node
        if networks is not None:
            for network in networks:
                network.output = [self]

    def to(self, network):
        # if we are going to an output, update our num_filters to be the num_neurons
        if isinstance(network, Output):
            self.layers[-1].params['num_filters'] = network.num_neurons
        self.output = network
        network.inputs.append(self)

    def __str__(self):
        if self.inputs is not None:
            return 'Add name='+self.name+' '+str(self.index)+', inputs='+','.join([str([inp]) for inp in self.inputs])
        else:
            return 'Add name='+self.name+' '+str(self.index)


class Mult:
    def __init__(self,
                 networks=None,
                 NLtype=NL.softplus,
                 bias=False):
        num_filters = None
        # TODO: this is a little hacky
        if networks is not None:
            num_filters = networks[0].layers[-1].params['num_filters']
            # make sure that the networks all have the same num_filters in their output
            for network in networks:
                assert network.layers[-1].params['num_filters'] == num_filters, "input networks must all have the same num_filters"

        self.name = '*'
        if networks is not None: # TODO: kind of a hack
            self.name = '*'.join(network.name for network in networks)

        self.index = -1 # index of network in the list of networks
        self.inputs = networks
        self.output = None

        # NDN params
        self.input_covariate = None # this should always be None for an Operator
        self.ffnet_type = NetworkType.mult.value
        self.layers = [PassthroughLayer(num_filters=num_filters, NLtype=NLtype, bias=bias)]
        # points its parents (the things to be multiplied) to this node
        if networks is not None:
            for network in networks:
                network.output = [self]

    def to(self, network):
        # if we are going to an output, update our num_filters to be the num_neurons
        if isinstance(network, Output):
            self.layers[-1].params['num_filters'] = network.num_neurons
        self.output = network
        network.inputs.append(self)

    def __str__(self):
        if self.inputs is not None:
            return 'Mult name='+self.name+' '+str(self.index)+', inputs='+','.join([str([inp]) for inp in self.inputs])
        else:
            return 'Mult name='+self.name+' '+str(self.index)


# network
class Network:
    def __init__(self, layers, name, network_type=NetworkType.normal):
        # internal params
        self.model = None # to be able to point to the model we are a part of
        self.name = name
        self.network_type = network_type
        self.index = -1 # index of network in the list of networks
        self.inputs = []
        self.output = None
        
        # NDNLayer params
        self.input_covariate = None # the covariate that goes to this network
        self.ffnet_type = network_type.value # get the value out of the network_type
        self.layers = layers
        for li, layer in enumerate(self.layers):
            layer.index = li
            layer.network = self # point the layer back to the network it is a part of
        
    def to(self, network):
        network.inputs.append(self)
        self.output = network
        if isinstance(network, Output):
            assert 'num_filters' not in self.layers[-1].params, 'num_filters should not be set on the last layer going to the Output'
            # TODO: maybe it is gross, 
            # but update this output layer num_filters to match the num_neurons
            network.update_num_neurons(network.num_neurons)
    
    def __str__(self):
        return 'Network name='+self.name+' '+str(self.index)+', len(layers)='+str(len(self.layers))+', inputs='+','.join([str([inp]) for inp in self.inputs])


# model
def _Network_to_FFnetwork(network):
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

        layer_dict = layer_type.layer_dict(**sanitized_layer_params)
        # NDN has a bug where certain parameters don't get copied over from the constructor
        # we need to set them separately from the constructor
        if 'reg_vals' in sanitized_layer_params:
            layer_dict['reg_vals'] = sanitized_layer_params['reg_vals']
        if 'output_norm' in sanitized_layer_params:
            layer_dict['output_norm'] = sanitized_layer_params['output_norm']
        if 'window' in sanitized_layer_params:
            layer_dict['window'] = sanitized_layer_params['window']

        NDNLayers.append(layer_dict)

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


def _Model_to_NDN(model, verbose):
    ffnets = []
    for network in model.networks:
        ffnets.append(_Network_to_FFnetwork(network))
    import pprint
    if verbose:
        print('====FF====')
        for i in range(len(model.networks)):
            print('---', model.networks[i].name, '---')
            pprint.pprint(ffnets[i])
    return NDN.NDN(ffnet_list=ffnets)


class Model:
    def __init__(self, output, verbose=False):
        self.NDN = None # set this Model's NDN to None to start
        self.inputs = []
        self.networks = []
        self.output = output
        
        # network.index --> network map
        self.netidx_to_model = {}
        # network.name --> network map
        self.networks_by_name = {}
        
        for network in self.traverse():
            # create groups
            if isinstance(network, Output):
                continue
            if isinstance(network, Input):
                self.inputs.append(network)
            else: # if it is a Network, Add, Mult or otherwise
                network.model = self # point the network back to the model
                self.networks.append(network)
                
        # b/c the NDNT requires that earlier networks have lower indices,
        # we need to reverse the network list and the associated indices
        self.networks.reverse()
        for idx, network in enumerate(self.networks):
            self.netidx_to_model[idx] = network
            assert network.name not in self.networks_by_name, "networks must have unique names"
            self.networks_by_name[network.name] = network
            network.index = idx # set the reversed depth-first index

        # create the NDN now
        self.NDN = _Model_to_NDN(self, verbose)

    def update_num_neurons(self, num_neurons, verbose=False):
        self.output.update_num_neurons(num_neurons)
        # update the NDN as well
        self.NDN = _Model_to_NDN(self, verbose)
    
    def __str__(self):
        return 'Model len(inputs)'+str(len(self.inputs))+\
            ', len(networks)='+str(len(self.networks))+\
            ', output='+str(self.output.num_neurons)

    # recursively traverse, depth-first, from the output to the inputs
    def _traverse(self, inp, out, networks, verbose=False):
        networks.append(inp)
        if verbose:
            if out is not None:
                print(inp.name, '-->', out.name)
            else:
                print(inp.name, '--> None')
        # base case
        if isinstance(inp, Input):
            return inp
        
        for prev_inp in inp.inputs:
            self._traverse(prev_inp, inp, networks, verbose)
    
    def traverse(self, verbose=False):
        # traverse starting from the output
        # this is the default behavior
        networks = []
        self._traverse(self.output, None, networks, verbose)
        return networks
    
    def draw_network(self, verbose=False):
        g = nx.DiGraph()
        
        for net in self.traverse(verbose):
            # add the node to the graph
            for inp in net.inputs:
                g.add_edge(inp.name, net.name)
                
        # draw graph
        nx.draw_networkx(g, with_labels=True)
