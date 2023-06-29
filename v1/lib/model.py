import sys
# TODO: figure out how to do this correctly before code review
sys.path.insert(0, '../../') # to have access to NDNT

import copy # needed to handle annoying python pass by reference
import pprint
import networkx as nx
import numpy as np
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
    
class RegVal(Enum):
    d2xt = 'd2xt',
    d2x = 'd2x',
    d2t = 'd2t',
    l1 = 'l1',
    l2 = 'l2',
    norm2 = 'norm2',
    norm = 'norm',
    pos = 'pos',
    neg = 'neg',
    orth = 'orth',
    bi_t = 'bi_t',
    glocalx = 'glocalx',
    glocalt = 'glocalt',
    localx = 'localx',
    localt = 'localt',
    trd = 'trd',
    local = 'local',
    glocal = 'glocal',
    max = 'max',
    max_filt = 'max_filt',
    max_space = 'max_space',
    center = 'center',
    edge_t = 'edge_t',
    edge_x = 'edge_x'
    
# TODO: make output_norm an enum
# TODO: make window an enum


# layer
# need to deepcopy the params
# to get around weird python inheritance junk
# where subclasses overwrite the superclass state
# https://stackoverflow.com/questions/15469579/when-i-instantiate-a-python-subclass-it-overwrites-base-class-attribute
# defines superset of all possible params
# converts params that are provided to the value required by the NDN
def _convert_params(internal_layer_type,
                    internal_freeze_weights=False,
                    internal_weights=None,
                    filter_dims=None,
                    filter_width=None,
                    window=None,
                    padding=None,
                    num_filters=None,
                    num_inh_percent=None,
                    bias=None,
                    norm_type=None,
                    NLtype=None,
                    initialize_center=None,
                    reg_vals=None,
                    output_norm=None,
                    pos_constraint=None,
                    temporal_tent_spacing=None,
                    num_iter=None,
                    output_config=None,
                    res_layer=None,
                    num_lags=None):
    params = {
        'internal_layer_type': internal_layer_type,
        'internal_freeze_weights': internal_freeze_weights,
        'internal_weights': internal_weights
    }

    if filter_dims is not None: params['filter_dims'] = filter_dims
    if filter_width is not None: params['filter_width'] = filter_width
    if window is not None: params['window'] = window
    if padding is not None: params['padding'] = padding
    if num_filters is not None: params['num_filters'] = num_filters
    if num_inh_percent is not None: params['num_inh_percent'] = num_inh_percent
    if bias is not None: params['bias'] = bias
    if norm_type is not None: params['norm_type'] = norm_type.value
    if NLtype is not None: params['NLtype'] = NLtype.value
    if initialize_center is not None: params['initialize_center'] = initialize_center
    if reg_vals is not None: params['reg_vals'] = reg_vals
    if output_norm is not None: params['output_norm'] = output_norm
    if pos_constraint is not None: params['pos_constraint'] = pos_constraint
    if temporal_tent_spacing is not None: params['temporal_tent_spacing'] = temporal_tent_spacing
    if num_iter is not None: params['num_iter'] = num_iter
    if output_config is not None: params['output_config'] = output_config
    if res_layer is not None: params['res_layer'] = res_layer
    if num_lags is not None: params['num_lags'] = num_lags
    
    return params
    

class Layer:
    def __init__(self,
                 num_filters:int=None,
                 num_inh_percent:float=None,
                 bias:bool=None,
                 norm_type:Norm=None,
                 NLtype:NL=None,
                 initialize_center:bool=None,
                 reg_vals:dict=None,
                 output_norm:bool=None,
                 pos_constraint:bool=None,
                 temporal_tent_spacing:int=None,
                 freeze_weights:bool=False,
                 weights=None):
        self.params = _convert_params(internal_layer_type=NDNLayer,
                                      internal_freeze_weights=freeze_weights,
                                      internal_weights=weights,
                                      num_filters=num_filters,
                                      num_inh_percent=num_inh_percent,
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
    
    def _get_weights(self):
        return self.network.model.NDN.networks[self.network.index].layers[self.index].get_weights()
    
    def _subunit_weights(self):
        weights = self.network.model.NDN.networks[self.network.index].layers[self.index].get_weights()
        # reshape weights by the number of filters in this layer
        return np.reshape(weights, (self.params['num_filters'], -1))

    # define property to make it easier to remember
    weights = property(_get_weights)
    subunit_weights = property(_subunit_weights)


class TemporalLayer:
    def __init__(self,
                 num_filters:int=None,
                 num_inh_percent:float=None,
                 num_lags:int=None,
                 bias:bool=None,
                 norm_type:Norm=None,
                 NLtype:NL=None,
                 initialize_center:bool=None,
                 reg_vals:dict=None,
                 output_norm:bool=None,
                 pos_constraint:bool=None,
                 temporal_tent_spacing:int=None,
                 freeze_weights:bool=False,
                 weights=None):
        self.params = _convert_params(internal_layer_type=Tlayer,
                                      internal_freeze_weights=freeze_weights,
                                      internal_weights=weights,
                                      num_filters=num_filters,
                                      num_inh_percent=num_inh_percent,
                                      num_lags=num_lags,
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
    def _get_weights(self):
        return self.network.model.NDN.networks[self.network.index].layers[self.index].get_weights()

    def _subunit_weights(self):
        weights = self.network.model.NDN.networks[self.network.index].layers[self.index].get_weights()
        # reshape weights by the number of filters in this layer
        return np.reshape(weights, (self.params['num_filters'], -1))

    # define property to make it easier to remember
    weights = property(_get_weights)
    subunit_weights = property(_subunit_weights)


# layer subclasses
class PassthroughLayer:
    def __init__(self, 
                 num_filters=None,
                 bias=None,
                 NLtype=None,
                 freeze_weights=False):
        self.params = _convert_params(internal_layer_type = ChannelLayer,
                                      internal_freeze_weights=freeze_weights,
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
                 padding=None,
                 num_filters=None,
                 num_inh_percent=None,
                 bias=None,
                 norm_type=None,
                 NLtype=None,
                 initialize_center=None,
                 reg_vals=None,
                 output_norm=None,
                 pos_constraint=None,
                 temporal_tent_spacing:int=None):
        self.params = _convert_params(internal_layer_type=ConvLayer,
                                      filter_dims=filter_dims,
                                      window=window,
                                      padding=padding,
                                      num_filters=num_filters,
                                      num_inh_percent=num_inh_percent,
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

    def _get_weights(self):
        return self.network.model.NDN.networks[self.network.index].layers[self.index].get_weights()

    # define property to make it easier to remember
    weights = property(_get_weights)

class IterativeConvolutionalLayer:
    def __init__(self,
                 filter_dims=None,
                 window=None,
                 padding='same', # default in the NDN
                 num_filters=None,
                 num_inh_percent=0.0,
                 bias=False,
                 norm_type=Norm.none,
                 NLtype=NL.linear,
                 initialize_center=False,
                 reg_vals=None,
                 output_norm=None,
                 pos_constraint=False,
                 temporal_tent_spacing:int=None,
                 num_iter=1,
                 output_config='last',
                 res_layer=True):
        self.params = _convert_params(internal_layer_type=IterLayer,
                                      filter_width=filter_dims, # Dan change filter_dims to filter_width here
                                      window=window,
                                      padding=padding,
                                      num_filters=num_filters,
                                      num_inh_percent=num_inh_percent,
                                      bias=bias,
                                      norm_type=norm_type,
                                      NLtype=NLtype,
                                      initialize_center=initialize_center,
                                      reg_vals=reg_vals,
                                      output_norm=output_norm,
                                      pos_constraint=pos_constraint,
                                      temporal_tent_spacing=temporal_tent_spacing,
                                      num_iter=num_iter,
                                      output_config=output_config,
                                      res_layer=res_layer)
        self.network = None # to be able to point to the network we are a part of
        self.index = None # to be able to point to the layer we are a part of in the NDN

    def _get_weights(self):
        return self.network.model.NDN.networks[self.network.index].layers[self.index].get_weights()

    # define property to make it easier to remember
    weights = property(_get_weights)

class TemporalConvolutionalLayer:
    def __init__(self,
                 filter_width=None,
                 num_lags=None,
                 window=None,
                 padding='spatial', # default in the NDN
                 num_filters=None,
                 num_inh_percent=0.0,
                 bias=False,
                 norm_type=Norm.none,
                 NLtype=NL.linear,
                 initialize_center=False,
                 reg_vals=None,
                 output_norm=None,
                 pos_constraint=False,
                 temporal_tent_spacing:int=None):
        
        self.params = _convert_params(internal_layer_type=TconvLayer,
                                      filter_width=filter_width,
                                      num_lags=num_lags,
                                      window=window,
                                      padding=padding,
                                      num_filters=num_filters,
                                      num_inh_percent=num_inh_percent,
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

    def _get_weights(self):
        return self.network.model.NDN.networks[self.network.index].layers[self.index].get_weights()

    # define property to make it easier to remember
    weights = property(_get_weights)

class IterativeTemporalConvolutionalLayer:
    def __init__(self,
                 num_lags=None,
                 filter_width=None,
                 window=None,
                 num_filters=None,
                 padding='spatial', # default in the NDN
                 num_inh_percent=0.0,
                 bias=False,
                 norm_type=Norm.none,
                 NLtype=NL.linear,
                 initialize_center=False,
                 reg_vals=None,
                 output_norm=None,
                 pos_constraint=False,
                 temporal_tent_spacing:int=None,
                 num_iter=1,
                 output_config='last',
                 res_layer=True,):
        self.params = _convert_params(internal_layer_type=IterTlayer,
                                      filter_width=filter_width,
                                      window=window,
                                      padding=padding,
                                      num_filters=num_filters,
                                      num_inh_percent=num_inh_percent,
                                      bias=bias,
                                      norm_type=norm_type,
                                      NLtype=NLtype,
                                      initialize_center=initialize_center,
                                      reg_vals=reg_vals,
                                      output_norm=output_norm,
                                      pos_constraint=pos_constraint,
                                      temporal_tent_spacing=temporal_tent_spacing,
                                      num_iter=num_iter,
                                      output_config=output_config,
                                      res_layer=res_layer,
                                      num_lags=num_lags)
        self.network = None # to be able to point to the network we are a part of
        self.index = None # to be able to point to the layer we are a part of in the NDN

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
        # if the first layer is a Tlayer, then we need to change the last dimension to be 1
        if network.layers[0].params['internal_layer_type'] is Tlayer:
            print('changing last dimension of input_dims to be 1')
            network.layers[0].params['input_dims'][-1] = 1
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


class Concat:
    def __init__(self,
                 NLtype,
                 networks=None,
                 bias=None,
                 freeze_weights=False):
        # TODO: this is a little hacky
        num_filters = None
        if networks is not None:
            num_filters = networks[0].layers[-1].params['num_filters']

        self.name = '.'
        if networks is not None: # TODO: kind of a hack
            self.name = '.'.join(network.name for network in networks)

        self.index = -1 # index of network in the list of networks
        self.inputs = networks
        self.output = None

        # NDN params
        self.input_covariate = None # this should always be None for an Operator
        self.ffnet_type = NetworkType.normal.value # normal is used for concatenation
        self.layers = [PassthroughLayer(num_filters=num_filters, NLtype=NLtype, bias=bias, freeze_weights=freeze_weights)]

        # points its parents (the things to be summed) to this node
        if networks is not None:
            for network in networks:
                network.output = self
    def to(self, network):
        # if we are going to an output, update our num_filters to be the num_neurons
        if isinstance(network, Output):
            self.layers[-1].params['num_filters'] = network.num_neurons
        self.output = network
        network.inputs.append(self)

    def __str__(self):
        if self.inputs is not None:
            return 'Concat name='+self.name+' '+str(self.index)+', inputs='+','.join([str([inp]) for inp in self.inputs])
        else:
            return 'Concat name='+self.name+' '+str(self.index)

class Add:
    def __init__(self,
                 NLtype,
                 networks=None,
                 bias=None):
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
        self.layers = [Layer(num_filters=num_filters, NLtype=NLtype, bias=bias)]

        # points its parents (the things to be summed) to this node
        if networks is not None:
            for network in networks:
                network.output = self

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
                 NLtype,
                 networks=None,
                 bias=None):
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
        self.layers = [Layer(num_filters=num_filters, NLtype=NLtype, bias=bias)]
        
        # points its parents (the things to be multiplied) to this node
        if networks is not None:
            for network in networks:
                network.output = self

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
                
        # convert the num_inh_percent to num_inh
        if 'num_inh_percent' in sanitized_layer_params \
                and sanitized_layer_params['num_inh_percent'] is not None \
                and sanitized_layer_params['num_inh_percent'] > 0:
            num_inh = int(sanitized_layer_params['num_inh_percent'] * sanitized_layer_params['num_filters'])
            sanitized_layer_params['num_inh'] = num_inh
            del sanitized_layer_params['num_inh_percent']

        # populate the filter_dims parameter for the TemporalConvolutionalLayer
        if layer_type == TconvLayer:
            sanitized_layer_params['filter_dims'] = [None, sanitized_layer_params['filter_width'], 1, sanitized_layer_params['num_lags']]
            del sanitized_layer_params['filter_width']
            del sanitized_layer_params['num_lags']
            
            if li == 0:
                sanitized_layer_params['filter_dims'][0] = 1
            else:
                # set the filter_dims num_channels to the previous layer's num_filters
                num_iter = 1
                if 'num_iter' in NDNLayers[-1]:
                    num_iter = NDNLayers[-1]['num_iter']
                sanitized_layer_params['filter_dims'][0] = NDNLayers[-1]['num_filters']*num_iter
            layer_dict = layer_type.layer_dict(**sanitized_layer_params)
        elif isinstance(network, Add) or isinstance(network, Mult) or isinstance(network, Concat):
            layer_dict = layer_type.layer_dict(**sanitized_layer_params)
            layer_dict['weights_initializer'] = 'ones'
        else:
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
    if verbose:
        print("=== MODEL ===")
        for i in range(len(model.networks)):
            print('---', model.networks[i].name, '---')
            pprint.pprint(ffnets[i])
    ndn_model = NDN.NDN(ffnet_list=ffnets, loss_type='poisson')
    for ni in range(len(ndn_model.networks)):
        network = model.networks[ni]
        to_network = network.output
        
        for li in range(len(ndn_model.networks[ni].layers)):
            layer = model.networks[ni].layers[li]
            weights = layer.params['internal_weights']
            if weights is not None:
                ndn_model.networks[ni].layers[li].weight.data = copy.deepcopy(weights.data)
            
            freeze_weights = layer.params['internal_freeze_weights']
            if freeze_weights: # set the weights to be frozen on the NDN model
                # turn off the parameters on the current layer
                ndn_model.networks[ni].layers[li].set_parameters(val=False)                
                # if this is not the last layer in the network,
                # turn off the weights on the next layer
                if li < len(ndn_model.networks[ni].layers)-1:
                    print('turning off weights on layer', li+1, 'of network', ni)
                    ndn_model.networks[ni].layers[li+1].set_parameters(val=False,name='weight')
                else:
                    # if this is the last layer in the network,
                    # turn off the weights on the to_network
                    if ni < len(ndn_model.networks)-1:
                        to_network_idx = to_network.index
                        print('turning off weights on network', to_network, to_network_idx) 
                        ndn_model.networks[to_network_idx].layers[0].set_parameters(val=False,name='weight')
    return ndn_model


class Model:
    def __init__(self, output, name='', create_NDN=True, verbose=False):
        self.name = name
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
        if create_NDN:
            self.NDN = _Model_to_NDN(self, verbose)

    def update_NDN(self, verbose=False):
        self.NDN = _Model_to_NDN(self, verbose)

    def update_num_neurons(self, num_neurons, verbose=False):
        self.output.update_num_neurons(num_neurons)
        # update the NDN as well
        self.update_NDN()

    # prevent the weights from being updated on forward
    def freeze_weights(self, network_names=None):
        for network in self.networks:
            # skip networks we don't want
            if network_names is not None and network.name not in network_names:
                continue
            self.NDN.set_parameters(name='weight', val=False,
                                    ffnet_target=network.index)

    # initialize the weights using a new random initialization
    def reinitialize_weights(self, verbose=False):
        # call the NDN create function again to reinitialize the weights
        self.update_NDN()

    # copy the weights from the previous model into this model
    def use_weights_from(self, other_model, network_names=None):
        # validate the models have the same structure before copying
        # throw Exception if they do not have same structure
        # https://docs.python.org/3/library/exceptions.html#exception-hierarchy
        
        # use all the network names if none are set
        if network_names is None:
            network_names = list(self.networks_by_name.keys())
        
        filtered_self_networks = [net for net in self.networks if net.name in network_names]
        filtered_other_networks = [net for net in other_model.networks if net.name in network_names]
        if len(filtered_self_networks) != len(filtered_other_networks):
            raise TypeError("models must have the same number of networks")
        for ni in range(len(other_model.NDN.networks)):
            # skip networks we don't want
            if self.networks[ni].name not in network_names:
                continue
            if len(other_model.NDN.networks[ni].layers) != len(self.NDN.networks[ni].layers):
                raise TypeError("networks must have the same number of layers")
            for li in range(len(other_model.NDN.networks[ni].layers)):
                prev_weight = other_model.NDN.networks[ni].layers[li].weight
                curr_weight = self.NDN.networks[ni].layers[li].weight
                if prev_weight.shape != curr_weight.shape:
                    raise TypeError("weights must have the same shape")

        for ni in range(len(other_model.NDN.networks)):
            # skip networks we don't want
            if self.networks[ni].name not in network_names:
                continue
            for li in range(len(other_model.NDN.networks[ni].layers)):
                prev_layer = other_model.NDN.networks[ni].layers[li]
                curr_layer = self.NDN.networks[ni].layers[li]
                # have to make the tensors temporarily not require grad to do this
                with torch.no_grad():
                    curr_layer.weight = copy.deepcopy(prev_layer.weight)

    def __str__(self):
        return 'Model len(inputs)'+str(len(self.inputs))+ \
            ', len(networks)='+str(len(self.networks))+ \
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

    def draw(self, verbose=False):
        g = nx.DiGraph()

        for net in self.traverse(verbose):
            # add the node to the graph
            for inp in net.inputs:
                g.add_edge(inp.name, net.name)

        # draw graph
        nx.draw_networkx(g, with_labels=True)

    def print_params(self, key):
        for network in self.networks:
            print(network.name)
            for li, layer in enumerate(network.layers):
                if key in layer.params:
                    print(li, layer.params[key])
                else:
                    print(li, 'None')
