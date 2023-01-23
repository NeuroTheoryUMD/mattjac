import copy # needed to handle annoying python pass by reference


from NDNT.modules.layers import *

import networkx as nx

from enum import Enum


# enums to make things easier to remember
class Norm(Enum):
    none = 0
    unit = 1
    max = 2

class NL(Enum):
    linear = None
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


# layer
# need to deepcopy the params
# to get around weird python inheritance junk
# where subclasses overwrite the superclass state
# https://stackoverflow.com/questions/15469579/when-i-instantiate-a-python-subclass-it-overwrites-base-class-attribute
class Layer:
    # TODO: make the params a kwarg list, we can specify required and optional params this way
    
    def __init__(self, params={}):
        """
        Create the layer from the params map if provided.
        :param params: params map
        :return: layer
        """
        self.network = None # to be able to point to the network we are a part of
        self.params = copy.deepcopy(params)
        # set the layer type to be a NDNLayer
        self.params['internal_layer_type'] = [NDNLayer]
    
    # TODO: be able to set this layer's weights as the 
    #       weights of a previous layer that maybe we can access by name
    #       from the Model API
    # like Layer().use_weights(prev_layer.get_layer('drift'))
    # or something like this...

    # builder pattern for constructing a layer with parameters
    def input_dims(self, *input_dimses):
        if input_dimses is not None: # only set it if it is not none
            self.params['input_dims'] = list(input_dimses)
        return self

    def num_filters(self, *num_filterses):
        self.params['num_filters'] = list(num_filterses)
        return self
    
    def num_inh(self, *num_inhs):
        self.params['num_inh'] = list(num_inhs)
        return self

    def bias(self, *biases):
        self.params['bias'] = list(biases)
        return self
    
    def norm_type(self, *norm_types):
        self.params['norm_type'] = [norm_type.value for norm_type in norm_types]
        return self

    def NLtype(self, *NLtypes):
        self.params['NLtype'] = [NLtype.value for NLtype in NLtypes]
        return self

    def initialize_center(self, *initialize_centers):
        self.params['initialize_center'] = list(initialize_centers)
        return self
        
    def reg_vals(self, *reg_valses):
        self.params['reg_vals'] = list(reg_valses)
        return self
    
    # make this layer like another layer
    def like(self, layer):
        # copy the other layer params into this layer's params
        self.params = copy.deepcopy(layer.params)
        return self
    
    def build(self):
        # convert the dictionary of lists into a list of lists of tuples
        return [[(k,v) for v in vs] for k,vs in self.params.items()]


# layer subclasses
class PassthroughLayer(Layer):
    def __init__(self, params={}):
        """
        Create the layer from the params map if provided.
        :param params: params map
        :return: layer
        """
        self.network = None # to be able to point to the network we are a part of
        self.params = copy.deepcopy(params)
        # set the layer type to be a ConvLayer
        self.params['internal_layer_type'] = [ChannelLayer]


class ConvolutionalLayer(Layer):
    def __init__(self, params={}):
        """
        Create the layer from the params map if provided.
        :param params: params map
        :return: layer
        """
        self.network = None # to be able to point to the network we are a part of
        self.params = copy.deepcopy(params)
        # set the layer type to be a ConvLayer
        self.params['internal_layer_type'] = [ConvLayer]

    def filter_dims(self, *filter_dimses):
        self.params['filter_dims'] = list(filter_dimses)
        return self
    
    def window(self, *windows):
        self.params['window'] = list(windows)
        return self

    def padding(self, *paddings):
        self.params['padding'] = list(paddings)
        return self
    
    def output_norm(self, *output_norms):
        self.params['output_norm'] = list(output_norms)
        return self


# utility "networks"
class Input: # holds the input info
    def __init__(self, covariate, input_dims):
        self.name = covariate
        self.index = -1 # index of network in the list of networks
        self.covariate = covariate
        self.input_dims = input_dims
        self.inputs = [] # this should always be empty for an Input
        self.output = None
        
        # this is a hack so that we can Cartesian product this along with the other nets
        # create a "virtual" layer
        self.layers = [Layer(params={'name': [self.name]})]

    def to(self, network):
        # set the input_dims of the network to be the desired Input.input_dims
        network.input_covariate = self.covariate
        network.layers[0].params['input_dims'] = [self.input_dims]
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
        
        # this is a hack so that we can Cartesian product this along with the other nets
        # create a "virtual" layer
        self.layers = [Layer(params={'name': [self.name]})]
    
    def __str__(self):
        return 'Output name='+self.name+', num_neurons='+str(self.num_neurons)
    

class Add:
    def __init__(self, networks=None, NLtype=NL.softplus.value):
        # TODO: this is a little hacky
        num_filters = None
        if networks is not None:
            num_filters = networks[0].layers[-1].params['num_filters']
            # make sure that the networks all have the same num_filters in their output
            for network in networks:
                assert len(network.layers[-1].params['num_filters']) == 1, "inputs into an Add must only have a single num_filter"
                assert network.layers[-1].params['num_filters'] == num_filters, "input networks must all have the same num_filters"

        self.name = '+'
        self.index = -1 # index of network in the list of networks
        self.inputs = networks
        self.output = None

        # NDN params
        self.input_covariate = None # this should always be None for an Operator
        self.ffnet_type = NetworkType.add.value

        # create a passthrough layer for the Sum
        self.layers = [PassthroughLayer(params={
            'num_filters': num_filters,
            'weights_initializer': 'ones',
            'NLtype': [NLtype],
            'bias': [True] # needs a bias since it is the only (e.g. last) layer
        })]

        # points its parents (the things to be summed) to this node
        if networks is not None:
            for network in networks:
                network.output = [self]

    def to(self, network):
        # if we are going to an output, update our num_filters to be the num_neurons
        if isinstance(network, Output):
            self.layers[-1].params['num_filters'] = [network.num_neurons]
        self.output = network
        network.inputs.append(self)

    def __str__(self):
        if self.inputs is not None:
            return 'Add name='+self.name+' '+str(self.index)+', inputs='+','.join([str([inp]) for inp in self.inputs])
        else:
            return 'Add name='+self.name+' '+str(self.index)


class Mult:
    def __init__(self, networks=None):
        num_filters = None
        # TODO: this is a little hacky
        if networks is not None:
            num_filters = networks[0].layers[-1].params['num_filters']
            # make sure that the networks all have the same num_filters in their output
            for network in networks:
                assert network.layers[-1].params['num_filters'] == num_filters, "input networks must all have the same num_filters"

        self.name = '*'
        self.index = -1 # index of network in the list of networks
        assert len(networks) > 1, 'At least 2 networks are required to Mult'
        self.networks_to_mult = list(networks)
        self.inputs = list(networks)
        self.output = None
        
        # NDN params
        self.input_covariate = None # this should always be None for an Operator
        self.ffnet_type = NetworkType.mult.value

        # create a passthrough layer for the Sum
        self.layers = [PassthroughLayer(params={
            'num_filters': num_filters,
            'weights_initializer': 'ones',
            'NLtype': [NLtype],
            'bias': [True] # needs a bias since it is the only (e.g. last) layer
        })]

        # points its parents (the things to be multiplied) to this node
        if networks is not None:
            for network in networks:
                network.output = [self]

    def to(self, network):
        # if we are going to an output, update our num_filters to be the num_neurons
        if isinstance(network, Output):
            self.layers[-1].params['num_filters'] = [network.num_neurons]
        self.output = network
        network.inputs.append(self)

    def __str__(self):
        if self.inputs is not None:
            return 'Mult name='+self.name+' '+str(self.index)+', inputs='+','.join([str([inp]) for inp in self.inputs])
        else:
            return 'Mult name='+self.name+' '+str(self.index)




# network
class Network:
    def __init__(self, layers, name=None, network_type=NetworkType.normal.value):
        # internal params
        self.model = None # to be able to point to the model we are a part of
        self.name = name
        self.index = -1 # index of network in the list of networks
        self.inputs = []
        self.output = None
        
        # NDNLayer params
        self.input_covariate = None # the covariate that goes to this network
        self.ffnet_type = network_type # default to being a normal network
        self.layers = layers
        for layer in self.layers:
            layer.network = self # point the layer back to the network it is a part of
    
    def add_layer(self, layer):
        self.layers.append(layer)
        layer.network = self
        
    def to(self, network):
        # if we are going to an output, update our num_filters to be the num_neurons 
        if isinstance(network, Output):
            if 'num_filters' in self.layers[-1].params:
                print('NUM_FILTERS', self.name, self.layers[-1].params['num_filters'])
            assert 'num_filters' not in self.layers[-1].params, 'num_filters should not be set on the last layer going to the Output'
            self.layers[-1].params['num_filters'] = [network.num_neurons]
        self.output = network
        network.inputs.append(self)

    def __str__(self):
        return 'Network name='+self.name+' '+str(self.index)+', len(layers)='+str(len(self.layers))+', inputs='+','.join([str([inp]) for inp in self.inputs])


# model
class Model:
    def __init__(self, output):
        self.NDN = None # set this Model's NDN to None to start
        # the template configuration for this model
        # to make it easy to compare hyperparameter differences between models
        # in a nice 2D combination chart!
        self.model_configuration = None

        self.inputs = []
        self.networks = []
        self.output = output
        
        # network.index --> network map
        netidx_to_model = {}
        
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
            netidx_to_model[idx] = network
            network.index = idx # set the reversed depth-first index
    
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

