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
        

class Sum:
    def __init__(self, networks=None):
        self.name = '+'
        self.index = -1 # index of network in the list of networks
        self.inputs = networks
        self.output = None
        
        # this is a hack so that we can Cartesian product this along with the other nets
        # create a "virtual" layer
        self.layers = [Layer(params={'name': [self.name]})]
        
        # points its parents (the things to be summed) to this node
        if networks is not None:
            for network in networks:
                network.children = [self]

    def to(self, network):
        self.output = network
        network.inputs.append(self)


class Mult:
    def __init__(self, networks=None):
        self.name = '*'
        self.index = -1 # index of network in the list of networks
        assert len(networks) > 1, 'At least 2 networks are required to Mult'
        self.networks_to_mult = list(networks)
        self.inputs = list(networks)
        self.output = None
        
        # this is a hack so that we can Cartesian product this along with the other nets
        # create a "virtual" layer
        self.layers = [Layer(params={'name': [self.name]})]

        # points its parents (the things to be multiplied) to this node
        if networks is not None:
            for network in networks:
                network.children = [self]

    def to(self, network):
        self.output = network
        network.inputs.append(self)


# network
class Network:
    def __init__(self, layers, name=None):
        # internal params
        self.model = None # to be able to point to the model we are a part of
        self.name = name
        self.index = -1 # index of network in the list of networks
        self.inputs = []
        self.output = None
        
        # NDNLayer params
        self.ffnet_type = NetworkType.normal.value # default to being a normal network
        self.layers = layers
        for layer in self.layers:
            layer.network = self # point the layer back to the network it is a part of
        
    def network_type(self, network_type):
        self.ffnet_type = network_type.value
        return self
    
    def add_layer(self, layer):
        self.layers.append(layer)
        layer.network = self
        
    def to(self, network):
        self.output = network
        network.inputs.append(self)

    def __str__(self):
        return 'Network name='+self.name+', len(layers)='+str(len(self.layers))+', inputs='+','.join([str([inp]) for inp in self.inputs])


# model
class Model:
    def __init__(self, output):
        self.FFnet = None # set this Model's FFnet to None to start
        # the template configuration for this model
        # to make it easy to compare hyperparameter differences between models
        # in a nice 2D combination chart!
        self.model_configuration = None

        self.inputs = []
        self.networks = []
        self.output = output
        
        # network.index --> network map
        netidx_to_model = {}
        
        network_idx = 0
        for network in self.traverse():
            # add to the map
            netidx_to_model[network_idx] = network
            # create groups
            if isinstance(network, Output):
                continue
            if isinstance(network, Input):
                self.inputs.append(network)
            else: # if it is a Network, Sum, Mult or otherwise
                network.index = network_idx # set the depth-first index
                network.model = self # point the network back to the model
                self.networks.append(network)
                network_idx += 1

    def add_input(self, inp):
        self.inputs.append(inp)
    
    def add_network(self, network):
        self.networks.append(network)
        network.model = self

    def set_output(self, outp):
        self.output = outp
    
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

