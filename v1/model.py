import itertools as it
import networkx as nx

from enum import Enum

import NDNT.NDNT as NDN
from NDNT.modules.layers import *
from NDNT.networks import *



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
class Layer:
    def __init__(self):
        # nice defaults
        self.params = {
            # internal params (b/c only the dictionary is passed around)
            # TODO: this is gross, but it's OK for now...
            'internal_layer_type': [NDNLayer]
        }

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
        for param, vals in layer.params.items():
            self.params[param] = vals
        return self
    
    def build(self):
        # convert the dictionary of lists into a list of lists of tuples
        return [[(k,v) for v in vs] for k,vs in self.params.items()]


# layer subclasses
class ConvolutionalLayer(Layer):
    def __init__(self):
        super().__init__()
        # pass the class so that we create the right layer class
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
        self.covariate = covariate
        self.input_dims = input_dims
        self.children = []
        self.parents = []
        
    def to(self, *networks):
        self.children = list(networks)
        # set this as the parent
        for network in networks:
            network.parents.append(self)


class Output: # holds the output info
    def __init__(self, num_neurons):
        self.name = 'output'
        self.num_neurons = num_neurons
        self.children = []
        self.parents = []
        

class Sum:
    def __init__(self, *networks):
        self.name = '+'
        assert len(networks) > 1, 'At least 2 networks are required to Sum'
        self.networks_to_sum = list(networks)
        self.children = []
        self.parents = list(networks)
        # points its parents (the things to be summed) to this node
        for network in networks:
            network.children = [self]

    def to(self, *networks):
        self.children = list(networks)
        # set this as the parent
        for network in networks:
            network.parents.append(self)


class Mult:
    def __init__(self, *networks):
        self.name = '*'
        assert len(networks) > 1, 'At least 2 networks are required to Mult'
        self.networks_to_mult = list(networks)
        self.children = []
        self.parents = []

# network
class Network:
    def __init__(self, name, *layers):
        # internal params
        self.name = name
        self.parents = []
        self.children = []
        
        # NDNLayer params
        self.ffnet_type = NetworkType.normal.value # default to being a normal network
        self.xstim_n = 'stim' # default to using the stim
        self.layers = list(layers)
        
    def network_type(self, network_type):
        self.ffnet_type = network_type.value
        return self
    
    def stim(self, stim):
        self.xstim_n = stim
        return self
        
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def to(self, *networks):
        self.children = list(networks)
        # set this as the parent
        for network in networks:
            network.parents.append(self)


# model
class Model:
    def __init__(self, *networks):
        # TODO: change this to self.inputs
        self.networks = list(networks)
    
    def add_network(self, network):
        self.networks.append(network)

    def build(self, data):
        # TODO: don't use a list of NDNs,
        # instead wrap the NDN in a container that also has its network topology
        # and network walking APIs
        NDNs = []
        
        networks_with_exploded_layers = []
        for network in self.networks:
            exploded_layers = [list(it.product(*layer.build())) for layer in network.layers]
            layer_groups = list(it.product(*exploded_layers))
            networks_with_exploded_layers.append(layer_groups)
        
        network_groups = list(it.product(*networks_with_exploded_layers))

        for network_group in network_groups:
            prev_net = -1

            FFnetworks = []
            for network, layers in zip(self.networks, network_group):
                NDNLayers = []
                for layer in layers:
                    # get params to pass
                    layer_params = {}
                    for k,v in layer:
                        # skip internal params
                        if not 'internal' in k:
                            layer_params[k] = v
                    
                    layer_dict = {k:v for k,v in layer}
                    layer_type = layer_dict['internal_layer_type']
                    NDNLayers.append(layer_type.layer_dict(**layer_params))
                
                # if we are on the last network
                # TODO: make this work better for more complicated topologies
                if prev_net + 1 == len(self.networks) - 1:
                    NDNLayers[-1]['num_filters'] = data.NC
                
                if prev_net < 0: # if we are on the first network
                    # set the input_dims of the first layer to the stim_dims
                    NDNLayers[0]['input_dims'] = data.stim_dims
                    FFnetworks.append(FFnetwork.ffnet_dict(
                        xstim_n=network.xstim_n,
                        layer_list=NDNLayers,
                        ffnet_type=network.ffnet_type))
                else: # if we are not on the first network
                    FFnetworks.append(FFnetwork.ffnet_dict(
                        xstim_n=network.xstim_n,
                        ffnet_n=[prev_net], # connect up to previous network, but in future walk the tree
                        layer_list=NDNLayers,
                        ffnet_type=network.ffnet_type))
                prev_net += 1
    
            NDNs.append(NDN.NDN(ffnet_list=FFnetworks))
            
        return NDNs

    
    def draw_network(self):
        g = nx.DiGraph()
        
        # find the roots
        roots = []
        for node in self.networks:
            if isinstance(node, Input):
                roots.append(node)
        
        print([root.name for root in roots])
        
        # depth-first traverse the network from each root
        # and construct the network
        nodes_to_visit = roots # stack
        while len(nodes_to_visit) > 0:
            node = nodes_to_visit.pop()
            nodes_to_visit.extend(node.children)
            print(node.name)
            
            # add the node to the graph
            for child in node.children:
                g.add_edge(node.name, child.name)
                
        # draw graph
        nx.draw_networkx(g, with_labels=True)
