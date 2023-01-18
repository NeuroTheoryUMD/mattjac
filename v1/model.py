import itertools as it
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
    

# builder pattern for constructing a layer with parameters
class Layer:
    def __init__(self):
        # nice defaults
        self.params = []
        
    def input_dims(self, *input_dimses):
        param_list = []
        for input_dims in input_dimses:
            param_list.append(('input_dims', input_dims))
        self.params.append(param_list)
        return self

    def num_filters(self, *num_filterses):
        param_list = []
        for num_filters in num_filterses:
            param_list.append(('num_filters', num_filters))
        self.params.append(param_list)
        return self

    def bias(self, *biases):
        param_list = []
        for bias in biases:
            param_list.append(('bias', bias))
        self.params.append(param_list)
        return self
    
    def norm_type(self, *norm_types):
        param_list = []
        for norm_type in norm_types:
            param_list.append(('norm_type', norm_type.value))
        self.params.append(param_list)
        return self

    def NLtype(self, *NLtypes):
        param_list = []
        for NLtype in NLtypes:
            param_list.append(('NLtype', NLtype.value))
        self.params.append(param_list)
        return self

    def initialize_center(self, *initialize_centers):
        param_list = []
        for initialize_center in initialize_centers:
            param_list.append(('initialize_center', initialize_center))
        self.params.append(param_list)
        return self
        
    def reg_vals(self, *reg_valses):
        param_list = []
        for reg_vals in reg_valses:
            param_list.append(('reg_vals', reg_vals))
        self.params.append(param_list)
        return self


class Model:
    def __init__(self):
        self.layers = []
        
    def add_network(self):
        # TODO support networks as well
        ...
        
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def build(self, data):
        # get tuples
        exploded_layers = [list(it.product(*layer.params)) for layer in self.layers]
        layer_groups = list(it.product(*exploded_layers))
        
        models = []
        for layers in layer_groups:
            NDNLayers = []
            for layer in layers:
                layer_dict = {k:v for k,v in layer}
                NDNLayers.append(NDNLayer.layer_dict(
                    input_dims=layer_dict['input_dims'], 
                    num_filters=layer_dict['num_filters'],
                    norm_type=layer_dict['norm_type'], 
                    NLtype=layer_dict['NLtype'], 
                    bias=layer_dict['bias'],
                    initialize_center=layer_dict['initialize_center'], 
                    reg_vals=layer_dict['reg_vals']))
                
            # set the input_dims of the first layer to the stim_dims
            NDNLayers[0]['input_dims'] = data.stim_dims
            # set the num_filters of the last layer to the data.NC
            NDNLayers[-1]['num_filters'] = data.NC
            
            models.append(NDN.NDN(layer_list=NDNLayers))
        
        return models
        
