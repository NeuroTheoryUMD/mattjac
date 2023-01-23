# NDN tools
import copy

import NDNT.NDNT as NDN
from NDNT.modules.layers import *
from NDNT.networks import *

# powerful iteration tools!
# https://stackoverflow.com/questions/12237283/how-to-iterate-in-a-cartesian-product-of-lists
import itertools as it


# class to make it obvious that we are tyring multiple values
class Try:
    def __init__(self, *args):
        self.val = args
        
    def get(self):
        return self.val
    

class TestedNDN:
    def __init__(self, layer_list):
        self.all_layer_params = list(it.product(*[layer.get_params() for layer in layer_list]))
    
    def fit(self, datas, **adam_pars):
        self.experiments = [] # list of experiments
        
        # support list of datasets or a singular dataset
        if type(datas) is Try:
            datas = datas.get() # get the list
        else:
            datas = [datas] # make it into a list
        
        i = 0
        for data in datas: # for each dataset
            for layer_params in self.all_layer_params: # for each model
                layers = [NDNLayer.layer_dict(**{k:v for k,v in layer_param}) for layer_param in layer_params]
                layers[0]['input_dims'] = data.stim_dims # set input_dims
                layers[-1]['num_filters'] = data.NC # set output_dims
                model = NDN.NDN(layer_list=layers) # make the model
                
                # TODO: support multiple adam_pars
                model.fit(data, **adam_pars, verbose=2)
                self.experiments.append((i, model, data))
                i += 1
    
    def eval_models(self):
        # TODO: make this API look more like the NDNT API 
        self.LLs = []
        for name, model, data in self.experiments:
            self.LLs.append((name, model.eval_models(data, null_adjusted=True)))
        return self.LLs


# passes the NDN API through,
# allowing multiple params to be passed in instead of individual params
class TestedNDNLayer:
    def __init__(self):
        self.param_lists = []

    def layer_dict(self, **kwargs):
        for kwarg in kwargs:
            k = kwarg
            v = kwargs[kwarg]
            if type(v) is Try:
                self.param_lists.append([(k,val) for val in v.get()])
            elif type(v) is dict:
                # go through the vals to make sure those aren't lists
                for k1,v1 in v.items():
                    # TODO: support multiple nested dicts
                   ...
            else:
                self.param_lists.append([(k,v)])
    
    def get_params(self):           
        # need to use * to unpack the list of params as individual params for it.product() to work    
        return it.product(*self.param_lists)
