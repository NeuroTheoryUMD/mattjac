import pandas as pd
import torch
import numpy as np
from NTdatasets.generic import GenericDataset


# contains model hierarchy with results, returned by model.predict(dataset)
class Results:
    def __init__(self, model):
        # results[frame:int][network_name:str][layer:int] = output:ndarray
        self._outputs = [] # layer outputs per network
        self.outputs_shape = ()
        self.inps = None # input (e.g. stim)
        self.robs = None # actual robs
        self.pred = None # predicted robs
        self.model = model

    def _set_outputs(self, outputs):
        assert len(self.model.networks) > 0, 'model must have at least one network'
        networks0_name = self.model.networks[0].name
        self._outputs = outputs
        self.outputs_shape = (len(self._outputs), len(self._outputs[0]), len(self._outputs[0][networks0_name]), self._outputs[0][networks0_name][0].shape)
    
    def _get_outputs(self):
            return self._outputs

    outputs = property(_get_outputs, _set_outputs)

def predict(inps, model, robs, device=torch.device("cuda:1"), verbose=False):
    # TODO: handle data.dfs (valid data frames, very important)
    # TODO: maybe take one that takes in a dataset,
    #       and one that takes in raw inps
    #(model({'stim': data.stim[test_inds]})  * data.dfs[test_inds]).detach().numpy()


    # TODO: for scaffold networks,
    #       if the previous network is defined as a scaffold,
    #       then, concatenate all previous outputs together before calling forward

    # get all the outputs
    num_inps = inps.shape[0]
    if verbose: print('num_inps', num_inps)
    prev_output = inps
    all_outputs = [{} for _ in range(num_inps)] # initialize all lists
    # initialize the network lists
    for ni in range(len(model.networks)):
        # populate the lists
        for inpi in range(num_inps):
            # add list of network layer predictions by network name
            all_outputs[inpi][model.networks[ni].name] = []
        # populate the network lists with the predictions
        for li in range(len(model.networks[ni].layers)):
            z = model.NDN.networks[ni].layers[li](prev_output)
            z_cpu = [z_i.detach().numpy() for z_i in z]
            z_torch = torch.tensor(np.array(z_cpu))
            # outputs is layers x num_inputs x dims
            # but we want, num_inputs x layers x dims
            # get the outputs for the current input inpi for each layer
            # expand each output dims to be 1xdim as well
            for i in range(num_inps): # for each input
                # add model predictions for this layer
                all_outputs[i][model.networks[ni].name].append(np.expand_dims(z_cpu[i], 0))
            if verbose:
                print(prev_output.shape, '-->', z_torch.shape)
            prev_output = z_torch

    # add predicted robs for time
    dataset = GenericDataset({'stim': inps}, device=device)
    pred = []
    #pred = model.NDN(dataset).detach().numpy()

    # all_outputs[frame:int][network_name:str][layer:int] = output:ndarray
    results = Results(model)
    results.inps = inps
    results.outputs = all_outputs
    results.robs = robs
    results.pred = pred
    return results