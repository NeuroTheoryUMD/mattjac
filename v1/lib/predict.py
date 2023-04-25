import sys
sys.path.insert(0, '../') # to have access to NTdatasets

import pandas as pd
import torch
import numpy as np
import model as mod
from NTdatasets.generic import GenericDataset


# contains model hierarchy with results, returned by model.predict(dataset)
class Results:
    def __init__(self, model):
        # results[frame:int][network_name:str][layer:int] = output:ndarray
        self._outputs = [] # layer outputs per network
        self.outputs_shape = ()
        self.jacobian = None # TODO: this is the Jacobian for the entire model
        self.jacobians = None # the DSTRFs for each layer for each network
        self.inps = None # input (e.g. stim)
        self.inps_shape = () # shape of the input (e.g. stim_dims)
        self.robs = None # actual robs
        self.pred = None # predicted robs
        self.r2 = None # TODO: r2 is a measure of how well the model fits the data
        self.model = model

    def _set_outputs(self, outputs):
        assert len(self.model.networks) > 0, 'model must have at least one network'
        networks0_name = self.model.networks[0].name
        self._outputs = outputs
        self.outputs_shape = (len(self._outputs), len(self._outputs[0]), len(self._outputs[0][networks0_name]), self._outputs[0][networks0_name][0].shape)
    
    def _get_outputs(self):
            return self._outputs

    outputs = property(_get_outputs, _set_outputs)


@torch.no_grad() # disable gradient calculation during inference
def predict(model, inps=None, robs=None, dataset=None, start=None, end=None,
            network_names_to_use=None, verbose=False) -> Results:
    assert (inps is not None and robs is not None) or (dataset is not None),\
           'either (inps and robs) or dataset is required'
    if dataset is not None:
        # handle data.dfs (valid data frames, very important)
        # element-wise multiply the stim by the valid frames
        # to keep only the valid datapoints
        robs = dataset['robs'] * dataset['dfs']
        inps = dataset['stim'] # TODO: should we remove any frame where a single neuron is invalid?
    
    # TODO: don't hardcode 'stim' down below,
    #       handle multiple covariates

    # TODO: for scaffold networks,
    #       if the previous network is defined as a scaffold,
    #       then, concatenate all previous outputs together before calling forward

    # get all the outputs
    num_inps = inps.shape[0]
    if verbose: print('num_inps', num_inps)
    prev_outputs = []
    all_outputs = [{} for _ in range(num_inps)] # initialize all lists
    all_jacobians = [{} for _ in range(num_inps)] # initialize all lists
    # initialize the network lists
    for ni in range(len(model.networks)):
        # skip networks that are not in the list of networks to use
        if network_names_to_use is not None and model.networks[ni].name not in network_names_to_use:
            continue
        
        # populate the lists
        for inpi in range(num_inps):
            # add list of network layer predictions by network name
            all_outputs[inpi][model.networks[ni].name] = []
            all_jacobians[inpi][model.networks[ni].name] = []
        # populate the network lists with the predictions
        for li in range(len(model.networks[ni].layers)):
            # concatenate the previous outputs if the previous network is a scaffold network
            if ni > 0 and model.networks[ni-1].network_type == mod.NetworkType.scaffold:
                print('concatenating scaffold network outputs')
                prev_output = torch.cat(prev_outputs, dim=1)
                prev_outputs = []
            elif ni == 0 and li == 0:
                prev_output = inps
            else:
                prev_output = prev_outputs[-1]
            
            print('prev_output shape', prev_output.shape, 'ni', ni, model.networks[ni].network_type, 'li', li)
            z = model.NDN.networks[ni].layers[li](prev_output)

            # calculate the Jacobian to get the DSTRF up through this layer
            for li in range(len(model.networks[ni].layers)):
                # calculate the Jacobian to get the DSTRF up through this layer
                def network_regular(x):
                    with torch.cuda.amp.autocast():
                        prev_z = x
                        for lii in range(li+1):
                            prev_z = model.NDN.networks[ni].layers[lii](prev_z)
                        return prev_z
                for i in range(len(inps)): # for each input
                    jacobian = torch.autograd.functional.jacobian(network_regular, inps[i], vectorize=True).cpu()
                    all_jacobians[i][model.networks[ni].name].append(jacobian)

            # TODO: concatenate the outputs of the layers if they are part of a scaffold network
            #       to return to the next network, don't just use the last layer's output

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
            prev_outputs.append(z_torch)

    # add predicted robs for time
    pred = model.NDN({'stim': inps}).detach().numpy()
    
    # calculate and populate the r2
    robs = robs.detach().numpy() # detach the robs from the graph
    r2 = 1 - np.sum((robs - pred)**2, axis=0) / np.sum((robs - np.mean(robs))**2, axis=0)
    
    # calculate and populate the Jacobian for the entire model
    def model_stim(x):
        with torch.cuda.amp.autocast():
            return model.NDN({'stim': x})
    jacobian = []
    for i in range(len(inps)):
        jacobian.append(torch.autograd.functional.jacobian(model_stim, inps[i], vectorize=True).cpu())

    # all_outputs[frame:int][network_name:str][layer:int] = output:ndarray
    results = Results(model)
    results.inps = inps
    results.outputs = all_outputs
    results.jacobians = all_jacobians
    results.robs = robs
    results.pred = pred
    results.r2 = r2
    results.jacobian = jacobian
    return results
