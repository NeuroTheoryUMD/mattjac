import sys
sys.path.insert(0, '../') # to have access to NTdatasets

import pandas as pd
import torch
import tqdm
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


def _get_scaffold_outputs(model, prev_outputs, ni, li, input_width=36):
    # check if the previous network is a scaffold network
    # and if so, combine the outputs of the previous network
    
    # if the previous network is a scaffold network,
    # then we need to get the output differently from the previous network
    if li == 0 and model.networks[ni-1].network_type == mod.NetworkType.scaffold:
        # go through the layers of the previous network, and combine the outputs
        # handle the Tconv network differently
        accumulated_prev_outputs = []
        for li in range(len(model.networks[ni-1].layers)):
            if isinstance(model.networks[ni-1].layers[li], mod.TemporalConvolutionalLayer):
                num_filters = model.networks[ni-1].layers[li].params['num_filters']
                tconv_layer_height = input_width * num_filters
                # add 1 to li to skip the input layer
                accumulated_prev_outputs.append(prev_outputs[li+1][:, :tconv_layer_height])
                print('li', li, 'tconv_layer_height', tconv_layer_height)
            else:
                accumulated_prev_outputs.append(prev_outputs[li+1])
        print('accumulated_prev_outputs', len(accumulated_prev_outputs))
        return torch.hstack(accumulated_prev_outputs)
    else:
        return prev_outputs[-1]

@torch.no_grad() # disable gradient calculation during inference
def predict(model, inps=None, robs=None, dataset=None,
            network_names_to_use=None, verbose=False, calc_jacobian=False) -> Results:
    """
    Predict the model outputs for the given inputs.
    Args:
        inps: If inps and robs are given, then the model is used to predict robs from inps.
        dataset: If dataset is given, then the model is used to predict robs from dataset['stim'].
        verbose: If verbose is True, then print out more information.
        calc_jacobian: If calc_jacobian is True, then calculate the Jacobian for each layer of each network.
    
    Returns:
         Results object
    """
    assert (inps is not None and robs is not None) or (dataset is not None),\
           'either (inps and robs) or dataset is required'
    if dataset is not None:
        # handle data.dfs (valid data frames, very important)
        # element-wise multiply the stim by the valid frames
        # to keep only the valid datapoints
        if 'robs' not in dataset:
            raise Exception('dataset must be sliced (e.g. dataset[0:10])')
        robs = dataset['robs'] * dataset['dfs']
        inps = dataset['stim'] # TODO: should we remove any frame where a single neuron is invalid?
    
    # TODO: don't hardcode 'stim' down below,
    #       handle multiple covariates

    # get all the outputs
    num_inps = inps.shape[0]
    if verbose: print('num_inps', num_inps)
    prev_outputs = [inps]
    all_jacobians = [{} for _ in range(num_inps)] # initialize all lists
    all_outputs = [{} for _ in range(num_inps)] # initialize all lists
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
            prev_output = _get_scaffold_outputs(model, prev_outputs, ni, li)
            
            if verbose:
                print('prev_output shape', prev_output.shape, 'ni', ni, model.networks[ni].network_type, 'li', li)

            # calls the forward method of the layer
            z = model.NDN.networks[ni].layers[li](prev_output)

            if calc_jacobian:
                # calculate the Jacobian to get the DSTRF up through this layer
                def network_regular(x):
                    prev_outputs_jac = [x]
                    with torch.cuda.amp.autocast():
                        for nii in range(len(model.networks)):
                            for lii in range(len(model.networks[nii].layers)):
                                prev_output_jac = _get_scaffold_outputs(model, prev_outputs_jac, nii, lii)
                                prev_outputs_jac.append(model.NDN.networks[nii].layers[lii](prev_output_jac))
                                if nii == ni and lii == li:
                                    return prev_outputs_jac[-1]
                
                for i in tqdm.tqdm(range(len(inps))): # for each input
                    jacobian = torch.autograd.functional.jacobian(network_regular, inps[i], vectorize=True).cpu()
                    all_jacobians[i][model.networks[ni].name].append(jacobian)

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

    jacobian = []
    if calc_jacobian:
        # calculate and populate the Jacobian for the entire model
        def model_stim(x):
            with torch.cuda.amp.autocast():
                return model.NDN({'stim': x})
        for i in tqdm.tqdm(range(len(inps))): # for each input
            jacobian.append(torch.autograd.functional.jacobian(model_stim, inps[i], vectorize=True).cpu())
        jacobian = torch.stack(jacobian).detach().numpy()

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
