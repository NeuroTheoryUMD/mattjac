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
        self.outputs = [] # layer outputs per network
        self.jacobian = None # this is the Jacobian for the entire model
        self.jacobians = None # the DSTRFs for each layer for each network
        self.inps = None # input (e.g. stim)
        self.inps_shape = () # shape of the input (e.g. stim_dims)
        self.robs = None # actual robs
        self.pred = None # predicted robs
        self.r2 = None # r2 is a measure of how well the model fits the data
        self.model = model


def _get_scaffold_outputs(model, all_outputs, inps, ni, li, input_width):
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
                tconv_layer_height = input_width * num_filters # use the first filter lag
                # add 1 to li to skip the input layer
                accumulated_prev_outputs.append(all_outputs[ni-1][li][:, :tconv_layer_height])
                #print('li', li, 'tconv_layer_height', tconv_layer_height)
            else:
                accumulated_prev_outputs.append(all_outputs[ni-1][li])
        #print('accumulated_prev_outputs', len(accumulated_prev_outputs))
        return torch.hstack(accumulated_prev_outputs)
    elif ni == 0 and li == 0: # we are in the first network, and the first layer, return the input
        return inps
    elif li > 0: # we are in the same network, but the next layer, return the previous layer's output
        return all_outputs[ni][li-1]
    elif li == 0: # we are in the next network, return the previous network's last layer's output
        return all_outputs[ni-1][-1]

@torch.no_grad() # disable gradient calculation during inference
def predict(model, inps=None, robs=None, dataset=None, verbose=False, calc_jacobian=False, max_network_and_layer=None) -> Results:
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
    input_width = model.networks[0].layers[0].params['input_dims'][1]
    if verbose: print('num_inps', num_inps)
    all_jacobians = [] # initialize all lists
    all_outputs = [] # initialize all lists
    # initialize the network lists
    for ni in range(len(model.networks)):
        # populate the network lists with the predictions
        all_outputs.append([])
        all_jacobians.append([])
        for li in range(len(model.networks[ni].layers)):
            # break if max_network_and_layer is reached
            if max_network_and_layer is not None and max_network_and_layer[0] > ni and max_network_and_layer[1] > li:
                break
            
            prev_output = _get_scaffold_outputs(model, all_outputs, inps, ni, li, input_width)
            
            if verbose:
                print('prev_output shape', prev_output.shape, 'ni', ni, model.networks[ni].network_type, 'li', li)

            should_calc_jacobian = False
            if calc_jacobian and max_network_and_layer is None:
                should_calc_jacobian = True
            elif calc_jacobian and max_network_and_layer[0] == ni and max_network_and_layer[1] == li:
                should_calc_jacobian = True

            if should_calc_jacobian:
                # calculate the Jacobian to get the DSTRF up through this layer
                def network_regular(inp):
                    all_outputs_jac = []
                    with torch.cuda.amp.autocast():
                        for nii in range(len(model.networks)):
                            all_outputs_jac.append([])
                            for lii in range(len(model.networks[nii].layers)):
                                prev_output_jac = _get_scaffold_outputs(model, all_outputs_jac, inp, nii, lii, input_width)
                                z_jac = model.NDN.networks[nii].layers[lii](prev_output_jac)
                                all_outputs_jac[nii].append(z_jac)
                                if nii == ni and lii == li:
                                    return z_jac
                
                layer_jacobians = []
                for i in tqdm.tqdm(range(len(inps))): # for each input
                    jacobian = torch.autograd.functional.jacobian(network_regular, inps[i], vectorize=True).cpu()
                    # reshape the jacobian to be number of filters by input shape
                    layer_jacobians.append(jacobian)
                # vertically stack the jacobians for each input
                stacked_jacobians = torch.vstack(layer_jacobians)
                num_timepoints = stacked_jacobians.shape[0]
                # reshape the jacobian based on the layer type
                if isinstance(model.networks[ni].layers[li], mod.TemporalConvolutionalLayer):
                    #jacobian = time x (filter_lags x num_filters x input_width) x input_size
                    num_filters = model.networks[ni].layers[li].params['num_filters']
                    filter_lags = 6
                    #stacked_jacobians = stacked_jacobians.reshape(num_timepoints, filter_lags, num_filters, input_width, -1)
                    stacked_jacobians = stacked_jacobians.reshape(num_timepoints, num_filters, input_width, filter_lags, -1)
                elif isinstance(model.networks[ni].layers[li], mod.IterativeTemporalConvolutionalLayer):
                    num_filters = model.networks[ni].layers[li].params['num_filters']
                    num_iter = model.networks[ni].layers[li].params['num_iter']
                    filter_lags = 1
                    stacked_jacobians = stacked_jacobians.reshape(num_timepoints, num_iter, num_filters, input_width, filter_lags, -1)
                elif isinstance(model.networks[ni].layers[li], mod.Network) and model.networks[ni-1].network_type == mod.NetworkType.scaffold:
                    num_filters = model.networks[ni].layers[li].params['num_filters']
                    stacked_jacobians = stacked_jacobians.reshape(num_timepoints, num_filters, -1)
                all_jacobians[ni].append(stacked_jacobians.detach().numpy())

            # calls the forward method of the layer
            z = model.NDN.networks[ni].layers[li](prev_output)
            
            all_outputs[ni].append(z)

            if verbose:
                print(prev_output.shape, '-->', z.shape)

    # add predicted robs for time
    pred = model.NDN({'stim': inps}).detach().numpy().reshape(robs.shape)
    
    # calculate and populate the r2
    robs = robs.detach().numpy() # detach the robs from the graph
    
    r2 = 1 - np.sum((robs - pred)**2, axis=0) / np.sum((robs - np.mean(robs))**2, axis=0)

    jacobian = []
    # if calc_jacobian:
    #     # calculate and populate the Jacobian for the entire model
    #     def model_stim(x):
    #         with torch.cuda.amp.autocast():
    #             return model.NDN({'stim': x})
    #     for i in tqdm.tqdm(range(len(inps))): # for each input
    #         jacobian.append(torch.autograd.functional.jacobian(model_stim, inps[i], vectorize=True).cpu())
    #     jacobian = torch.stack(jacobian).detach().numpy()
    #     jacobian = jacobian.squeeze()

    # reshape the outputs
    for ni in range(len(model.networks)):
        for li in range(len(model.networks[ni].layers)):
            # break if max_network_and_layer is reached
            if max_network_and_layer is not None and (ni, li) > max_network_and_layer:
                break
                
            output = all_outputs[ni][li]
            num_timepoints = output.shape[0]
            # reshape the output based on the layer type
            if isinstance(model.networks[ni].layers[li], mod.TemporalConvolutionalLayer):
                #jacobian = time x (filter_lags x num_filters x input_width) x input_size
                num_filters = model.networks[ni].layers[li].params['num_filters']
                filter_lags = 6
                output = output.reshape(num_timepoints, num_filters, input_width, filter_lags, -1)
            elif isinstance(model.networks[ni].layers[li], mod.IterativeTemporalConvolutionalLayer):
                num_filters = model.networks[ni].layers[li].params['num_filters']
                num_iter = model.networks[ni].layers[li].params['num_iter']
                filter_lags = 1
                output = output.reshape(num_timepoints, num_iter, num_filters, input_width, filter_lags, -1)
            elif isinstance(model.networks[ni].layers[li], mod.Network) and model.networks[ni-1].network_type == mod.NetworkType.scaffold:
                num_filters = model.networks[ni].layers[li].params['num_filters']
                output = output.reshape(num_timepoints, num_filters)
            all_outputs[ni][li] = output.detach().numpy()

    # populate the results object
    results = Results(model)
    results.inps = inps
    results.outputs = all_outputs
    results.jacobians = all_jacobians
    results.robs = robs
    results.pred = pred
    results.r2 = r2
    results.jacobian = jacobian
    return results


def predict_batch(model, dataset, end=None, calc_jacobian=False, verbose=False, batch_size=100000):
    # calculate the results of the model
    if end is None:
        end = dataset.NT

    results = None
    for nt in tqdm.tqdm(range(batch_size, end+1, batch_size)):
        batch_results = predict(model, dataset=dataset[nt-batch_size:nt], calc_jacobian=calc_jacobian, verbose=verbose)
        if results is None:
            results = batch_results
        else:
            # combine the properties in the results    
            for ni in range(len(results.outputs)):
                for li in range(len(results.outputs[ni])):
                    results.outputs[ni][li] = np.concatenate([results.outputs[ni][li], batch_results.outputs[ni][li]])
            results.robs = np.concatenate([results.robs, batch_results.robs], axis=0)
            results.pred = np.concatenate([results.pred, batch_results.pred], axis=0)
            results.r2 = np.sum([results.r2, batch_results.r2])
            
            if calc_jacobian:
                for ni in range(len(results.jacobians)):
                    for li in range(len(results.jacobians[ni])):
                        results.jacobians[ni][li] = np.concatenate([results.jacobians[ni][li], batch_results.jacobians[ni][li]], axis=0)
                results.jacobian = np.concatenate([results.jacobian, batch_results.jacobian], axis=0)
    results.r2 /= end//batch_size
    return results


def calc_preds(dataset, model, end=None, batch_size=100000):
    # calculate the results of the model
    if end is None:
        end = dataset.NT
    preds = []
    for nt in tqdm.tqdm(range(batch_size, end+1, batch_size)):
        output = model.NDN(dataset[nt-batch_size:nt])
        preds.append(output.detach().cpu().numpy())
    return np.concatenate(preds)


def calc_STAs(dataset, preds, end=None):
    # calculate the STA
    num_lags = dataset.num_lags
    Reff = dataset.robs * dataset.dfs
    nspks = torch.sum(Reff, axis=0)
    pred = (dataset.stim[:end].T@preds / nspks).reshape([-1, num_lags, dataset.NC]).detach().numpy()
    stas = (dataset.stim[:end].T@dataset.robs[:end] / nspks).reshape([-1, num_lags, dataset.NC]).detach().numpy()
    return stas, pred