import sys
sys.path.insert(0, '../') # to have access to NDNT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from model import Input, Output
import predict
import torch
import os

import plotly.graph_objects as go
import plotly.express as px
from plotly import subplots
from scipy.optimize import linear_sum_assignment


# plotting methods to plot:
# - weights of a network
# - all networks in model
# - predictions per network
# - stimulus
# - robs
# - all at once in model

# light and dark modes
def lightmode():
    plt.style.use('default')
def darkmode():
    # use dark background for all our plots, because it is better
    plt.style.use('dark_background')
def graymode():
    plt.style.use('bmh') # Bayesian methods for hackers

def imagesc( img, ax=None, cmap=None, balanced=None, aspect=None, max=None, colrow=True, axis_labels=True, origin='lower'):
    """Modifications of plt.imshow that choose reasonable defaults"""
    if balanced is None:
        # Make defaults depending on img
        if np.sign(np.max(img)) == np.sign(np.min(img)):
            balanced = False
        else:
            balanced = True
    if balanced:
        imin = -np.max(abs(img))
        imax = np.max(abs(img))
    else:
        imin = np.min(img)
        imax = np.max(img)

    if max is not None:
        imin = -max
        imax = max

    if aspect is None:
        if img.shape[0] == img.shape[1]:
            aspect = 1
        else:
            aspect = 'auto'

    if colrow:  # then plot with first axis horizontal, second axis vertical
        if ax is None:
            plt.imshow( img.T, cmap=cmap, interpolation='none', aspect=aspect, vmin=imin, vmax=imax, origin=origin)
        else:
            ax.imshow( img.T, cmap=cmap, interpolation='none', aspect=aspect, vmin=imin, vmax=imax, origin=origin)
    else:  # this is like imshow: row, column
        if ax is None:
            plt.imshow( img, cmap=cmap, interpolation='none', aspect=aspect, vmin=imin, vmax=imax, origin=origin)
        else:
            ax.imshow( img, cmap=cmap, interpolation='none', aspect=aspect, vmin=imin, vmax=imax, origin=origin)
    if not axis_labels:
        figgy = plt.gca()
        figgy.axes.xaxis.set_ticklabels([])
        figgy.axes.yaxis.set_ticklabels([])
# END imagesc

def plot_aligned_filters(models, model_names=None, figsize=(10,5), cmap='gray'):
    """
    Plot the filters of the models in the same order as the first model.
    models: list of models to plot
    model_names: list of model names to use as row labels
    figsize: tuple of figure size (width, height)
    cmap: colormap to use, default is 'gray'
    """
    
    num_filters = None
    min_num_lags = None
    max_num_lags = None
    model_names = model_names
    filters = []
    num_inh = None

    # calculate the normalized cross-correlation between the first best model and the other models as pairs
    # and match them using the Hungarian algorithm (linear_sum_assignment)
    # then plot the filters in the same order
    
    # get the numbers of lags, filters, and model names
    for model in models:
        weights = model.networks[0].layers[0].weights
        filters.append(weights)
        
        if num_inh is None:
            num_inh = model.networks[0].layers[0].params['num_inh']    
        else:
            assert num_inh == model.networks[0].layers[0].params['num_inh'], 'The number of inhibitory neurons must be the same across models'

        if min_num_lags is None:
            min_num_lags = weights.shape[1]
        elif min_num_lags > weights.shape[1]:
            min_num_lags = weights.shape[1] # take the min

        if max_num_lags is None:
            max_num_lags = weights.shape[1]
        elif max_num_lags < weights.shape[1]:
            max_num_lags = weights.shape[1] # take the max

        if num_filters is None:
            num_filters = weights.shape[2]
        else:
            assert num_filters == weights.shape[2], 'The number of filters must be the same across models'

        if model_names is None:
            model_names = [model.name]
        else:
            model_names.append(model.name)
            
    # set the num_exc
    num_exc = num_filters - num_inh
    
    # expand the shorter filters to the same number of lags as the longest filter
    for i in range(len(filters)):
        if filters[i].shape[1] < max_num_lags:
            # pad with zeros
            filters[i] = np.concatenate((filters[i], np.zeros((filters[i].shape[0], max_num_lags - filters[i].shape[1], filters[i].shape[2]))), axis=1)
    
    # normalize the truncated_filters by max-min
    normalized_filters = []
    for i in range(len(filters)):
        normalized_filters.append((filters[i] - np.min(filters[i])) / (np.max(filters[i]) - np.min(filters[i])))
    
    # determine the best match for each filter
    assignments_exc = []
    assignments_inh = []
    assignment_costs_exc = []
    assignment_costs_inh = []
    for k in range(1, len(models)):
        # calculate the normalized cross-correlation between the first best model and the other models as pairs
        adjacency_exc = np.zeros((num_exc, num_exc))
        for i in range(num_exc):
            for j in range(num_exc):
                adjacency_exc[i,j] = np.corrcoef(normalized_filters[0][:,:,i].flatten(), normalized_filters[k][:,:,j].flatten())[0,1]
        
        adjacency_inh = np.zeros((num_inh, num_inh))
        for i in range(num_inh):
            for j in range(num_inh):
                adjacency_inh[i,j] = np.corrcoef(normalized_filters[0][:,:,i+num_exc].flatten(), normalized_filters[k][:,:,j+num_exc].flatten())[0,1]
    
        # align the truncated_filters using the Hungarian algorithm
        row_ind_exc, col_ind_exc = linear_sum_assignment(adjacency_exc, maximize=True)
        row_ind_inh, col_ind_inh = linear_sum_assignment(adjacency_inh, maximize=True)
        assignment_costs_exc.append(adjacency_exc[row_ind_exc, col_ind_exc])
        assignment_costs_inh.append(adjacency_inh[row_ind_inh, col_ind_inh])
        # add the num_exc to the col_ind_inh
        col_ind_inh = [i+num_exc for i in col_ind_inh]
        assignments_exc.append(col_ind_exc)
        assignments_inh.append(col_ind_inh)
    
    # order the assignments by the mean assignment_costs in descending order
    sorted_assignment_costs_exc = np.mean(assignment_costs_exc, axis=0)
    sorted_assignment_costs_exc = sorted_assignment_costs_exc.argsort()[::-1]
    sorted_assignments_exc = []
    for j in range(len(assignments_exc)):
        sorted_assignments_exc.append([assignments_exc[j][i] for i in sorted_assignment_costs_exc])
    
    sorted_assignment_costs_inh = np.mean(assignment_costs_inh, axis=0)
    sorted_assignment_costs_inh = sorted_assignment_costs_inh.argsort()[::-1]
    sorted_assignments_inh = []
    for j in range(len(assignments_inh)):
        sorted_assignments_inh.append([assignments_inh[j][i] for i in sorted_assignment_costs_inh])
        
    # combine the assignments
    sorted_assignments = [sorted_assignment_costs_exc.tolist() + sorted_assignment_costs_inh.tolist()]
    for i in range(len(sorted_assignments_exc)):
        sorted_assignments.append(sorted_assignments_exc[i] + sorted_assignments_inh[i])
    
    # get the max value across the truncated_filters
    max_vals = [np.max(filters[i]) for i in range(len(filters))]
    
    # plot the truncated_filters in the same order
    fig, axs = plt.subplots(len(filters), num_filters, figsize=figsize)
    for i in range(0, num_filters):
        imagesc(filters[0][:,:,sorted_assignments[0][i]], ax=axs[0,i], cmap=cmap, max=max_vals[0])
        # turn off the axes
        axs[0,i].set_xticklabels([])
        axs[0,i].set_yticklabels([])
        # label the y-axis
        if i == 0:
            axs[0,i].set_ylabel(model_names[0])
        for j in range(1, len(filters)):
            imagesc(filters[j][:,:,sorted_assignments[j][i]], ax=axs[j,i], cmap=cmap, max=max_vals[j])
            # turn off the axes
            axs[j,i].set_xticklabels([])
            axs[j,i].set_yticklabels([])
            # label the y-axis
            if i == 0:
                axs[j,i].set_ylabel(model_names[j])
    
    # label the excitatory columns and inhibitory columns
    for i in range(num_exc):
        axs[0,i].set_title('Excitatory', fontsize=14)
    for i in range(num_exc, num_filters):
        axs[0,i].set_title('Inhibitory', fontsize=14)
    
    fig.text(0.5, 0.04, 'Filters', ha='center', fontsize=14)
    fig.text(0, 0.5, 'Models', va='center', rotation='vertical', fontsize=14)
    plt.subplots_adjust(left=0.02) # shift the subplots left to make room for the y-axis labels
    fig.suptitle('Filters aligned across the models', fontsize=16)
    plt.show()


def plot_layer_weights(layer,
                       fig=None,
                       figsize=(5,10),
                       max_cols=8,
                       wspace=0.3,
                       hspace=0.3,
                       cmap='gray',
                       verbose=False):
    if fig is None:
        fig = plt.figure(figsize=figsize)

    if verbose:
        print(layer.shape, end=' --> ')

    # if it is 2D, make it 4D
    if len(layer.shape) == 2:
        # insert it at the beginning, so it matches the format of the 3D stuff
        layer = np.expand_dims(layer, axis=0)
        prev_weights = 1
        cur_weights = 1

    # if it is 3D, make it 4D
    elif len(layer.shape) == 3:
        layer = np.swapaxes(layer, 0, -1)
        prev_weights = 1
        cur_weights = layer.shape[0]

    # if it is 4D, swap the axes
    elif len(layer.shape) == 4:
        layer = np.swapaxes(layer, 0, 2)
        box_height, box_width, prev_weights, cur_weights = tuple(layer.shape)
        layer = layer.reshape(box_height, box_width, prev_weights*cur_weights)
        layer = np.swapaxes(layer, 0, -1)
        layer = np.swapaxes(layer, -2, -1)

    if verbose:
        print(layer.shape)

    # make the layer easier to iterate through and plot
    # (36, 10, 1, 1) --> (1, 1, 10, 36)
    # (36, 10, 8) --> (8, 10, 36) = 8 rows of images: 36 width x 10 height
    num_boxes = layer.shape[0]

    num_rows = 1
    num_cols = max_cols
    if num_boxes > max_cols:
        num_rows = num_boxes // max_cols + 1

    grid = plt.GridSpec(num_rows, num_cols, wspace=wspace, hspace=hspace)
    box_idx = 0
    prev_weight = 0
    cur_weight = 0
    for i in range(0, num_rows, 1):
        for j in range(0, num_cols):
            # stop plotting if there are no more subunits in the layer
            if box_idx == num_boxes:
                break

            box = layer[box_idx,:,:]

            box_ax = fig.add_subplot(grid[i,j])
            # to index the last axis for arrays with any number of axes
            imin = np.min(box.flatten())
            imax = np.max(box.flatten())
            box_ax.set_axis_off() # remove axis
            box_ax.imshow(box, vmin=imin, vmax=imax, interpolation='none', aspect='auto', cmap=cmap, origin='lower')
            box_ax.set_title('C'+str(cur_weight)+',P'+str(prev_weight), pad=10) # add padding to leave room for the title

            box_idx += 1 # move onto the next box
            prev_weight += 1
            if prev_weight == prev_weights:
                prev_weight = 0
                cur_weight += 1
            

def plot_network_weights(network, 
                         figsize=(5,10),
                         max_cols=8,
                         wspace=1,
                         hspace=5,
                         cmap='gray',
                         verbose=False):
    # make the figure and axes
    fig, axs = plt.subplots(nrows=len(network.layers), ncols=1,
                            #constrained_layout=True,
                            figsize=figsize)
    
    # plot the network name as the title
    fig.suptitle(network.name)

    # plot the layers and outputs
    current_row = 1
    for l in range(0, len(network.layers)): # go through each layer
        # https://stackoverflow.com/questions/52273546/matplotlib-typeerror-axessubplot-object-is-not-subscriptable
        if len(network.layers) > 1: # plt.subplots only returns a list of axs if nrows > 1
            # get the subplotspec specific to this subplot
            subplotspec = axs[l].get_subplotspec()
        else:
            subplotspec = axs.get_subplotspec()
        subfig = fig.add_subfigure(subplotspec)
        subfig.suptitle('Layer ' + str(l))

        layer = network.layers[l].weights
        
        plot_layer_weights(layer, subfig, figsize, max_cols, wspace, hspace, cmap, verbose)
        
        current_row += 1


def plot_model_weights(model,
                       figsize=(10,10),
                       max_cols=8,
                       wspace=0.5,
                       hspace=0.8,
                       cmap='gray',
                       verbose=False):
    for network in model.networks:
        if not isinstance(network, Input) and not isinstance(network, Output):
            plot_network_weights(network, figsize, max_cols, wspace, hspace, cmap, verbose)



def plot_stim(stim, 
              stim_dims,
              fig=None,
              title=None,
              figsize=(10,5)):
    # reshape the input to make it presentable
    inp = stim.numpy().reshape(stim_dims).T

    # make the figure and axes
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if title is not None:
        fig.suptitle(title)

    # plot the input
    fig.suptitle('Stimulus')
    input_ax = fig.add_subplot()
    # to index the last axis for arrays with any number of axes
    imin = np.min(inp.flatten())
    imax = np.max(inp.flatten())
    input_ax.set_axis_off() # remove axis
    # flip the input upside down
    input_ax.imshow(np.flipud(inp), 
                    vmin=imin, vmax=imax, interpolation='none',
                    aspect='auto', cmap='binary')


def plot_robs(robs, pred=None, smooth=None, neuron=None, figsize=(5, 10)):
    # plot the robs and prediction for a single neuron
    if neuron is not None:
        robs = robs[:,neuron]
        if pred is not None:
            pred = pred[:,neuron]
    
    robs_to_plot = robs
    pred_to_plot = None
    if pred is not None:
        pred_to_plot = pred
    if smooth is not None:
        # moving average of the robs
        robs_to_plot = np.convolve(robs, np.ones(smooth)/smooth)
        if pred is not None:
            pred_to_plot = np.convolve(pred, np.ones(smooth)/smooth)
    fig = plt.figure(figsize=figsize)
    plt.plot(robs_to_plot, label='robs')
    if pred is not None:
        # make the prediction use a dotted line
        plt.plot(pred_to_plot, label='pred', linestyle='dotted')
        plt.legend()
    # add a title
    if neuron is not None:
        plt.title('Neuron ' + str(neuron))
    else:
        plt.title('All neurons')
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing rate (Hz)')
    
    plt.show()


def plot_activity(trials, start=0, end=100, figsize=(30,20), title_param='name'):
    # compare the activity of the subunits and neurons for different reg values
    # (concatenate the weights from all the layers)
    # plot the weights for each trial
    fig = plt.figure(figsize=figsize)
    # turn gridlines off
    for i,trial in enumerate(trials):
        ax = fig.add_subplot(len(trials),1,i+1)
        ax.grid(False) # turn gridlines off
        model = trial.model
        results = predict.predict(model,
                                  dataset=trial.dataset[start:end],
                                  network_names_to_use=['core'],
                                  calc_jacobian=False)
        # only plot the first layer
        im = [np.squeeze(results.outputs[i]['core'][0]) for i in range(start, end)]
        imax = np.max(im)
        imin = -imax
        ax.imshow(im, cmap='gray', vmin=imin, vmax=imax)
        # choose the param
        if title_param == 'name':
            param_value = trial.name
        elif title_param == 'description':
            param_value = trial.description
        else:
            param_value = str(trial.trial_params[title_param])
        
        ax.set_title(title_param+' = '+param_value, fontsize=12)
    plt.show()
    

def plot_jacobians(results, model, frame, neuron=None, figsize=(20,10), max_cols=8):
    # set some params that we can pass in later
    network = 'core'
    width = 36
    lags = 10
    middle = width//2
    
    num_layers = len(model.networks_by_name[network].layers)
    num_subunits = [layer.params['num_filters'] for layer in model.networks_by_name[network].layers]

    # get the min and max of the jacobians for each layer
    imaxes = []
    for l in range(num_layers):
        imax = torch.max(results.jacobians[frame][network][l])
        imaxes.append(imax)
            
    cols = max_cols
    rows = sum(num_subunits[l] for l in range(num_layers))//cols + 1
    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(rows, cols)
    i = 0
    # plot the stim for the current time
    ax_stim = fig.add_subplot(grid[0, :])
    ax_stim.imshow(results.inps[frame].reshape(width,lags).T, cmap='gray', origin='lower')
    ax_stim.set_title('stimulus')
    
    # get the weights for a neuron from the readout layer
    if neuron is not None:
        readout_weights = model.networks_by_name['readout'].layers[0].weights[:,:,neuron]
    
    for l in range(num_layers):
        for u in range(num_subunits[l]):
            ax_cluster = fig.add_subplot(grid[i//cols+1, i%cols])
            subunit_jacobian = results.jacobians[frame][network][l][0,width*u:width*(u+1),:]
            if neuron is not None:
                # rescale the neuron weights to be between 0 and 1
                neuron_weights = torch.tensor(readout_weights[(l+1)*u,:])
                neuron_weights = (neuron_weights - torch.min(neuron_weights)) / (torch.max(neuron_weights) - torch.min(neuron_weights))
                # copy the readout weights to a new axis
                #neuron_weights = torch.unsqueeze(neuron_weights, 1)
                #neuron_weights = neuron_weights.expand(len(neuron_weights), lags)
                # multiply the subunit jacobian by the readout weights for the neuron
                subunit_jacobian = subunit_jacobian[middle].reshape(width,lags) * neuron_weights[middle]
            else:
                subunit_jacobian = subunit_jacobian[middle].reshape(width,lags)

            imax = torch.max(subunit_jacobian) #imaxes[l]
            imin = torch.min(subunit_jacobian) #-imax
            
            # plot the jacobian            
            ax_cluster.imshow(subunit_jacobian.T,
                              vmin=imin, vmax=imax,
                              interpolation='none',
                              cmap='gray',
                              origin='lower')
            # draw a vertical line at the middle point
            ax_cluster.axvline(x=middle, color='r', linestyle='--', alpha=0.7)
            ax_cluster.set_title('layer ' + str(l) + ' - subunit ' + str(u+1))
            i += 1
    plt.show()


def _extract_imframes(start, end, num_subunits, layer, width, lags, middle, results):
    imframes = []
    imax = 0
    for frame in range(start, end):
        subunit_jacobians = []
        for u in range(num_subunits):
            subunit_jacobian = results.jacobians[frame]['core'][layer][0,width*u:width*(u+1),:]
            im = subunit_jacobian[middle,:].reshape(width,lags).T # don't take the mean, just plot the center position
            imax = max(imax, torch.max(im))
            subunit_jacobians.append(im)
        imframes.append(subunit_jacobians)
    imax = float(imax.detach().numpy())
    imin = -imax
    return imframes, imax, imin

def plot_jacobians_animated(resultsA, modelA, start, end, layer, resultsB=None, modelB=None, max_cols=8, framerate=10):
    # set some params that we can pass in later
    network = 'core'
    width = 36
    lags = 10
    middle = width//2

    num_subunitsA = modelA.networks_by_name[network].layers[layer].params['num_filters']
    num_subunitsB = 0
    if resultsB is not None:
        num_subunitsB = modelB.networks_by_name[network].layers[layer].params['num_filters']
        
    num_subunits = num_subunitsA + num_subunitsB
    
    imframesA, imaxA, iminA, = _extract_imframes(start, end, num_subunitsA, layer, width, lags, middle, resultsA)
    imframesB = []
    imaxB = float('-inf')
    iminB = float('inf')
    if resultsB is not None:
        imframesB, imaxB, iminB = _extract_imframes(start, end, num_subunitsB, layer, width, lags, middle, resultsB)
        
    # combine the imframes
    imframes = []
    for i in range(len(imframesA)):
        imframes.append(imframesA[i] + imframesB[i])
    
    # get the smallest min and largest max
    imax = max(imaxA, imaxB)
    imin = min(iminA, iminB)

    cols = max_cols
    rows = (num_subunitsA+num_subunitsB) // cols + 1

    fig = subplots.make_subplots(rows=rows, cols=cols,
                        row_titles=['A']*(num_subunitsA//cols)+['B']*(num_subunitsB//cols))
    # change the color scheme of the figures
    fig.update_layout(coloraxis=dict(colorscale='gray'))
    
    # add traces for the subunits
    for i in range(num_subunits):
        row,col = np.unravel_index(i, (rows,cols))
        fig.add_trace(px.imshow(imframes[0][i], zmin=imin, zmax=imax).data[0],
                      row=row+1, col=col+1)
    
    frames = []
    for i in range(start, end):
        frame_data = []
        for j in range(num_subunits):
            frame_data.append(
                px.imshow(imframes[i][j], zmin=imin, zmax=imax)
                .update_traces(xaxis=f"x{j+1}", yaxis=f"y{j+1}")
                .data[0]
            )
        frames.append(
            go.Frame(
                name=str(i),
                data=frame_data,
            )
        )
    
    figa = go.Figure(data=fig.data, frames=frames, layout=fig.layout)
    
    # add slider
    figa.update_layout(
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "animation_frame="},
                "len": 0.9,
                "steps": [
                    {
                        "args": [
                            [fr.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "fromcurrent": True,
                            },
                        ],
                        "label": fr.name,
                        "method": "animate",
                    }
                    for fr in figa.frames
                ]
            }
        ]
    )
    
    # add play button
    figa.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 1000//framerate, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 300, "easing": "quadratic-in-out"},
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ]
    )
    figa.show()



########################################################################    


def simulate(model, results, timestep, context=100):
    # TODO: just iterate over the model inside the render_network function
    layers = []
    for n in range(len(model.NDN.networks)):
        for l in range(len(model.NDN.networks[n].layers)):
            layer = model.NDN.networks[n].layers[l].get_weights()
            layers.append(layer)
    
    # get the inps and robs at the given timestep
    # predict with this inps
    res = predict.predict(model, inps=inps, robs=robs)
    
    # render the network at that timestep
    # show some context around the robs for niceness
    return render_network(inps, stim_dims, res.outputs, layers)


def render_network(inp, stim_dims,
                   outputs, layers,
                   figsize=(5,5),
                   title=None,
                   max_cols=8,
                   cmap='gray',
                   linewidth=3,
                   linecolor='red',
                   wspace=0.4,
                   hspace=0.3,
                   verbose=False):
    # TODO: for scaffold networks,
    #       if the previous network is defined as a scaffold,
    #       then, concatenate all previous outputs together before calling forward
    # TODO: for parallel networks,
    #       come up with a way to visualize these parallel networks on the same row

    # reshape the input to make it presentable
    input_dims = stim_dims[1], stim_dims[3]
    inp = inp.numpy().reshape(input_dims).T

    # make the figure and axes
    fig, axs = plt.subplots(nrows=len(layers)+1, ncols=1,
                            constrained_layout=True,
                            figsize=figsize)
    if title is not None:
        fig.suptitle(title)

    # plot the input
    subplotspec = axs[0].get_subplotspec()
    subfig = fig.add_subfigure(subplotspec)
    subfig.suptitle('Stimulus')
    input_ax = subfig.add_subplot()
    # to index the last axis for arrays with any number of axes
    imin = np.min(inp.flatten())
    imax = np.max(inp.flatten())
    input_ax.set_axis_off() # remove axis
    # flip the input upside down
    input_ax.imshow(np.flipud(inp), vmin=imin, vmax=imax,
                    interpolation='none', aspect='auto', cmap='binary')

    # plot the layers and outputs
    current_row = 1
    for l in range(0, len(layers)): # go through each layer
        # get the subplotspec specific to this subplot
        subplotspec = axs[l+1].get_subplotspec()
        subfig = fig.add_subfigure(subplotspec)
        subfig.suptitle('Layer ' + str(l))

        layer = layers[l]
        output = outputs[l]

        # if it is 2D, make it 3D
        if len(layer.shape) == 2:
            # insert it in the middle so it matches the format of the 3D stuff
            layer = np.expand_dims(layer, 1)

        # make the layer easier to iterate through and plot
        # (36, 10, 8) --> (8, 10, 36) = 8 rows of images: 36 width x 10 height
        layer = np.swapaxes(layer, 0, 2)
        num_boxes = layer.shape[0]

        num_rows = 1
        num_cols = max_cols
        if num_boxes > max_cols:
            num_rows = num_boxes // max_cols + 1

        grid = plt.GridSpec(num_rows*2, num_cols, wspace=wspace, hspace=hspace)

        box_idx = 0
        for i in range(0, num_rows*2, 2):
            # increment the global row to keep track through the layers
            for j in range(0, num_cols):
                # stop plotting if there are no more subunits in the layer
                if box_idx == num_boxes:
                    break

                box = layer[box_idx,:,:]

                box_ax = subfig.add_subplot(grid[i,j])
                # to index the last axis for arrays with any number of axes
                imin = np.min(box.flatten())
                imax = np.max(box.flatten())
                box_ax.set_axis_off() # remove axis
                box_ax.imshow(box, vmin=imin, vmax=imax, interpolation='none', aspect='auto', cmap=cmap)

                output_ax = subfig.add_subplot(grid[i+1,j])
                # to index the last axis for arrays with any number of axes
                imin = np.min(box.flatten())
                imax = np.max(box.flatten())
                output_ax.set_axis_off() # remove axis
                output_ax.imshow(output, vmin=imin, vmax=imax, interpolation='none', aspect='auto', cmap=cmap)
                
                # TODO: add in robs and pred at the end
                
                

                # draw line between input and output
                # centerTopX = (box.shape[1]-1) // 2
                # centerTopY = box.shape[0]-1
                # centerBottomX = (output.shape[1]-1) // 2
                # centerBottomY = output.shape[0]-1
                # centerTopPoint = (centerTopX, centerTopY)
                # centerBottomPoint = (centerBottomX, centerBottomY)
                # TODO: fix the arrow drawing
                # con = ConnectionPatch(xyA=centerTopPoint, xyB=centerBottomPoint,
                #                       coordsA="data", coordsB="data",
                #                       axesA=box_ax, axesB=output_ax,
                #                       color=linecolor, arrowstyle='->', 
                #                       linewidth=linewidth)
                # output_ax.add_artist(con)

                box_idx += 1 # move onto the next box
        current_row += 2
    return fig
