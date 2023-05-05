import sys
sys.path.insert(0, '../') # to have access to NDNT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import Input, Output
import predict
import torch
import os

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


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

def plot_layer_weights(layer,
                       fig=None,
                       figsize=(5,10),
                       max_cols=8,
                       wspace=0.3,
                       hspace=0.3,
                       cmap='gray'):
    if fig is None:
        fig = plt.figure(figsize=figsize)

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
            box_ax.imshow(box, vmin=imin, vmax=imax, interpolation='none', aspect='auto', cmap=cmap)
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
                         cmap='gray'):
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
        
        plot_layer_weights(layer, subfig, figsize, max_cols, wspace, hspace, cmap)
        
        current_row += 1


def plot_model_weights(model,
                       figsize=(10,10),
                       max_cols=8,
                       wspace=0.5,
                       hspace=0.8,
                       cmap='gray'):
    for network in model.networks:
        if not isinstance(network, Input) and not isinstance(network, Output):
            plot_network_weights(network, figsize, max_cols, wspace, hspace, cmap)



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


def plot_robs(robs, pred=None, smooth=None, figsize=(5,10)):
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
        plt.plot(pred_to_plot, label='pred')
        plt.legend()
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
    

def plot_jacobians():
    # TODO: clean up and make this work
    num_subunits = [32, 16, 8]
    rows = 8
    cols = 8
    for t in range(0,NT):
        fig = plt.figure(figsize=(20,10))
        grid = matplotlib.gridspec.GridSpec(rows, cols)
        i = 0
        # plot the stim for the current time
        ax_stim = fig.add_subplot(grid[0, :])
        ax_stim.imshow(inps[t].reshape(36,10).T, interpolation='none', cmap='gray', origin='lower')
        ax_stim.set_title('stimulus')
        for l in range(3):
            for u in range(num_subunits[l]):
                ax_cluster = fig.add_subplot(grid[i//cols+1, i%cols])
                subunit_jacobian = all_jacobians[t][model.networks[ni].name][l][0,36*u:36*(u+1),:]
                imax = torch.max(torch.abs(subunit_jacobian))
                imin = -imax
                ax_cluster.imshow(torch.mean(subunit_jacobian, axis=0).reshape(36,10).T,
                                  vmin=imin, vmax=imax,
                                  interpolation='none',
                                  cmap='gray',
                                  origin='lower')
                # draw a vertical line at point 15
                ax_cluster.axvline(x=15, color='r', linestyle='--')
                ax_cluster.set_title('layer ' + str(l) + ' - subunit ' + str(u+1))
                i += 1
    
        # ffmpeg requires that the digits start at 0
        plt.savefig(os.path.join('viz/dstrfs_layers', f'frame-{t:04d}.png'))
        plt.close(fig)
    
    
def plot_jacobians_animated():
    num_filterses = [16, 8, 8]

    imframes = []
    imax = 0
    for frame in range(start, end):
        #for layer in range(3):
        layer = 0
        subunit_jacobians = []
        for u in range(num_filterses[layer]):
            subunit_jacobian = results_more_reg.jacobians[frame]['core'][layer][0,36*u:36*(u+1),:]
            im = subunit_jacobian[15,:].reshape(36,10).T # don't take the mean, just plot the center position
            imax = max(imax, torch.max(im))
            subunit_jacobians.append(im)
        imframes.append(subunit_jacobians)
    imax = float(imax.detach().numpy())
    imin = -imax

    fig = make_subplots(rows=4, cols=4)
    # change the color scheme of the figures
    fig.update_layout(coloraxis=dict(colorscale='gray'))
    
    # add traces for the subunits
    for i in range(16):
        row,col = np.unravel_index(i, (4,4))
        fig.add_trace(px.imshow(imframes[0][i], zmin=imin, zmax=imax).data[0],
                      row=row+1, col=col+1)
    
    frames = []
    for i in range(start, end):
        frame_data = []
        for j in range(16):
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
                                "frame": {"duration": 500, "redraw": True},
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
