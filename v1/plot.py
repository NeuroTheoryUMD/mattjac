import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import Input, Output


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

def plot_layer_weights(layer,
                       fig=None,
                       figsize=(5,10),
                       max_cols=8,
                       wspace=0.3,
                       hspace=0.3,
                       cmap='gray'):
    if fig is None:
        fig = plt.figure(figsize=figsize)

    # if it is 2D, make it 3D
    if len(layer.shape) == 2:
        # insert it in the middle, so it matches the format of the 3D stuff
        layer = np.expand_dims(layer, 1)

    # make the layer easier to iterate through and plot
    # (36, 10, 8) --> (8, 10, 36) = 8 rows of images: 36 width x 10 height
    layer = np.swapaxes(layer, 0, 2)
    num_boxes = layer.shape[0]

    num_rows = 1
    num_cols = max_cols
    if num_boxes > max_cols:
        num_rows = num_boxes // max_cols + 1

    grid = plt.GridSpec(num_rows, num_cols, wspace=wspace, hspace=hspace)

    box_idx = 0
    for i in range(0, num_rows, 1):
        # increment the global row to keep track through the layers
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
            box_ax.imshow(box, vmin=imin, vmax=imax, aspect='auto', cmap=cmap)

            box_idx += 1 # move onto the next box
            

def plot_network_weights(network, 
                         figsize=(10,10),
                         max_cols=8,
                         wspace=0.5,
                         hspace=0.3,
                         cmap='gray'):
    # make the figure and axes
    fig, axs = plt.subplots(nrows=len(network.layers), ncols=1,
                            constrained_layout=True,
                            figsize=figsize)
    
    # plot the network name as the title
    fig.suptitle(network.name)

    # plot the layers and outputs
    current_row = 1
    for l in range(0, len(network.layers)): # go through each layer
        # get the subplotspec specific to this subplot
        subplotspec = axs[l].get_subplotspec()
        subfig = fig.add_subfigure(subplotspec)
        subfig.suptitle('Layer ' + str(l))

        layer = network.layers[l].weights
        
        plot_layer_weights(layer, subfig, figsize, max_cols, wspace, hspace, cmap)
        
        current_row += 1

        
def plot_model_weights(model,
                       figsize=(10,10),
                       max_cols=8,
                       wspace=0.5,
                       hspace=0.3,
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
                    vmin=imin, vmax=imax,
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
    
    

########################################################################    



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
                    aspect='auto', cmap='binary')

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
                box_ax.imshow(box, vmin=imin, vmax=imax, aspect='auto', cmap=cmap)

                output_ax = subfig.add_subplot(grid[i+1,j])
                # to index the last axis for arrays with any number of axes
                imin = np.min(box.flatten())
                imax = np.max(box.flatten())
                output_ax.set_axis_off() # remove axis
                output_ax.imshow(output, vmin=imin, vmax=imax, aspect='auto', cmap=cmap)

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
