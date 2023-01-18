# utilities to make plotting data simpler and easier

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import NDNT.utils as utils # some other utilities
import math
import networkx as nx
import NDNT.NDNT as NDN
from matplotlib.patches import ConnectionPatch
import torch
import tqdm
import os


# light and dark modes
def lightmode():
    plt.style.use('default')
def darkmode():
    # use dark background for all our plots, because it is better
    plt.style.use('dark_background')

# make new figure for simplicity
def fig(w=20, h=10):
    # close all figures if we have too many to avoid memory leak
    if plt.gcf().number > 20:
        plt.close("all")
    # make a new figure
    figure = plt.figure()
    figure.set_size_inches(w, h)  # width X height in inches
    return figure


# make a bunch of subplots of the provided vectors, and their names
def plots(*args, title=None):
    # show the legend if the labels are provided
    show_legend = True if type(args[0]) is tuple else False
    for i, arg in enumerate(args):
        plt.subplot(len(args), 1, i+1)
        # show the title if provided, but it has to be on top of the first subplot
        if i == 0 and title is not None:
            plt.title(title)
        if show_legend:
            plt.plot(arg[0], label=arg[1])
            plt.legend()
        else:
            plt.plot(arg)
    plt.show()


# turn a flat list of things into a grid of things
def list_to_grid(objects, width):
    height = math.ceil(len(objects) / width)
    grid = []
    for i in range(height):
        grid.append([])
        for j in range(width):
            idx = j + width*i
            if idx < len(objects):
                grid[i].append(objects[idx])
            else:
                grid[i].append(None)
    return grid, height


# use networkx to draw the network to see what it looks like
def draw_model(model, names=None):
    # put letters if no names are provided
    if names is None:
        names = [str(num) for num in range(len(model.networks))]
    
    # TODO: go through each network, and its layers and connect them into a network
    # TODO: add any attributes to the nodes and edges that make it clearer what they are doing
    assert type(model) == NDN.NDN, "model needs to be of type NDN"

    # define the network
    g = nx.DiGraph()

    # create the labels dictionary
    labels = {idx: name for idx, name in enumerate(names)}

    # go through the networks in the model
    for netidx, network in enumerate(model.networks):
        # add the network as a node
        g.add_node(netidx)

        # add edges between data and networks
        if network.xstim_n is not None:
            g.add_edge(network.xstim_n, netidx)
            # add the stim name to the map of node labels
            labels[network.xstim_n] = network.xstim_n

        # add edges between networks
        if network.ffnets_in is not None:
            for inputidx in network.ffnets_in:
                g.add_edge(inputidx, netidx)
                # attribute the edge if the network type is something
                # TODO: think about how to represent the attrs,
                #       but don't get too deep into this yet.
                #       Maybe I can use D3 for this to make it render nicer.
                if network.network_type is not None:
                    g[inputidx][netidx]['type'] = network.network_type
        # inputs to each network
        # QUESTION: can a network have ffnets_in and xstim_n?

    # draw the network
    # TODO: fix the layout to draw an inverted tree
    # nx.draw_networkx(g, pos=hierarchy_pos(g))
    nx.draw_networkx(g, labels=labels, with_labels=True)



# gets the position in a grid
def get_pos(shape, row, col, row_step=1):
    assert len(shape) == 2, "shape must have (num_rows, num_cols)"
    num_rows, num_cols = shape
    # divide row / row_step to get the actual number of rows
    # since each row has row_step # of things stacked in it
    actual_rows = row//row_step
    return actual_rows*num_cols + col

# plots the layers of the NIM that have been passed to it
# Example usage:
# layer0 = nim.networks[0].layers[0].get_weights()
# layer1 = nim.networks[0].layers[1].get_weights()
# output = nim.networks[0].layers[2].get_weights()
# m.plot_layers([layer0, layer1, output], shapes=[(1,8), (2,4), (2,6)])
def plot_layersNIM(layers, shapes):
    # plot receptive field at intermediate layers (weighted average of previous layers per column)
    prev_weighted_subs = []
    for l in range(len(layers)): # go through each layer
        # make the previous_weighted_subs more useable, if they exist yet
        if len(prev_weighted_subs) > 0:
            prev_weighted_subs = np.array(prev_weighted_subs)
            prev_weighted_subs = np.swapaxes(prev_weighted_subs, 0,2)
        # empty the current subs to be able to pass them along
        current_subs = []
        
        layer = layers[l]
        num_rows,num_cols = shapes[l]
    
        # multiply the previous layer features by the subunit's weighting of it
        if l == 0: # plot the first layer
            fig = plt.figure(figsize=(num_cols*2,num_rows*2))
            fig.suptitle('layer ' + str(l))
    
            layer = np.swapaxes(layer, 0,1) # rotate the image
            grid = plt.GridSpec(num_rows, num_cols, wspace=0.4, hspace=0.3)
            for i in range(num_rows):
                for j in range(num_cols):
                    pos = get_pos((num_rows,num_cols), i,j)
                    # stop plotting if there are no more subunits in the layer
                    if pos > layer.shape[-1]-1:
                        break
                    ax = fig.add_subplot(grid[i,j])
                    # to index the last axis for arrays with any number of axes
                    subunit = layer.take(indices=pos, axis=len(layer.shape)-1) # equivalent to layer[:,:,sub]
                    current_subs.append(subunit)
                    imin = np.min(subunit.flatten())
                    imax = np.max(subunit.flatten())
                    #isns.imshow(subunit, ax=axes.flatten()[sub], cbar=False, vmin=imin, vmax=imax)
                    ax.set_axis_off() # remove axis
                    ax.imshow(subunit, vmin=imin, vmax=imax, aspect='auto', cmap='inferno')
    
        else:
            fig = plt.figure(figsize=(num_cols*2,(num_rows+1)*1.2))
            fig.suptitle('layer ' + str(l))
    
            # plot the subunit weights
            grid = plt.GridSpec(num_rows*2, num_cols, wspace=0.4, hspace=0.3)
            
            num_subunits = layer.shape[-1]-1
            double_num_rows = num_rows*2 # to plot 2 things in each column
            for i in range(0, num_rows*2, 2): # go through two rows at a time
                for j in range(0, num_cols):
                    # draw the backward weights
                    # get the position given that we are stepping 2 for each col
                    pos = get_pos((num_rows,num_cols), i,j, row_step=2)
                    # stop plotting if there are no more subunits in the layer
                    if pos > num_subunits:
                        break
                    
                    ax = fig.add_subplot(grid[i,j])
                    # if the weights are 1D
                    if len(layer[:,pos].shape) == 1:
                        # expand their dimensions to make them 2D to plot
                        weights = np.expand_dims(layer[:,pos], 0)
                    else:
                        # if they are 2D, just display them
                        weights = layer[:,pos]
                    imin = np.min(weights.flatten())
                    imax = np.max(weights.flatten())
                    ax.set_axis_off() # remove axis
                    ax.imshow(weights, vmin=imin, vmax=imax, cmap='inferno')
            
                    # draw the weighted average of the previous filters
                    # as a linear-approximation of what this filter does
                    ax = fig.add_subplot(grid[i+1,j])
                    # TODO: this is gross
                    if l == 1:
                        weighted_sub = np.mean(layer[:,pos] * layers[l-1], axis=len(layers[l-1].shape)-1)
                    else:
                        weighted_sub = np.mean(weights * prev_weighted_subs, axis=len(layers[l-1].shape)-1)
                    current_subs.append(weighted_sub)
                    if l == 1:
                        # swap axes again if this is the second layer
                        weighted_sub = weighted_sub.swapaxes(0,1)
                    imin = np.min(weighted_sub.flatten())
                    imax = np.max(weighted_sub.flatten())
                    ax.set_axis_off() # remove axis
                    ax.imshow(weighted_sub, vmin=imin, vmax=imax, aspect='auto', cmap='inferno')
        
        # pass the current subs along to the next layer
        prev_weighted_subs = current_subs
    
    plt.show()


def simulate_network(inp, stim_dims,
                     model,
                     figsize=(5,5),
                     title=None,
                     max_cols=8,
                     cmap='gray',
                     linewidth=3,
                     linecolor='red',
                     wspace=0.4,
                     hspace=0.3,
                     verbose=False):
    # give the input an extra dimension for plotting
    inps = inp.unsqueeze(0)
    # pass none for savefig_loc since we don't expect to save the figure here
    # set start_idx to 0
    return simulate_network_batch(inps, stim_dims, model, 0, None,
                                  figsize, title, max_cols, cmap,
                                  linewidth, linecolor,
                                  wspace, hspace, verbose)
    

def simulate_network_batch(inps, stim_dims,
                           model,
                           start_idx=0,
                           savefig_loc=None,
                           figsize=(5,5),
                           title=None,
                           max_cols=8,
                           cmap='gray',
                           linewidth=3,
                           linecolor='red',
                           wspace=0.4,
                           hspace=0.3,
                           verbose=False):

    # if we are saving to video
    if savefig_loc is not None:
        model_filename = os.path.basename(savefig_loc)
        
        # make the savefig directory if it does not exist
        if not os.path.exists(savefig_loc):
            # make intermediate directories as well
            os.makedirs(savefig_loc)
        frame_dir = os.path.join(savefig_loc, model_filename+'_frames')
        if not os.path.exists(frame_dir):
            os.mkdir(frame_dir)

    # get all the layers
    # count layers to get number of rows
    # get the layers
    layers = []
    for n in range(len(model.networks)):
        for l in range(len(model.networks[n].layers)):
            layer = model.networks[n].layers[l].get_weights()
            layers.append(layer)

    # get all the outputs
    prev_output = inps
    all_outputs = []
    for n in range(len(model.networks)):
        for l in range(len(model.networks[n].layers)):
            z = model.networks[n].layers[l](prev_output)
            # TODO: not entirely sure if I need to detach twice
            z_cpu = torch.tensor(np.array([z_i.detach().numpy() for z_i in z]))
            all_outputs.append(z_cpu.numpy())
            if verbose:
                print(prev_output.shape, '-->', z_cpu.shape)
            prev_output = z_cpu
    
    figs = []
    for i in tqdm.tqdm(range(inps.shape[0])):
        inp = inps[i]
        # outputs is layers x num_inputs x dims
        # but we want, num_inputs x layers x dims
        # get the outputs for the current input i for each layer
        # expand each output dims to be 1xdim as well
        outputs = [np.expand_dims(all_outputs[l][i],0) for l in range(len(layers))]
        
        fig = render_network(inp, stim_dims,
                              outputs, layers,
                              figsize, title, max_cols, cmap,
                              linewidth, linecolor,
                              wspace, hspace, verbose)
        # save the fig to the frame directory
        if savefig_loc is not None:
            # add the start_idx so we can start at a later point in the inputs
            with open(os.path.join(frame_dir, str(i+start_idx)+'.png'), 'wb') as f:
                fig.savefig(f)
            plt.close(fig) # close the fig, so we don't render it every time
        figs.append(fig)

    # save to video with ffmpeg
    if savefig_loc is not None:
        os.system("ffmpeg -f image2 -r 5/1 -i ./"+savefig_loc+"/"+model_filename+"_frames/%d.png -vcodec mpeg4 -y "+savefig_loc+"/"+model_filename+".mp4")
    
    return figs
    

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
