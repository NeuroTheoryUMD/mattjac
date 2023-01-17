# utilities to make plotting data simpler and easier

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import NDNT.utils as utils # some other utilities
import math
import networkx as nx
import NDNT.NDNT as NDN


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


def plot_glm_filters(glm, nc):
    ws = glm.get_weights()
    utils.ss(5,6)

    for cc in range(nc):
        plt.subplot(5,6, cc+1)
        utils.imagesc(ws[:,:,cc])
        plt.colorbar()
    plt.show()


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



def plot_layers(layers):
    # plot receptive field at intermediate layers (weighted average of previous layers per column)
    plt.style.use('dark_background')
    
    shapes = [(1,8), (1,11)]
    for l in range(len(layers)): # go through each layer
        layer = layers[l]
        num_rows,num_cols = shapes[l]
    
        # multiply the previous layer features by the subunit's weighting of it
        if l == 0: # plot the first layer
            fig, axes = plt.subplots(num_rows,num_cols, figsize=(num_cols*2,num_rows*2))
            fig.suptitle('layer ' + str(l))
    
            layer = np.swapaxes(layer, 0,1) # rotate the image
            #g = isns.ImageGrid(layer, axis=len(layer.shape)-1, cbar=False, col_wrap=cols[l], cmap='inferno', vmax=imax, vmin=imin)
            for sub in range(layer.shape[-1]): # go through each subunit
                ax = axes.flatten()[sub]
                # to index the last axis for arrays with any number of axes
                subunit = layer.take(indices=sub, axis=len(layer.shape)-1) # equivalent to layer[:,:,sub]
                imin = np.min(subunit.flatten())
                imax = np.max(subunit.flatten())
                #isns.imshow(subunit, ax=axes.flatten()[sub], cbar=False, vmin=imin, vmax=imax)
                ax.set_axis_off() # remove axis
                ax.imshow(subunit, vmin=imin, vmax=imax, aspect='auto', cmap='inferno')
    
        else:
            fig, axes = plt.subplots(num_rows+1,num_cols,
                                     figsize=(num_cols*2,(num_rows+1)*1.2),
                                     gridspec_kw={'height_ratios': [1, 5]})
            fig.suptitle('layer ' + str(l))
    
            # plot the subunit weights
            axidx = 0
            for sub in range(layer.shape[1]): # for each subunit
                ax = axes.flatten()[axidx]
                weights = np.expand_dims(layer[:,sub], 0)
                imin = np.min(weights.flatten())
                imax = np.max(weights.flatten())
                ax.set_axis_off() # remove axis
                ax.imshow(weights, vmin=imin, vmax=imax, cmap='inferno')
                axidx += 1
    
            # plot the subunit weightings in the subsequent layers
            for sub in range(layer.shape[1]): # for each subunit
                ax = axes.flatten()[axidx]
                weighted_sub = np.mean(layer[:,sub] * layers[l-1], axis=len(layers[l-1].shape)-1)
                weighted_sub = np.swapaxes(weighted_sub, 0,1)
                imin = np.min(weighted_sub.flatten())
                imax = np.max(weighted_sub.flatten())
                ax.set_axis_off() # remove axis
                ax.imshow(weighted_sub, vmin=imin, vmax=imax, aspect='auto', cmap='inferno')
                axidx += 1
    
    plt.show()
