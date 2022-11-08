import sys
import os
import h5py
import torch
import scipy.io as sio
import numpy as np
import pickle
import random
import networkx as nx
import matplotlib.pyplot as plt
import NTdatasets.HN.HNdatasets as datasets
import NDNT.utils as utils
import NDNT.NDNT as NDN
from NDNT.modules.layers import ChannelLayer, NDNLayer
from NDNT.networks import FFnetwork
from time import time
from torch import nn
from copy import deepcopy

device = torch.device("cuda:0")  # use the GPU
dtype = torch.float32  # make arrays be type float32

datadir = '../../data/hn/'
dirname = './savedir/'


def define_glm_layer(num_cells, stim_dims, as_layer=True):
    # if this is the only network to train, use a 'softplus' activation function
    # if this is part of a larger network, use a 'linear' activation function
    nl_type = 'lin' if as_layer else 'softplus'

    glm_layer = NDNLayer.layer_dict(
        input_dims=stim_dims,
        num_filters=num_cells,
        bias=True,
        initialize_center=True,
        NLtype=nl_type)

    # regularization method (use d2xt with lambda weight 0.02)
    glm_layer['reg_vals'] = {'d2xt': 0.02, 'bcs': {'d2xt': 1}}

    # only return the FFnetwork wrapped NDNLayer when combining this with other layers
    if as_layer:
        return FFnetwork.ffnet_dict(xstim_n='stim', layer_list=[glm_layer])
    else:
        return glm_layer


def define_comb_layer(num_cells):
    comb_layer = ChannelLayer.layer_dict(
        num_filters=num_cells,
        NLtype='softplus',
        bias=True)
    comb_layer['weights_initializer'] = 'ones'  # initialize to just pass the data through this "network"

    return FFnetwork.ffnet_dict(xstim_n=None, ffnet_n=[0, 1], layer_list=[comb_layer], ffnet_type='add')


def define_latent_layer(num_latents, num_cells, as_layer=True):
    L2reg = 0.01

    nl_type = 'lin' if as_layer else 'softplus'

    # input --> LVs (X --> Z)
    # encoder
    AClin_layer = NDNLayer.layer_dict(
        # what are these input_dims?
        input_dims=[num_cells, 1, 1, 1], num_filters=num_latents,  # num_LVs --> |Z|
        norm_type=1,  # normalization needed to keep the scale
        # this adjusts the activation threshold by making the output spikes more or less
        bias=False,  # don't put a bias on the input
        NLtype='lin')  # 'lin' -> linear, no nonlinearity applied here

    # LVs --> output (Z --> X')
    # decoder
    ACout_layer = NDNLayer.layer_dict(
        num_filters=num_cells,
        bias=True,
        NLtype=nl_type)

    # apply L2 regularization on the output
    # TODO: I don't understand why, though
    ACout_layer['reg_vals'] = {'l2': L2reg}

    # return the autoencoder as an FFnetwork (TODO: do we want to do this?)
    return FFnetwork.ffnet_dict(
        xstim_n='robs',
        layer_list=[AClin_layer, ACout_layer])


def fit_model(model, data):
    # define parameters for the LBFGS optimizer to fit the GLM
    lbfgs_pars = utils.create_optimizer_params(
        optimizer_type='lbfgs',
        tolerance_change=1e-10,
        tolerance_grad=1e-10,
        batch_size=2000,
        history_size=100,
        max_epochs=3,  # do we really only want to use a maximum of 3 epochs?
        max_iter=2000)

    # Fit the model
    model.fit(data, force_dict_training=True, **lbfgs_pars, verbose=1)

    # Evaluate model using null-adjusted log-likelihood
    return model.eval_models(data[data.val_inds], null_adjusted=True)


def train_autoencoder(data, num_latents):
    # define the untrained autoencoder model to learn the latent vars
    latent_layer = define_latent_layer(num_latents, data.NC, as_layer=False)

    latent_model = NDN.NDN(ffnet_list=[latent_layer], loss_type='poisson')

    # fit the model on the data
    lls = fit_model(latent_model, data)

    return lls, latent_model


# TRAIN THE GLM+AUTOENCODER MODEL
def train_autoencoder_and_glm(data, num_latents, pretrained_glm):
    # define the untrained GLM model
    # use a linear output since it gets combined with the autoencoder in the channel layer
    glm_layer = define_glm_layer(data.NC, data.stim_dims, as_layer=True)
    # define the untrained autoencoder model to learn the latent vars
    latent_layer = define_latent_layer(num_latents=num_latents, num_cells=data.NC)
    # define the untrained combined model
    comb_layer = define_comb_layer(data.NC)

    # define a model that combines info from the pretrained GLM and the autoencoder
    pretrained_glm_plus_latent_model = NDN.NDN(ffnet_list=[glm_layer, latent_layer, comb_layer], loss_type='poisson')

    # initialize GLM network with weights from the previously trained GLM
    pretrained_glm_plus_latent_model.networks[0].layers[0].weight = deepcopy(
        pretrained_glm.networks[0].layers[0].weight)

    # fit the model on the data
    lls = fit_model(pretrained_glm_plus_latent_model, data)

    return lls, pretrained_glm_plus_latent_model


def train_drift(data):
    lbfgs_pars = utils.create_optimizer_params(
        optimizer_type='lbfgs',
        tolerance_change=1e-10,
        tolerance_grad=1e-10,
        batch_size=2000,
        history_size=100,
        max_epochs=3,
        max_iter=2000)

    # Fit all cells at once
    Dreg = 0.005
    drift_pars = NDNLayer.layer_dict(
        input_dims=[1, 1, 1, data.Xdrift.shape[1]], num_filters=data.NC, bias=False, norm_type=0, NLtype='lin')
    drift_pars['reg_vals'] = {'d2t': Dreg, 'bcs': {'d2t': 0}}

    # for stand-alone drift model
    drift_parsN = deepcopy(drift_pars)
    drift_parsN['NLtype'] = 'softplus'

    drift_netN = FFnetwork.ffnet_dict(xstim_n='Xdrift', layer_list=[drift_parsN])

    # Stand-alone drift model
    drift_mod = NDN.NDN(ffnet_list=[drift_netN], loss_type='poisson')
    # Check out all the parameters in the model
    # drift_mod.list_parameters()
    drift_mod.fit(data, force_dict_training=True, **lbfgs_pars, verbose=0)
    # evaluate model using null-adjusted log-likelihood
    LLsNULL = drift_mod.eval_models(data[data.val_inds], null_adjusted=False)
    print("Mean LL-null:", np.mean(LLsNULL))

    return LLsNULL, drift_netN, drift_mod


def train_glm_with_drift(data, drift_net, drift_mod):
    lbfgs_pars = utils.create_optimizer_params(
        optimizer_type='lbfgs',
        tolerance_change=1e-10,
        tolerance_grad=1e-10,
        batch_size=2000,
        history_size=100,
        max_epochs=3,
        max_iter=2000)

    glm_layer_lin = NDNLayer.layer_dict(
        input_dims=data.stim_dims,
        num_filters=data.NC,
        bias=False, initialize_center=True,
        NLtype='lin')
    glm_layer_lin['reg_vals'] = {'d2xt': 0.02, 'bcs': {'d2xt': 1}}
    glm_net = FFnetwork.ffnet_dict(xstim_n='stim', layer_list=[glm_layer_lin])

    comb_layer = ChannelLayer.layer_dict(
        num_filters=data.NC, NLtype='softplus', bias=True)
    comb_layer['weights_initializer'] = 'ones'

    comb_net = FFnetwork.ffnet_dict(xstim_n=None, ffnet_n=[0, 1], layer_list=[comb_layer], ffnet_type='add')

    glmD0 = NDN.NDN(ffnet_list=[glm_net, drift_net, comb_net], loss_type='poisson')
    # Copy drift from drift_mod and do not fit
    # stuff the pretrained drift model weights into the drift_net component
    glmD0.networks[1].layers[0].weight.data = deepcopy(drift_mod.networks[0].layers[0].weight.data)
    glmD0.set_parameters(val=False, name='weight', ffnet_target=2)
    glmD0.set_parameters(val=False, ffnet_target=1)
    glmD0.fit(data, force_dict_training=True, **lbfgs_pars, verbose=0)
    # Evaluate model using null-adjusted log-likelihood
    LLs0 = glmD0.eval_models(data[data.val_inds], null_adjusted=False)
    return LLs0, glmD0


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    width: horizontal space allocated for this branch - avoids overlap with other branches
    vert_gap: gap between levels of hierarchy
    vert_loc: vertical location of root
    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed
        """
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


# TODO: use networkx to draw the network to see what it looks like
def draw_network(model, names):
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
