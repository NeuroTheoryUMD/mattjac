import sys
import os
import h5py
import torch
import scipy.io as sio
import numpy as np
import pickle
import matplotlib.pyplot as plt
import NTdatasets.HN.HNdatasets as datasets
import NDNT.utils as utils
import NDNT.NDNT as NDN
from NDNT.modules.layers import * # it's generally not good practice to import *
from NDNT.networks import *
from time import time
from torch import nn
from copy import deepcopy

device = torch.device("cuda:0") # use the GPU
dtype = torch.float32 # make arrays be type float32

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
    glm_layer['reg_vals'] = {'d2xt': 0.02, 'bcs':{'d2xt': 1}}

    # only return the FFnetwork wrapped NDNLayer when combining this with other layers
    if as_layer:
        return FFnetwork.ffnet_dict(xstim_n = 'stim', layer_list = [glm_layer])
    else:
        return glm_layer


def define_comb_layer(num_cells):
    comb_layer = ChannelLayer.layer_dict(
        num_filters = num_cells,
        NLtype='softplus',
        bias=True)
    comb_layer['weights_initializer'] = 'ones' # initialize to just pass the data through this "network"

    return FFnetwork.ffnet_dict(xstim_n = None, ffnet_n=[0,1], layer_list = [comb_layer], ffnet_type='add')


def define_latent_layer(num_latents, num_cells, as_layer=True):
    L2reg = 0.01

    nl_type = 'lin' if as_layer else 'softplus'

    # input --> LVs (X --> Z)
    AClin_layer = NDNLayer.layer_dict(
        # what are these input_dims?
        input_dims=[num_cells, 1, 1, 1], num_filters=num_latents, # num_LVs --> |Z|
        norm_type=1, # normalization needed to keep the scale
        # this adjusts the activation threshold by making the output spikes more or less
        bias=False, # don't put a bias layer on the input
        NLtype='lin') # 'lin' -> linear layer

    # LVs --> output (Z --> X')
    ACout_layer = NDNLayer.layer_dict(
        num_filters=num_cells,
        bias=True,
        NLtype=nl_type)

    # apply L2 regularization on the output for some reason
    ACout_layer['reg_vals'] = {'l2': L2reg}

    # return the autoencoder as an FFnetwork (TODO: do we want to do this?)
    return FFnetwork.ffnet_dict(
        xstim_n = 'robs',
        layer_list = [AClin_layer, ACout_layer])


def fit_model(model, data):
    # define parameters for the LBFGS optimizer to fit the GLM
    lbfgs_pars = utils.create_optimizer_params(
        optimizer_type='lbfgs',
        tolerance_change=1e-10,
        tolerance_grad=1e-10,
        batch_size=2000,
        history_size=100,
        max_epochs=3, # do we really only want to use a maximum of 3 epochs?
        max_iter = 2000)

    # Fit the model
    model.fit(data, force_dict_training=True, **lbfgs_pars, verbose=1)

    # Evaluate model using null-adjusted log-likelihood
    return model.eval_models(data[data.val_inds], null_adjusted=True)


def train_autoencoder(data, num_latents):
    # define the untrained autoencoder model to learn the latent vars
    latent_layer = define_latent_layer(num_latents, data.NC, as_layer=False)

    latent_model = NDN.NDN(ffnet_list = [latent_layer], loss_type='poisson')

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
    pretrained_glm_plus_latent_model = NDN.NDN(ffnet_list = [glm_layer, latent_layer, comb_layer], loss_type='poisson')

    # initialize GLM network with weights from the previously trained GLM
    pretrained_glm_plus_latent_model.networks[0].layers[0].weight = deepcopy(pretrained_glm.networks[0].layers[0].weight)

    # fit the model on the data
    lls = fit_model(pretrained_glm_plus_latent_model, data)

    return lls, pretrained_glm_plus_latent_model


