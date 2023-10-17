import sys
sys.path.append('../')

import numpy as np
import optuna
import torch
import pickle
import glob
import os
import itertools

# NDN tools
import NDNT.utils as utils # some other utilities
import NDNT.NDNT as NDN
from NDNT.modules.layers import *
from NDNT.networks import *

from ColorDataUtils.multidata_utils import MultiExperiment
from ColorDataUtils.model_validation import validate_model, MockTrial

device = torch.device("cuda:1")
dtype = torch.float32

datadir = '/Data/ColorV1/'




##################################################
folder_name = '../models/ori_09a'
proj_filter_cents = [0.0001, 0.001, 0.01]
proj_filter_width = [17, 19, 21, 23]
num_filters_projs = [24, 20]
num_filters_conv0 = [16]
num_filters_conv1 = [12,  8]
expts_names = ['J220715', 'J220722', 'J220707', 'J220801']
expts_array_types = ['UT', 'UT', 'UT', 'UT']
#expts_names = ['J220715']
#expts_array_types = ['UT']
##################################################


hyperparams = itertools.product(*[proj_filter_cents,
                                 proj_filter_width,
                                 num_filters_projs,
                                 num_filters_conv0,
                                 num_filters_conv1])

# load data
expts = MultiExperiment(names=expts_names,
                        datadir=datadir,
                        num_lags=16,
                        et_metric_thresh=0.8,
                        array_types=expts_array_types,
                        luminance_only=True)
data, drift_terms, mu0s = expts.load()


# define params
adam_parsT = utils.create_optimizer_params(
    optimizer_type='AdamW',
    batch_size=2, # * 240 timesteps
    num_workers=0,
    learning_rate=0.0017,
    early_stopping_patience=4,
    optimize_graph=False,
    weight_decay=0.235)

adam_parsT['device'] = device

# setup
data.device = device
NCv = data.NC
NT = data.robs.shape[0]
NA = data.Xdrift.shape[1]


# some good starting parameters
Treg = 0.01
Mreg = 0.0001
Creg = None
Dreg = 0.5


# load previous model
cnn_baseline32 = NDN.NDN.load_model('../models/ori_01/model_baseline32.pkl')

for cproj, wproj, fproj, fconv0, fconv1 in hyperparams:
    # define model
    lgn_layer = STconvLayer.layer_dict(
        input_dims = data.stim_dims,
        num_filters=4,
        bias=False,
        norm_type=1,
        filter_dims=[1,  # channels
                     9,  # width
                     9,  # height
                     14], # lags
        NLtype='relu',
        initialize_center=True)
    lgn_layer['output_norm']='batch'
    lgn_layer['window']='hamming'
    lgn_layer['reg_vals'] = {'d2t':Treg,
                             'center': Creg, # None
                             'edge_t':100} # just pushes the edge to be sharper

    proj_layer = OriConvLayer.layer_dict(
    #proj_layer = ConvLayer.layer_dict(
        num_filters=fproj,
        bias=False,
        norm_type=1,
        num_inh=fproj//2,
        filter_dims=wproj,
        #filter_width=wproj,
        NLtype='lin',
        initialize_center=True,
        angles=[0, 30, 60, 90, 120, 150])
    proj_layer['window']='hamming'
    proj_layer['reg_vals'] = {'center': cproj}
    proj_layer['output_norm'] = 'batch'

    conv3D_layer0 = ConvLayer3D.layer_dict(
    #conv3D_layer0 = ConvLayer.layer_dict(
        num_filters=fconv0,
        num_inh=fconv0//2,
        bias=False,
        norm_type=1,
        filter_width=7,
        NLtype='relu',
        initialize_center=False)
    conv3D_layer0['output_norm'] = 'batch'
    
    conv3D_layer1 = ConvLayer3D.layer_dict(
    #conv3D_layer1 = ConvLayer.layer_dict(
        num_filters=fconv1,
        num_inh=fconv1//2,
        bias=False,
        norm_type=1,
        filter_width=5,
        NLtype='relu',
        initialize_center=False)
    conv3D_layer1['output_norm'] = 'batch'

    scaffold_net = FFnetwork.ffnet_dict(
        ffnet_type='scaffold',
        xstim_n='stim',
        layer_list=[lgn_layer, proj_layer, conv3D_layer0, conv3D_layer1],
        scaffold_levels=[1,2,3],
        num_lags_out=None)

    ## 1: READOUT
    # reads out from a specific location in the scaffold network
    # this location is specified by the mus
    readout_pars = ReadoutLayer.layer_dict(
        num_filters=NCv,
        NLtype='lin',
        bias=False,
        pos_constraint=True)
    # for defining how to sample from the mu (location) of the receptive field
    readout_pars['gauss_type'] = 'isotropic'
    readout_pars['reg_vals'] = {'max': Mreg}

    readout_net = FFnetwork.ffnet_dict(
        xstim_n = None,
        ffnet_n=[0],
        layer_list = [readout_pars],
        ffnet_type='readout')

    ## 2: DRIFT
    drift_pars = NDNLayer.layer_dict(
        input_dims=[1,1,1,NA],
        num_filters=NCv,
        bias=False,
        norm_type=0,
        NLtype='lin')
    drift_pars['reg_vals'] = {'d2t': Dreg}

    drift_net = FFnetwork.ffnet_dict(xstim_n = 'Xdrift', layer_list = [drift_pars])

    ## 3: COMB 
    comb_layer = ChannelLayer.layer_dict(
        num_filters=NCv,
        NLtype='softplus',
        bias=True)
    comb_layer['weights_initializer'] = 'ones'

    comb_net = FFnetwork.ffnet_dict(
        xstim_n = None,
        ffnet_n=[1,2],
        layer_list=[comb_layer],
        ffnet_type='add')

    cnn = NDN.NDN(ffnet_list = [scaffold_net, readout_net, drift_net, comb_net],
                    loss_type='poisson')
    cnn.block_sample = True

    # set the weights for the LGN layer
    cnn.networks[0].layers[0].weight.data = cnn_baseline32.networks[0].layers[0].weight.data.clone().detach()
    #cnn.networks[0].layers[0].set_parameters(val=False) # don't fit these weights

    ## Network 1: readout: fixed mus / sigmas
    cnn.networks[1].layers[0].sample = False
    # mus and sigmas are the centers and "widths" of the receptive field center to start at
    cnn.networks[1].layers[0].mu.data = torch.tensor(mu0s, dtype=torch.float32)
    cnn.networks[1].set_parameters(val=False, name='mu')
    cnn.networks[1].set_parameters(val=False, name='sigma')

    ## Network 2: drift: not fit
    cnn.networks[2].layers[0].weight.data = torch.tensor(drift_terms, dtype=torch.float32)
    cnn.networks[2].set_parameters(val=False)

    ## Network 3: Comb
    cnn.networks[-1].set_parameters(val=False, name='weight')

    cnn = cnn.to(device)


    # fit the model
    cnn.fit(data, **adam_parsT, verbose=2)
    LLs = cnn.eval_models(data, data_inds=data.val_blks, batch_size=5)

    null_adjusted_LLs = expts.LLsNULL-LLs

    print(np.mean(null_adjusted_LLs))

    # save the model
    filename = folder_name+'/model_'+str(cproj)+'_'+str(wproj)+'_'+str(fproj)+'_'+str(fconv0)+'_'+str(fconv1)
    cnn.save_model(filename+'.pkl')
    # save the LLs
    with open(filename+'.npy', 'wb') as f:
        np.save(f, null_adjusted_LLs)

