#!/usr/bin/env python
# coding: utf-8

folder_name = 'models/cnns_multi_05'
num_trials = 10
testing = False
##############################################


import sys
sys.path.append('./lib')

from copy import deepcopy

import sys
import os
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import optuna
import torch
import pickle
import dill
from torch import nn

# NDN tools
import NDNT.utils as utils # some other utilities
import NDNT.NDNT as NDN
from NDNT.modules.layers import *
from NDNT.networks import *
from time import time

import ColorDataUtils.ConwayUtils as CU
import ColorDataUtils.EyeTrackingUtils as ETutils
from ColorDataUtils.multidata_utils import MultiExperiment
from NDNT.utils import imagesc   # because I'm lazy
from NDNT.utils import ss        # because I'm real lazy

device = torch.device("cuda:1")
dtype = torch.float32

datadir = '/home/dbutts/ColorV1/Data/'
dirname = '/home/dbutts/ColorV1/CLRworkspace/'

class Model:
    def __init__(self, ndn_model, LLs, trial):
        self.ndn_model = ndn_model
        self.LLs = LLs
        self.trial = trial


# load data
num_lags=16
expt_names = ['J220715','J220722','J220801','J220808']
#expt_names = ['J220715']
expts = MultiExperiment(expt_names)
data, drift_terms, mu0s = expts.load(datadir,
                                     num_lags=num_lags,
                                     et_metric_thresh=0.8,
                                     array_types=['UT'],
                                     luminance_only=True)


# fit params
# adam_parsT = utils.create_optimizer_params(
#     optimizer_type='AdamW', batch_size=4, num_workers=0,
#     learning_rate=0.01, early_stopping_patience=10,  # changed from 4
#     optimize_graph=False, weight_decay = 0.2)
adam_parsT = utils.create_optimizer_params(
    optimizer_type='AdamW',
    batch_size=2, # * 240 timesteps
    num_workers=0,
    learning_rate=0.0017,
    early_stopping_patience=4,
    optimize_graph=False,
    weight_decay=0.235)
adam_parsT['device'] = device
adam_parsT['accumulated_grad_batches'] = 6
if testing:
    adam_parsT['max_epochs'] = 1


# setup
data.device = device
NCv = data.NC
NT = data.robs.shape[0]
NA = data.Xdrift.shape[1]

# with open('models/cnns_03/cnn_12.pkl', 'rb') as f:
#     cnn12 = pickle.load(f).ndn_model
# print(cnn12.networks[0].layers[0].get_weights().shape)


# some good starting parameters
Treg = 0.001
Xreg = None #0.000001
Mreg = 0.0001
Creg = None
Dreg = 0.5

# define model
def objective(trial):
    LGNpars = STconvLayer.layer_dict(
        input_dims = data.stim_dims,
        num_filters=4,
        num_inh=2,
        bias=False,
        norm_type=1,
        filter_dims=[1,  # channels
                     7,  # width
                     7,  # height
                     14], # lags
        NLtype='relu',
        initialize_center=True)
    LGNpars['output_norm']='batch'
    LGNpars['window']='hamming'
    LGNpars['reg_vals'] = {'d2x':Xreg,
                           'd2t':Treg,
                           'center': Creg, # None
                           'edge_t':100} # just pushes the edge to be sharper

    num_subs = trial.suggest_int('num_subs', 10, 50)
    num_inh = trial.suggest_float('num_inh', 0.1, 0.7)
    # num_subs0 = trial.suggest_int('num_subs_l0', 20, 80)
    # num_subs1 = trial.suggest_int('num_subs_l1', 20, 80)
    # num_subs2 = trial.suggest_int('num_subs_l2', 20, 80)
    # num_inh0 = trial.suggest_float('num_inh_l0', 0.1, 0.7)
    # num_inh1 = trial.suggest_float('num_inh_l1', 0.1, 0.7)
    # num_inh2 = trial.suggest_float('num_inh_l2', 0.1, 0.7)
    # conv_l0_filter_width = trial.suggest_int('conv_l0_filter_width', 7, 39, step=2)
    # conv_l1_filter_width = trial.suggest_int('conv_l1_filter_width', 5, 39, step=2)
    # conv_l2_filter_width = trial.suggest_int('conv_l2_filter_width', 3, 39, step=2)
    proj_pars = ConvLayer.layer_dict(
        num_filters=num_subs,
        bias=False,
        norm_type=1,
        num_inh=int(num_inh*num_subs),
        filter_dims=trial.suggest_int('proj_filter_width', 7, 29, step=2),
        NLtype=trial.suggest_categorical('proj_NLtype', ['lin', 'relu']),
        initialize_center=True,
        pos_constraint=True)
    proj_pars['output_norm']='batch'
    proj_pars['window']='hamming'

    # conv_layer0 = STconvLayer.layer_dict(
    #     num_filters=num_subs1,
    #     num_inh=int(num_inh1*num_subs1),
    #     bias=False,
    #     pos_constraint=True,
    #     norm_type=1,
    #     conv_dims=[conv_l0_filter_width, conv_l0_filter_width, 2],
    #     NLtype='relu',
    #     initialize_center=False)
    # conv_layer0['output_norm'] = 'batch'

    # conv_layer0 = ConvLayer.layer_dict(
    #     num_filters=num_subs1,
    #     num_inh=int(num_inh1*num_subs1),
    #     bias=False,
    #     pos_constraint=True,
    #     norm_type=1,
    #     filter_dims=conv_l0_filter_width,
    #     NLtype='relu',
    #     initialize_center=False)
    # conv_layer0['output_norm'] = 'batch'
    # 
    # conv_layer1 = ConvLayer.layer_dict(
    #     num_filters=num_subs1,
    #     num_inh=int(num_inh1*num_subs1),
    #     bias=False,
    #     pos_constraint=True,
    #     norm_type=1,
    #     filter_dims=conv_l1_filter_width,
    #     NLtype='relu',
    #     initialize_center=False)
    # conv_layer1['output_norm'] = 'batch'
    # 
    # conv_layer2 = ConvLayer.layer_dict(
    #     num_filters=num_subs2,
    #     num_inh=int(num_inh2*num_subs2),
    #     bias=False,
    #     pos_constraint=True,
    #     norm_type=1,
    #     filter_dims=conv_l2_filter_width,
    #     NLtype='relu',
    #     initialize_center=False)
    # conv_layer2['output_norm'] = 'batch'

    iter_layer = IterLayer.layer_dict(
        num_filters=num_subs,
        num_inh=int(num_inh*num_subs),
        bias=False,
        num_iter=trial.suggest_int('num_iter', 2, 8),
        output_config='full',
        pos_constraint=True,
        norm_type=1,
        filter_width=trial.suggest_int('iter_filter_width', 7, 47, step=2),
        num_lags=2,
        NLtype='relu',
        initialize_center=False,
        res_layer=False)
    iter_layer['output_norm'] = 'batch'

    scaffold_net =  FFnetwork.ffnet_dict(
        ffnet_type='scaffold',
        xstim_n='stim',
        layer_list=[LGNpars, proj_pars, iter_layer],
        scaffold_levels=[1,2])

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


    ## Network 0: LGN
    # copy weights from a good LGN model
    #cnn.networks[0].layers[0] = deepcopy(cnn12.networks[0].layers[0]) # copy weights

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

    cnn.fit(data, **adam_parsT, verbose=2)
    LLs = cnn.eval_models(data, data_inds=data.val_blks, batch_size=5)

    null_adjusted_LLs = expts.LLsNULL-LLs

    cnn_model = Model(cnn, null_adjusted_LLs, trial)

    with open(folder_name+'/cnn_'+str(trial.number)+'.pkl', 'wb') as f:
        pickle.dump(cnn_model, f)

    # dump the intermediate study, but this will be off by one trial
    with open(folder_name+'/study.pkl', 'wb') as f:
        pickle.dump(study, f)

    return np.mean(null_adjusted_LLs)


#study = optuna.load_study(study_name='

#study = optuna.create_study(direction='maximize',
#                            study_name='Four datasets w/o initializing LGN weights')

with open(folder_name+'/study.pkl', 'rb') as f:
    study = pickle.load(f)

# # enqueue initial parameters
# num_iters = [3, 5, 7]
# proj_NLtypes = ['lin', 'relu']
# for proj_NLtype in proj_NLtypes:
#     for num_iter in num_iters:
#         study.enqueue_trial(
#             {'proj_NLtype': proj_NLtype, # best = lin
#              'num_subs': 38, # best = 38
#              'num_inh': 0.4, # best = 0.4
#              'num_iter': num_iter, # best = 5
#              'proj_filter_width': 21, # best = 21
#              'iter_filter_width': 9}) # best = 9

#proj_NLtypes = ['lin', 'relu']
#for proj_NLtype in proj_NLtypes:
#    study.enqueue_trial(
        # {'proj_NLtype': proj_NLtype,
        #  'proj_filter_width': 17,
        #  'num_subs_l0': 48,
        #  'num_subs_l1': 48,
        #  'num_subs_l2': 48,
        #  'num_inh_l0': 0.5,
        #  'num_inh_l1': 0.5,
        #  'num_inh_l2': 0.5,
        #  'conv_l0_filter_width': 15,
        #  'conv_l1_filter_width': 9,
        #  'conv_l2_filter_width': 5})

# FOR CNN
# study.enqueue_trial({
#     'conv_l0_filter_width': 7,
#     'conv_l1_filter_width': 13,
#     'conv_l2_filter_width': 15,
#     'num_inh_l0': 0.248047,
#     'num_inh_l1': 0.184399,
#     'num_inh_l2': 0.448961,
#     'num_subs_l0': 47,
#     'num_subs_l1': 47,
#     'num_subs_l2': 31,
#     'proj_NLtype': 'relu',
#     'proj_filter_width': 13,
# })

# FOR ITER
# study.enqueue_trial({
#     'iter_filter_width': 9,
#     'num_inh': 0.4,
#     'num_iter': 5,
#     'num_subs': 38,
#     'proj_NLtype': 'relu',
#     'proj_filter_width': 21,
# })


study.optimize(objective, n_trials=num_trials)

# dump the final study
with open(folder_name+'/study.pkl', 'wb') as f:
    pickle.dump(study, f)

print(study.best_trial.number, study.best_params)
