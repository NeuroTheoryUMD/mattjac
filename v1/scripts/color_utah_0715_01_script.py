#!/usr/bin/env python
# coding: utf-8

# # imports

import os

folder_name = 'cnns_multi_0715_00'
num_trials = 1

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# In[1]:


import sys
sys.path.append('./lib')

import h5py
import h5py

# setup paths
iteration = 1 # which version of this tutorial to run (in case want results in different dirs)
NBname = 'color_cloud_initial{}'.format(iteration)

myhost = os.uname()[1] # get name of machine
print("Running on Computer: [%s]" %myhost)

datadir = '/home/dbutts/ColorV1/Data/'
dirname = '/home/dbutts/ColorV1/CLRworkspace/' # Working directory 

import numpy as np
import torch
import pickle
import os
import optuna
from time import time
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.io as sio
from copy import deepcopy

# plotting
import matplotlib.pyplot as plt

# Import torch
import torch
from torch import nn

# NDN tools
import NDNT.utils as utils # some other utilities
import NDNT.NDNT as NDN
from NDNT.modules.layers import NDNLayer, ConvLayer, STconvLayer, Tlayer, ChannelLayer, IterLayer, IterSTlayer, ReadoutLayer
from NDNT.networks import FFnetwork
from time import time
import dill

from NTdatasets.generic import GenericDataset
import NTdatasets.conway.cloud_datasets as datasets

# Utilities
import ColorDataUtils.ConwayUtils as CU
import ColorDataUtils.EyeTrackingUtils as ETutils
from NDNT.utils import imagesc   # because I'm lazy
from NDNT.utils import ss        # because I'm real lazy

device = torch.device("cuda:1")
dtype = torch.float32

# Where saved models and checkpoints go -- this is to be automated
print( 'Save_dir =', dirname)
print(device)

class Model:
    def __init__(self, ndn_model, LLs, trial):
        self.ndn_model = ndn_model
        self.LLs = LLs
        self.trial = trial


# # load data (all stim)

# In[2]:


fn = 'Jocamo_220715_full_CC_ETCC_nofix_v08_packaged'
num_lags=16

t0 = time()
data = datasets.ColorClouds(
    datadir=datadir, filenames=[fn], eye_config=3, drift_interval=16,
    luminance_only=True, binocular=False, include_MUs=True, num_lags=num_lags,
    trial_sample=True)
t1 = time()
print(t1-t0, 'sec elapsed')

NT = data.robs.shape[0]
NA = data.Xdrift.shape[1]
print("%d (%d valid) time points"%(NT, len(data)))
#data.valid_inds = np.arange(NT, dtype=np.int64)

lam_units = np.where(data.channel_map < 32)[0]
ETunits = np.where(data.channel_map >= 32)[0]
UTunits = np.where(data.channel_map >= 32+127)[0]

print( "%d laminar units, %d ET units"%(len(lam_units), len(ETunits)))

# Replace DFs
matdat = sio.loadmat(datadir+'Jocamo_220715_full_CC_ETCC_nofix_v08_DFextra.mat')
data.dfs = torch.tensor( matdat['XDF'][:NT, :], dtype=torch.float32 )

# Pull correct saccades
matdat = sio.loadmat( datadir+'Jocamo_220715_full_CC_ETCC_v08_ETupdate.mat')
sac_ts_all = matdat['ALLsac_bins'][0, :]

data.process_fixations( sac_ts_all )
sac_tsB = matdat['sac_binsB'][0, :]
sac_tsL = matdat['sac_binsL'][0, :]
sac_tsR = matdat['sac_binsR'][0, :]

NFIX = torch.max(data.fix_n).detach().numpy()
print(NFIX, 'fixations')
et1kHzB = matdat['et1kHzB']
et60B = matdat['et60HzB']
et60all = matdat['et60Hz_all']


# In[3]:


# Set cells-to-analyze and pull best model configuration and mus
Reff = torch.mul(data.robs[:, UTunits], data.dfs[:, UTunits]).numpy()
nspks = np.sum(Reff, axis=0)
a = np.where(nspks > 10)[0]
valET = UTunits[a]
NCv = len(valET)
print("%d out of %d units used"%(len(valET), len(UTunits)))

## CONVERT LLsNULL, which is based on 

# Read in previous data
dirname2 = dirname+'0715/et/'
matdat = sio.loadmat(dirname2+'LLsGLM.mat')
Dreg = matdat['Dreg']
top_corner = matdat['top_corner'][:, 0]

data.set_cells(valET)


# In[4]:


# Load shifts and previous models
dirname2 = dirname+'0715/et/'
SHfile = sio.loadmat( dirname2 + 'BDshifts1.mat' )
fix_n = SHfile['fix_n']
shifts = SHfile['shifts']
metricsLL = SHfile['metricsLL']
metricsTH = SHfile['metricsTH']
ETshifts = SHfile['ETshifts']
ETmetrics = SHfile['ETmetrics']
Ukeeps = SHfile['Ctrain']
XVkeeps = SHfile['Cval']

# Make 60x60 STAs (and GLMs)
Xshift = 14 #8+4 
Yshift = -3 #-10+4
NX = 60

new_tc = np.array([top_corner[0]-Xshift, top_corner[1]-Yshift], dtype=np.int64)
data.draw_stim_locations(top_corner = new_tc, L=NX)

data.assemble_stimulus(top_corner=[new_tc[0], new_tc[1]], L=NX, fixdot=0, shifts=-shifts, num_lags=num_lags)


# In[6]:


goodfix = np.where(ETmetrics[:,1] < 0.80)[0]
valfix = torch.zeros([ETmetrics.shape[0], 1], dtype=torch.float32)
valfix[goodfix] = 1.0
# Test base-level performance (full DFs and then modify DFs)
#DFsave = deepcopy(data2.dfs)  # this is also in data.dfs
data.dfs_out *= valfix
print("%0.1f%% fixations remaining"%(100*len(goodfix)/ETmetrics.shape[0]))

dirname2 = dirname+'0715/NewGLMs/'
matdat = sio.loadmat(dirname2+'J0715ProcGLMinfo.mat')
LLsNULL = matdat['LLsNULL'][:,0]
LLsGLM = matdat['LLsGLM'][:,0]
LLsGLM2 = matdat['LLsGLM2'][:,0]
drift_terms = matdat['drift_terms']
valET = matdat['cells']
RFcenters = matdat['RFcenters']
#'Gregs': Gopt[:,None], 'XTregs': Xopt
#'top_corner': new_tc[:, None]})
mu0s = utils.pixel2grid(deepcopy(RFcenters[:, [1, 0]]), L=NX)


# In[ ]:

adam_parsT = utils.create_optimizer_params(
    optimizer_type='AdamW',
    batch_size=2, # * 240 timesteps
    num_workers=0,
    learning_rate=0.0017,
    early_stopping_patience=4,
    optimize_graph=False,
    weight_decay=0.235)
adam_parsT['device'] = device
# how many batches to wait for before calculating the gradient
adam_parsT['accumulated_grad_batches'] = 6


# some good starting parameters
Treg = 0.01
Xreg = 0.000001
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
                     9,  # width
                     9,  # height
                     14], # lags
        NLtype='relu',
        initialize_center=True)
    LGNpars['output_norm']='batch'
    LGNpars['window']='hamming'
    LGNpars['reg_vals'] = {'d2x':Xreg,
                           'd2t':Treg,
                           'center': Creg,
                           'edge_t':100} # just pushes the edge to be sharper

    # num_subs = trial.suggest_int('num_subs', 10, 50)
    # num_inh = trial.suggest_float('num_inh', 0.1, 0.7)
    num_subs0 = trial.suggest_int('num_subs_l0', 20, 80)
    num_subs1 = trial.suggest_int('num_subs_l1', 20, 80)
    num_subs2 = trial.suggest_int('num_subs_l2', 20, 80)
    num_inh0 = trial.suggest_float('num_inh_l0', 0.1, 0.7)
    num_inh1 = trial.suggest_float('num_inh_l1', 0.1, 0.7)
    num_inh2 = trial.suggest_float('num_inh_l2', 0.1, 0.7)
    conv_l0_filter_width = trial.suggest_int('conv_l0_filter_width', 7, 39, step=2)
    conv_l1_filter_width = trial.suggest_int('conv_l1_filter_width', 5, 39, step=2)
    conv_l2_filter_width = trial.suggest_int('conv_l2_filter_width', 3, 39, step=2)
    proj_pars = ConvLayer.layer_dict(
        num_filters=num_subs0,
        bias=False,
        norm_type=1,
        num_inh=int(num_inh0*num_subs0),
        filter_dims=trial.suggest_int('proj_filter_width', 9, 29, step=2),
        NLtype=trial.suggest_categorical('proj_NLtype', ['lin', 'relu']),
        initialize_center=True)
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

    conv_layer0 = ConvLayer.layer_dict(
        num_filters=num_subs1,
        num_inh=int(num_inh1*num_subs1),
        bias=False,
        pos_constraint=True,
        norm_type=1,
        filter_dims=conv_l0_filter_width,
        NLtype='relu',
        initialize_center=False)
    conv_layer0['output_norm'] = 'batch'

    conv_layer1 = ConvLayer.layer_dict(
        num_filters=num_subs1,
        num_inh=int(num_inh1*num_subs1),
        bias=False,
        pos_constraint=True,
        norm_type=1,
        filter_dims=conv_l1_filter_width,
        NLtype='relu',
        initialize_center=False)
    conv_layer1['output_norm'] = 'batch'

    conv_layer2 = ConvLayer.layer_dict(
        num_filters=num_subs2,
        num_inh=int(num_inh2*num_subs2),
        bias=False,
        pos_constraint=True,
        norm_type=1,
        filter_dims=conv_l2_filter_width,
        NLtype='relu',
        initialize_center=False)
    conv_layer2['output_norm'] = 'batch'

    # iter_layer = IterLayer.layer_dict(
    #     num_filters=num_subs,
    #     num_inh=int(num_inh*num_subs),
    #     bias=False,
    #     num_iter=trial.suggest_int('num_iter', 2, 8),
    #     output_config='full',
    #     pos_constraint=True,
    #     norm_type=1,
    #     filter_width=trial.suggest_int('iter_filter_width', 7, 47, step=2),
    #     #num_lags=2,
    #     NLtype='relu',
    #     initialize_center=False)
    # iter_layer['output_norm'] = 'batch'

    scaffold_net =  FFnetwork.ffnet_dict(
        ffnet_type='scaffold',
        xstim_n='stim',
        layer_list=[LGNpars, proj_pars, conv_layer0, conv_layer1, conv_layer2],
        scaffold_levels=[1,2,3,4])

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

    null_adjusted_LLs = LLsNULL-LLs

    cnn_model = Model(cnn, null_adjusted_LLs, trial)

    with open(folder_name+'/cnn_'+str(trial.number)+'.pkl', 'wb') as f:
        pickle.dump(cnn_model, f)

    # dump the intermediate study, but this will be off by one trial
    with open(folder_name+'/study.pkl', 'wb') as f:
        pickle.dump(study, f)

    return np.mean(null_adjusted_LLs)



study = optuna.create_study(direction='maximize')

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

proj_NLtypes = ['lin', 'relu']
for proj_NLtype in proj_NLtypes:
    study.enqueue_trial(
        {'proj_NLtype': proj_NLtype,
         'proj_filter_width': 17,
         'num_subs_l0': 48,
         'num_subs_l1': 48,
         'num_subs_l2': 48,
         'num_inh_l0': 0.5,
         'num_inh_l1': 0.5,
         'num_inh_l2': 0.5,
         'conv_l0_filter_width': 15,
         'conv_l1_filter_width': 9,
         'conv_l2_filter_width': 5})


study.optimize(objective, n_trials=num_trials)

# dump the final study
with open(folder_name+'/study.pkl', 'wb') as f:
    pickle.dump(study, f)

print(study.best_trial.number, study.best_params)
