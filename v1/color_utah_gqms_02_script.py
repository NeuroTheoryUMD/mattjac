#!/usr/bin/env python
# coding: utf-8

# # imports

import os

folder_name = 'models/cnns_03'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# In[1]:


import sys
sys.path.append('./lib')

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
from NDNT.modules.layers import NDNLayer, ConvLayer, STconvLayer, Tlayer, ChannelLayer, IterSTlayer, ReadoutLayer
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


fn = 'Jocamo_220715_full_CC_ETCC_nofix_v08'
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

data.assemble_stimulus(top_corner=[new_tc[0], new_tc[1]], L=NX, fixdot=0, shifts=-shifts)


# In[6]:


goodfix = np.where(ETmetrics[:,1] < 0.80)[0]
valfix = torch.zeros([ETmetrics.shape[0], 1], dtype=torch.float32)
valfix[goodfix] = 1.0
# Test base-level performance (full DFs and then modify DFs)
#DFsave = deepcopy(data2.dfs)  # this is also in data.dfs
data.dfs_out *= valfix
("%0.1f%% fixations remaining"%(100*len(goodfix)/ETmetrics.shape[0]))
dirname2 = dirname+'0715/NewGLMs/'
matdat = sio.loadmat(dirname2+'J0715ProcGLMinfo.mat')


fw0 = 7
Treg = 0.01
Xreg = 0.000001
Mreg = 0.0001
Creg = None
Dreg = 0.5
Gnet = 0.005
num_iter = 4


# define model components
#set up fits
Treg = 1
Xreg = 20 # [20]
L1reg = 0.1 # [0.5]
GLreg = 10.0 # [4.0]

# drift network
drift_pars1 = NDNLayer.layer_dict(
    input_dims=[1,1,1,NA], num_filters=1, bias=False, norm_type=0, NLtype='lin')
drift_pars1['reg_vals'] = {'d2t': Dreg, 'bcs':{'d2t':0} }
# for stand-alone drift model
drift_pars1N = deepcopy(drift_pars1)
drift_pars1N['NLtype'] = 'softplus'
drift_net =  FFnetwork.ffnet_dict( xstim_n = 'Xdrift', layer_list = [drift_pars1] )

# glm net
glm_layer = Tlayer.layer_dict(
    input_dims=data.stim_dims, num_filters=1, bias=False, num_lags=num_lags,
    NLtype='lin', initialize_center = True)
glm_layer['reg_vals'] = {'d2x': Xreg, 'd2t': Treg, 'l1': L1reg, 'glocalx': GLreg,'edge_t':10}
stim_net =  FFnetwork.ffnet_dict( xstim_n = 'stim', layer_list = [glm_layer] )

# gqm net
num_subs = 2
gqm_layer = Tlayer.layer_dict(
    input_dims=data.stim_dims, num_filters=num_subs, num_inh=0, bias=False, num_lags=num_lags,
    NLtype='square', initialize_center = True)
gqm_layer['reg_vals'] = {'d2x': Xreg, 'd2t': Treg, 'l1': L1reg, 'glocalx': GLreg,'edge_t':10}
stim_qnet =  FFnetwork.ffnet_dict( xstim_n = 'stim', layer_list = [gqm_layer] )

#combine glm
comb_layer = NDNLayer.layer_dict(
    num_filters = 1, NLtype='softplus', bias=False)
comb_layer['weights_initializer'] = 'ones'

net_comb = FFnetwork.ffnet_dict(
    xstim_n = None, ffnet_n=[0,1],
    layer_list = [comb_layer], ffnet_type='add')

#combine gqm
comb2_layer = ChannelLayer.layer_dict(
    num_filters = 1, NLtype='softplus', bias=False)
comb2_layer['weights_initializer'] = 'ones'

net2_comb = FFnetwork.ffnet_dict(
    xstim_n = None, ffnet_n=[0,1,2],
    layer_list = [comb2_layer], ffnet_type='normal')
net2_comb['layer_list'][0]['bias'] = True


# In[7]:
import optuna

drifts = []

for cc in range(NCv):
    drift_model_filename = 'models/glms_03/drift_model_cc'+str(valET[cc])+'.pkl'
    drift_study_filename = 'models/glms_03/drift_study_cc'+str(valET[cc])+'.pkl'

    # continue if the file already exists
    if os.path.isfile(drift_model_filename):
        # load the model and continue
        print('loading model', cc)
        with open(drift_model_filename, 'rb') as f:
            drifts.append(pickle.load(f))



# In[8]:
glms = []
for i, cc in enumerate(range(NCv)):
    glm_model_filename = 'models/glms_03/glm_model_cc'+str(valET[cc])+'.pkl'
    glm_study_filename = 'models/glms_03/glm_study_cc'+str(valET[cc])+'.pkl'

    drift_weights = drifts[i].ndn_model.networks[0].layers[0].weight.data[:,0]

    # continue if the file already exists
    if os.path.isfile(glm_model_filename):
        # load the model and continue
        print('loading model', cc)
        with open(glm_model_filename, 'rb') as f:
            glms.append(pickle.load(f))
        continue

    data.set_cells(valET[cc])

    LLsNULL = drifts[i].LLs

    glm = NDN.NDN(ffnet_list = [stim_net, drift_net, net_comb], loss_type='poisson')
    glm.block_sample=True
    glm.networks[1].layers[0].weight.data[:,0] = deepcopy(drift_weights)
    glm.networks[1].layers[0].set_parameters(val=False)
    glm.networks[2].layers[0].set_parameters(val=False,name='weight')

    glms_temp = []

    def objective(trial):
        lbfgs_pars = utils.create_optimizer_params(
            optimizer_type='lbfgs',
            tolerance_change=trial.suggest_float('tolerance_change', 1e-10, 1e-6),
            tolerance_grad=trial.suggest_float('tolerance_grad', 1e-10, 1e-6),
            history_size=100,
            batch_size=20,
            max_epochs=3,
            max_iter=500,
            device=device)

        glm.networks[0].layers[0].reg.vals['d2x'] = trial.suggest_float('d2x', 10, 30)
        glm.networks[0].layers[0].reg.vals['d2t'] = trial.suggest_float('d2t', 0.5, 2)
        glm.networks[0].layers[0].reg.vals['d2xt'] = trial.suggest_float('d2xt', 0.001, 10000)
        glm.networks[0].layers[0].reg.vals['l1'] = trial.suggest_float('l1', 0.01, 1)
        glm.networks[0].layers[0].reg.vals['glocalx'] = trial.suggest_float('glocalx', 0.001, 20)

        glm.fit(data, force_dict_training=True, trial=trial, **lbfgs_pars)
        LL = glm.eval_models(data[data.val_blks], null_adjusted=False)[0]

        null_adjusted_LL = LLsNULL - LL

        glm_model = Model(glm, null_adjusted_LL, trial)
        glms_temp.append(glm_model)

        return null_adjusted_LL

    study = optuna.create_study(direction='maximize')

    # enqueue initial parameters
    study.enqueue_trial(
        {'d2t': 1,
         'd2x': 20,
         'd2xt': 0.001,
         'l1': 0.1,
         'glocalx': 10.0,
         'tolerance_change': 1e-8,
         'tolerance_grad': 1e-8})
    study.enqueue_trial(
        {'d2t': 1,
         'd2x': 20,
         'd2xt': 0.01,
         'l1': 0.1,
         'glocalx': 10.0,
         'tolerance_change': 1e-8,
         'tolerance_grad': 1e-8})
    study.enqueue_trial(
        {'d2t': 1,
         'd2x': 20,
         'd2xt': 0.1,
         'l1': 0.1,
         'glocalx': 10.0,
         'tolerance_change': 1e-8,
         'tolerance_grad': 1e-8})
    study.enqueue_trial(
        {'d2t': 1,
         'd2x': 20,
         'd2xt': 1,
         'l1': 0.1,
         'glocalx': 10.0,
         'tolerance_change': 1e-8,
         'tolerance_grad': 1e-8})
    study.enqueue_trial(
        {'d2t': 1,
         'd2x': 20,
         'd2xt': 10,
         'l1': 0.1,
         'glocalx': 10.0,
         'tolerance_change': 1e-8,
         'tolerance_grad': 1e-8})

    study.optimize(objective, n_trials=50)

    best_model = glms_temp[study.best_trial.number]

    glms.append(best_model)

    # save the model
    with open(glm_model_filename, 'wb') as f:
        pickle.dump(best_model, f)

    # save the study
    with open(glm_study_filename, 'wb') as f:
        pickle.dump(study, f)

    print(study.best_trial.number, study.best_params)


# In[9]:
gqms = []
for i, cc in enumerate(range(NCv)):
    gqm_model_filename = 'models/glms_03/gqm_model_cc'+str(valET[cc])+'.pkl'
    gqm_study_filename = 'models/glms_03/gqm_study_cc'+str(valET[cc])+'.pkl'

    drift_weights = drifts[i].ndn_model.networks[0].layers[0].weight.data[:,0]

    # continue if the file already exists
    if os.path.isfile(gqm_model_filename):
        # load the model and continue
        print('loading model', cc)
        with open(gqm_model_filename, 'rb') as f:
            gqms.append(pickle.load(f))
        continue

    data.set_cells(valET[cc])

    LLsNULL = drifts[i].LLs

    # get the best reg_vals for the GLM
    best_reg_vals = glms[i].ndn_model.networks[0].layers[0].reg.vals
    stim_net['layer_list'][0]['reg_vals'] = deepcopy(best_reg_vals)
    gqm = NDN.NDN(ffnet_list = [stim_net, drift_net, stim_qnet, net2_comb], loss_type='poisson')
    gqm.networks[0].layers[0] = deepcopy(glms[i].ndn_model.networks[0].layers[0])
    gqm.block_sample=True
    gqm.networks[3].layers[0].set_parameters(val=False,name='weight')
    gqm.networks[1].layers[0].weight.data[:,0] = deepcopy(
        drifts[i].ndn_model.networks[0].layers[0].weight.data[:,0])
    gqm.networks[1].layers[0].set_parameters(val=False)

    gqms_temp = []

    def objective(trial):
        lbfgs_pars = utils.create_optimizer_params(
            optimizer_type='lbfgs',
            tolerance_change=trial.suggest_float('tolerance_change', 1e-10, 1e-6),
            tolerance_grad=trial.suggest_float('tolerance_grad', 1e-10, 1e-6),
            history_size=100,
            batch_size=20,
            max_epochs=3,
            max_iter = 500,
            device = device)

        # linear reg vals
        gqm.networks[0].layers[0].reg.vals['d2x'] = trial.suggest_float('lin_d2x', 10, 30)
        gqm.networks[0].layers[0].reg.vals['d2t'] = trial.suggest_float('lin_d2t', 0.5, 2)
        gqm.networks[0].layers[0].reg.vals['d2xt'] = trial.suggest_float('lin_d2xt', 0.001, 10000)
        gqm.networks[0].layers[0].reg.vals['l1'] = trial.suggest_float('lin_l1', 0.01, 1)
        gqm.networks[0].layers[0].reg.vals['glocalx'] = trial.suggest_float('lin_glocalx', 0.001, 20)

        # quadratic reg vals
        gqm.networks[2].layers[0].reg.vals['d2x'] = trial.suggest_float('quad_d2x', 10, 30)
        gqm.networks[2].layers[0].reg.vals['d2t'] = trial.suggest_float('quad_d2t', 0.5, 2)
        gqm.networks[2].layers[0].reg.vals['d2xt'] = trial.suggest_float('quad_d2xt', 0.001, 10000)
        gqm.networks[2].layers[0].reg.vals['l1'] = trial.suggest_float('quad_la', 0.01, 1)
        gqm.networks[2].layers[0].reg.vals['glocalx'] = trial.suggest_float('quad_glocalx', 0.001, 20)

        gqm.fit( data, force_dict_training=True, **lbfgs_pars)
        LL = gqm.eval_models(data[data.val_blks], null_adjusted=False)[0]

        null_adjusted_LL = LLsNULL - LL

        gqm_model = Model(gqm, null_adjusted_LL, trial)
        gqms_temp.append(gqm_model)

        return null_adjusted_LL

    study = optuna.create_study(direction='maximize')

    # enqueue initial parameters
    study.enqueue_trial(
        {'lin_d2t': best_reg_vals['d2t'],
         'lin_d2x': best_reg_vals['d2x'],
         'lin_d2xt': best_reg_vals['d2xt'],
         'lin_l1': best_reg_vals['l1'],
         'lin_glocalx': best_reg_vals['glocalx'],
         'quad_d2t': 1,
         'quad_d2x': 20,
         'quad_d2xt': 0.001,
         'quad_l1': 0.1,
         'quad_glocalx': 10.0,
         'tolerance_change': 1e-8,
         'tolerance_grad': 1e-8})
    study.enqueue_trial(
        {'lin_d2t': best_reg_vals['d2t'],
         'lin_d2x': best_reg_vals['d2x'],
         'lin_d2xt': best_reg_vals['d2xt'],
         'lin_l1': best_reg_vals['l1'],
         'lin_glocalx': best_reg_vals['glocalx'],
         'quad_d2t': 1,
         'quad_d2x': 20,
         'quad_d2xt': 0.01,
         'quad_l1': 0.1,
         'quad_glocalx': 10.0,
         'tolerance_change': 1e-8,
         'tolerance_grad': 1e-8})
    study.enqueue_trial(
        {'lin_d2t': best_reg_vals['d2t'],
         'lin_d2x': best_reg_vals['d2x'],
         'lin_d2xt': best_reg_vals['d2xt'],
         'lin_l1': best_reg_vals['l1'],
         'lin_glocalx': best_reg_vals['glocalx'],
         'quad_d2t': 1,
         'quad_d2x': 20,
         'quad_d2xt': 0.1,
         'quad_l1': 0.1,
         'quad_glocalx': 10.0,
         'tolerance_change': 1e-8,
         'tolerance_grad': 1e-8})
    study.enqueue_trial(
        {'lin_d2t': best_reg_vals['d2t'],
         'lin_d2x': best_reg_vals['d2x'],
         'lin_d2xt': best_reg_vals['d2xt'],
         'lin_l1': best_reg_vals['l1'],
         'lin_glocalx': best_reg_vals['glocalx'],
         'quad_d2t': 1,
         'quad_d2x': 20,
         'quad_d2xt': 1,
         'quad_l1': 0.1,
         'quad_glocalx': 10.0,
         'tolerance_change': 1e-8,
         'tolerance_grad': 1e-8})
    study.enqueue_trial(
        {'lin_d2t': best_reg_vals['d2t'],
         'lin_d2x': best_reg_vals['d2x'],
         'lin_d2xt': best_reg_vals['d2xt'],
         'lin_l1': best_reg_vals['l1'],
         'lin_glocalx': best_reg_vals['glocalx'],
         'quad_d2t': 1,
         'quad_d2x': 20,
         'quad_d2xt': 10,
         'quad_l1': 0.1,
         'quad_glocalx': 10.0,
         'tolerance_change': 1e-8,
         'tolerance_grad': 1e-8})

    study.optimize(objective, n_trials=50)

    best_model = gqms_temp[study.best_trial.number]

    gqms.append(best_model)

    with open(gqm_model_filename, 'wb') as f:
        pickle.dump(best_model, f)

    with open(gqm_study_filename, 'wb') as f:
        pickle.dump(study, f)

    print(study.best_trial.number, study.best_params)