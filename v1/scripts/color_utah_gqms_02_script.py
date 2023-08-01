#!/usr/bin/env python
# coding: utf-8

# # imports

import os

folder_name = 'glms_04_debug4'

# Set up parameters
d2x_vals = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
glocal_vals = [0.001, 0.1, 1, 10, 100, 1000] # glocal
l1_vals = [0.001, 0.01, 0.1, 1] # L1

val_lists_map = {'d2x': d2x_vals, 'd2t': d2x_vals, 'glocalx': glocal_vals, 'l1': l1_vals}

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

from ColorDataUtils.optimize import Optimizer


device = torch.device("cuda:0")
dtype = torch.float32

# Where saved models and checkpoints go -- this is to be automated
print( 'Save_dir =', dirname)
print(device)

class Model:
    def __init__(self, ndn, LLs):
        self.ndn = ndn
        self.LLs = LLs

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

data.assemble_stimulus(top_corner=[new_tc[0], new_tc[1]], L=NX, fixdot=0, shifts=-shifts, num_lags=num_lags)


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


lbfgs_pars = utils.create_optimizer_params(
    optimizer_type='lbfgs',
    tolerance_change=1e-8,
    tolerance_grad=1e-8,
    history_size=100,
    batch_size=20,
    max_epochs=3,
    max_iter=500,
    device=device)


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
def fit_drift(i, cc, folder):
    drift_model_filename = 'models/'+folder+'/drift_model_cc'+str(cc)+'.pkl'

    # continue if the file already exists
    if os.path.isfile(drift_model_filename):
        # load the model and continue
        print('loading model', cc)
        with open(drift_model_filename, 'rb') as f:
            drift = pickle.load(f)
        return drift

    data.set_cells(valET[cc])
    
    drift_ndn = NDN.NDN(
        layer_list = [drift_pars1N], loss_type='poisson')
    drift_ndn.block_sample=True
    drift_ndn.networks[0].xstim_n = 'Xdrift'

    drift_ndn.fit( data, force_dict_training=True, train_inds=None, **lbfgs_pars, verbose=0, version=1)
    LL = drift_ndn.eval_models(data[data.val_blks], null_adjusted=False)[0]

    drift_model = Model(drift_ndn, LL)

    with open(drift_model_filename, 'wb') as f:
        pickle.dump(drift_model, f)

    return drift_model

drifts = []
#for i, cc in enumerate(range(NCv)):
for i, cc in enumerate([5,23,39,60,4,93,15,36,37,44]):
    drift = fit_drift(i, cc, folder_name)
    drifts.append(drift)

# In[8]:
def fit_glm(i, cc, folder):
    glm_model_filename = 'models/'+folder+'/glm_model_cc'+str(cc)+'.pkl'

    drift_weights = drifts[i].ndn.networks[0].layers[0].weight.data[:,0]

    # continue if the file already exists
    if os.path.isfile(glm_model_filename):
        # load the model and continue
        print('loading model', cc)
        with open(glm_model_filename, 'rb') as f:
            glm = pickle.load(f)
        return glm

    data.set_cells(valET[cc])

    LLsNULL = drifts[i].LLs

    # optimize the d2x, d2t, d2xt, and glocalx first
    def objective(key, val, best_model):
        if best_model is None:
            glm = NDN.NDN(ffnet_list = [stim_net, drift_net, net_comb], loss_type='poisson')
            glm.block_sample=True
            glm.networks[1].layers[0].weight.data[:,0] = deepcopy(drift_weights)
            glm.networks[1].layers[0].set_parameters(val=False)
            glm.networks[2].layers[0].set_parameters(val=False,name='weight')
        else:
            glm = deepcopy(best_model)

        # set the reg_vals for the key
        glm.networks[0].layers[0].reg.vals[key] = val

        glm.fit(data, force_dict_training=True, **lbfgs_pars)
        LL = glm.eval_models(data[data.val_blks], null_adjusted=False)[0]

        null_adjusted_LL = LLsNULL - LL

        return glm, null_adjusted_LL

    optimizer = Optimizer(objective, val_lists_map)

    best_model, best_LL, best_vals, best_LLs = optimizer.optimize()

    glm_model = Model(best_model, best_LL)

    with open(glm_model_filename, 'wb') as f:
        pickle.dump(glm_model, f)

    return glm_model

glms = []
#for i, cc in enumerate(range(NCv)):
for i, cc in enumerate([5,23,39,60,4,93,15,36,37,44]):
    glm = fit_glm(i, cc, folder_name)
    glms.append(glm)

# In[9]:
# fit a GQM using my simpler Optimizer
def fit_gqm2(i, cc, folder):
    gqm_model_filename = 'models/'+folder+'/gqm_model_cc'+str(cc)+'.pkl'

    drift_weights = drifts[i].ndn.networks[0].layers[0].weight.data[:,0]

    # continue if the file already exists
    if os.path.isfile(gqm_model_filename):
        # load the model and continue
        print('loading model', cc)
        with open(gqm_model_filename, 'rb') as f:
            gqm = pickle.load(f)
        return gqm

    data.set_cells(valET[cc])

    LLsNULL = drifts[i].LLs

    # get the best reg_vals for the GLM
    best_reg_vals = glms[i].ndn.networks[0].layers[0].reg.vals

    # optimize the d2x, d2t, d2xt, and glocalx first
    def objective(key, val, best_model):
        # get the best reg_vals for the GLM
        stim_net['layer_list'][0]['reg_vals'] = deepcopy(best_reg_vals)

        if best_model is None:
            gqm = NDN.NDN(ffnet_list = [stim_net, drift_net, stim_qnet, net2_comb], loss_type='poisson')
            gqm.networks[0].layers[0] = deepcopy(glms[i].ndn.networks[0].layers[0])
            gqm.block_sample=True
            gqm.networks[3].layers[0].set_parameters(val=False,name='weight')
            gqm.networks[1].layers[0].weight.data[:,0] = deepcopy(drift_weights)
            gqm.networks[1].layers[0].set_parameters(val=False)
        else:
            gqm = deepcopy(best_model)

        # set the reg_vals for the key
        gqm.networks[2].layers[0].reg.vals[key] = val

        gqm.fit(data, force_dict_training=True, **lbfgs_pars)
        LL = gqm.eval_models(data[data.val_blks], null_adjusted=False)[0]

        null_adjusted_LL = LLsNULL - LL

        return gqm, null_adjusted_LL

    optimizer = Optimizer(objective, val_lists_map)

    best_model, best_LL, best_vals, best_LLs = optimizer.optimize()

    gqm_model = Model(best_model, best_LL)

    with open(gqm_model_filename, 'wb') as f:
        pickle.dump(gqm_model, f)

    return gqm_model

gqms = []
#for i, cc in enumerate(range(NCv)):
for i, cc in enumerate([5,23,39,60,4,93,15,36,37,44]):
    gqm = fit_gqm2(i, cc, folder_name)
    gqms.append(gqm)
