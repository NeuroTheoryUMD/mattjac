import sys
sys.path.append('../')

import numpy as np
import optuna
import torch
import pickle

# NDN tools
import NDNT.utils as utils # some other utilities
import NDNT.NDNT as NDN
from NDNT.modules.layers import *
from NDNT.networks import *
from time import time

import matplotlib
import matplotlib.pyplot as plt
import ColorDataUtils.mattplotlib as mplt
from ColorDataUtils.multidata_utils import MultiExperiment
from NDNT.utils import imagesc   # because I'm lazy
from NDNT.utils import ss        # because I'm real lazy

device = torch.device("cuda:1")
dtype = torch.float32

datadir = '/home/dbutts/ColorV1/Data/'
dirname = '/home/dbutts/ColorV1/CLRworkspace/'


import NDNT.utils as utils # some other utilities
import NDNT.NDNT as NDN
from NDNT.modules.layers import *
from NDNT.networks import *

sys.path.append('../lib')

# Load Data
num_lags = 10
expts = ['expt04']
# this can handle multiple experiments
#expts = ['expt04', 'expt05']

from NTdatasets.cumming.monocular import MultiDataset
data = MultiDataset(
    datadir='../Mdata/', filenames=expts, include_MUs=False,
    time_embed=True, num_lags=num_lags )
imagesc(data.dfs.detach().numpy())
print("%d cells, %d time steps."%(data.NC, data.NT))


adam_parsT = utils.create_optimizer_params(
    optimizer_type='AdamW',
    batch_size=2, # * 240 timesteps
    num_workers=0,
    learning_rate=0.0017,
    early_stopping_patience=1,#4,
    optimize_graph=False,
    weight_decay=0.235,
    max_epochs=1)
adam_parsT['device'] = device

# setup
data.device = device
NCv = data.NC
NT = data.robs.shape[0]


conv_layer0 = ConvLayer.layer_dict(
    num_filters=6,
    num_inh=3,
    bias=False,
    norm_type=1,
    filter_dims=21,
    NLtype='relu',
    initialize_center=False,
    input_dims = data.stim_dims)

readout_layer = NDNLayer.layer_dict(
    num_filters=11,
    pos_constraint=True, # because we have inhibitory subunits on the first layer
    norm_type=0,
    NLtype='softplus',
    bias=True,
    initialize_center=True,
    reg_vals={'glocalx': 0.01})

conv_net = FFnetwork.ffnet_dict(
    xstim_n='stim',
    layer_list=[conv_layer0, readout_layer])

cnn = NDN.NDN(ffnet_list = [conv_net],
              loss_type='poisson')
cnn.block_sample = True

cnn.fit(data, **adam_parsT, verbose=2)
LLs = cnn.eval_models(data, data_inds=data.val_inds, batch_size=5)
print(np.mean(LLs))


with open('cnn_block_sample.pkl', 'wb') as f:
    pickle.dump(cnn, f)

