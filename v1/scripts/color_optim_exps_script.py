# Dataset preprocessor to prepare the data for training
import numpy as np
import torch
import pickle
import os
import optuna
from time import time
import scipy.io as sio
import matplotlib.pyplot as plt

from NTdatasets.generic import GenericDataset
from NTdatasets.cumming.monocular import MultiDataset
import NTdatasets.conway.cloud_datasets as datasets

import ColorDataUtils.ConwayUtils as CU
import ColorDataUtils.EyeTrackingUtils as ETutils

# NDN tools
import NDNT.utils as utils # some other utilities
import NDNT.NDNT as NDN
from NDNT.modules.layers import *
from NDNT.networks import *
from time import time
from copy import deepcopy

import sys
sys.path.append('./lib')
import runner2 as r
import model as m
import experiment as exp
import plot

from NDNT.utils import imagesc   # because I'm lazy
from NDNT.utils import ss        # because I'm real lazy

class Model:
    def __init__(self, ndn_model, LLs, trial):
        self.ndn_model = ndn_model
        self.LLs = LLs
        self.trial = trial


datadir = '/home/dbutts/ColorV1/Data/'
dirname = '/home/dbutts/ColorV1/CLRworkspace/'
fn = 'Jocamo_220715_full_CC_ETCC_nofix_v08'
#fn = 'Jocamo_220727_full_CC_ETCC_nofix_v08'
num_lags=12

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
matdat = sio.loadmat(datadir+'Jocamo_220715_full_CC_ETCC_v08_ETupdate.mat')
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

print(lam_units, 'laminar units', data.robs.shape, 'robs', data.dfs.shape, 'dfs')


# Set cells-to-analyze and pull best model configuration and mus
Reff = torch.mul(data.robs[:, lam_units], data.dfs[:, lam_units]).numpy()
nspks = np.sum(Reff, axis=0)
a = np.where(nspks > 10)[0]
vallam = lam_units[a]
NCv = len(vallam)
print("%d out of %d units used"%(len(vallam), len(lam_units)))

## CONVERT LLsNULL, which is based on 

# Read in previous data
dirname2 = dirname+'0715/et/'
matdat = sio.loadmat(dirname2+'LLsGLM.mat')
Dreg = matdat['Dreg']
top_cornerUT = matdat['top_corner'][:, 0]

data.set_cells(vallam)

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

top_corner_lam = [938, 515]

# Make 60x60 STAs (and GLMs)
Xshift = 0 #8+4 
Yshift = 0 #-10+4
NX = 60

new_tc = np.array([top_corner_lam[0]-Xshift, top_corner_lam[1]-Yshift], dtype=np.int64)
data.draw_stim_locations(top_corner = new_tc, L=NX)

data.assemble_stimulus(top_corner=[new_tc[0], new_tc[1]], L=NX, fixdot=0, shifts=-shifts)


goodfix = np.where(ETmetrics[:,1] < 0.80)[0]
valfix = torch.zeros([ETmetrics.shape[0], 1], dtype=torch.float32)
valfix[goodfix] = 1.0
# Test base-level performance (full DFs and then modify DFs)
#DFsave = deepcopy(data2.dfs)  # this is also in data.dfs
data.dfs_out *= valfix


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device0 = torch.device("cpu")
dtype = torch.float32

goodfix = np.where(ETmetrics[:,1] < 0.80)[0]
valfix = torch.zeros([ETmetrics.shape[0], 1], dtype=torch.float32)
valfix[goodfix] = 1.0
# Test base-level performance (full DFs and then modify DFs)
#DFsave = deepcopy(data2.dfs)  # this is also in data.dfs
data.dfs_out *= valfix
print("%0.1f%% fixations remaining"%(100*len(goodfix)/ETmetrics.shape[0]))

lbfgs_pars = utils.create_optimizer_params(
    optimizer_type='lbfgs',
    tolerance_change=1e-8,
    tolerance_grad=1e-8,
    history_size=100,
    batch_size=20,
    max_epochs=3,
    max_iter = 500,
    device = device)

adam_parsT = utils.create_optimizer_params(
    optimizer_type='AdamW', batch_size=5, num_workers=0,
    learning_rate=0.01, early_stopping_patience=10,  # changed from 4
    optimize_graph=False, weight_decay = 0.2)
adam_parsT['device'] = device
adam_parsT['accumulated_grad_batches']=6


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

X2opt = np.zeros([NCv,2])
T2opt = np.zeros([NCv,2])
GLopt = np.zeros([NCv,2])
L1opt = np.zeros([NCv,2])
LLsNULL = np.zeros(NCv)
LLsLR = np.zeros([NCv,4])
LLsQR = np.zeros([NCv,4])

rvals = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
rvalsG = [0.001, 0.1, 1, 10, 100, 1000] # glocal
rvalsL = [0.001, 0.01, 0.1, 1] # L1


matt_drifts = []
dan_drifts = []
matt_glms = []
dan_glms = []
for cc in range(NCv):
    matt_drift_filename = 'models/optim/matt_drift_cc'+str(cc)+'.pkl'

    # continue if the file already exists
    if os.path.isfile(matt_drift_filename):
        # load the model and continue
        print('loading model', cc)
        with open(matt_drift_filename, 'rb') as f:
            matt_drifts.append(pickle.load(f))

    matt_glm_filename = 'models/optim/matt_glm_cc'+str(cc)+'.pkl'

    # # continue if the file already exists
    if os.path.isfile(matt_glm_filename):
        # load the model and continue
        print('loading model', cc)
        with open(matt_glm_filename, 'rb') as f:
            matt_glms.append(pickle.load(f))

matt_gqms = []
for cc in range(NCv):
    gqm_filename = 'models/optim/matt_gqm_cc'+str(cc)+'.pkl'

    drift_weights = matt_drifts[cc].ndn_model.networks[0].layers[0].weight.data[:,0]

    # # continue if the file already exists
    if os.path.isfile(gqm_filename):
        # load the model and continue
        print('loading model', cc)
        with open(gqm_filename, 'rb') as f:
            matt_gqms.append(pickle.load(f))
        continue

    data.set_cells([vallam[cc]])

    LLsNULL_cc = matt_drifts[cc].LLs

    # get the best reg_vals for the GLM
    #best_reg_vals = matt_glms[cc].ndn_model.networks[0].layers[0].reg.vals
    #stim_net['layer_list'][0]['reg_vals'] = deepcopy(best_reg_vals)
    matt_gqm = NDN.NDN(ffnet_list = [stim_net, drift_net, stim_qnet, net2_comb], loss_type='poisson')
    matt_gqm.networks[0].layers[0] = deepcopy(matt_glms[cc].ndn_model.networks[0].layers[0])
    matt_gqm.block_sample=True
    matt_gqm.networks[3].layers[0].set_parameters(val=False,name='weight')
    matt_gqm.networks[1].layers[0].weight.data[:,0] = deepcopy(
        matt_drifts[cc].ndn_model.networks[0].layers[0].weight.data[:,0])
    matt_gqm.networks[1].layers[0].set_parameters(val=False)

    matt_gqms_temp = []

    def objective(trial):
        lbfgs_pars = utils.create_optimizer_params(
            optimizer_type='lbfgs',
            tolerance_change=1e-10*trial.suggest_int('tolerance_change', 10, 1e6, log=True),
            tolerance_grad=1e-10*trial.suggest_int('tolerance_grad', 10, 1e6, log=True),
            history_size=100,
            batch_size=20,
            max_epochs=3,
            max_iter = 500,
            device = device)

        # adjust the stim_net regularization
        matt_gqm.networks[0].layers[0].reg.vals['d2x'] = 1e-5*trial.suggest_int('d2x', 10, 1e10, log=True)
        matt_gqm.networks[0].layers[0].reg.vals['d2t'] = 1e-5*trial.suggest_int('d2t', 10, 1e10, log=True)
        matt_gqm.networks[0].layers[0].reg.vals['d2xt'] = 1e-5*trial.suggest_int('d2xt', 10, 1e10, log=True)
        matt_gqm.networks[0].layers[0].reg.vals['l1'] = 1e-5*trial.suggest_int('l1', 10, 1e10, log=True)
        matt_gqm.networks[0].layers[0].reg.vals['glocalx'] = 1e-5*trial.suggest_int('glocalx', 10, 1e10, log=True)
        matt_gqm.networks[0].layers[0].reg.vals['edge_t'] = 1e-5*trial.suggest_int('edge_t', 10, 1e10, log=True)

        # adjust the stim_qnet regularization
        matt_gqm.networks[2].layers[0].reg.vals['d2x'] = 1e-5*trial.suggest_int('d2x', 10, 1e10, log=True)
        matt_gqm.networks[2].layers[0].reg.vals['d2t'] = 1e-5*trial.suggest_int('d2t', 10, 1e10, log=True)
        matt_gqm.networks[2].layers[0].reg.vals['d2xt'] = 1e-5*trial.suggest_int('d2xt', 10, 1e10, log=True)
        matt_gqm.networks[2].layers[0].reg.vals['l1'] = 1e-5*trial.suggest_int('l1', 10, 1e10, log=True)
        matt_gqm.networks[2].layers[0].reg.vals['glocalx'] = 1e-5*trial.suggest_int('glocalx', 10, 1e10, log=True)
        matt_gqm.networks[2].layers[0].reg.vals['edge_t'] = 1e-5*trial.suggest_int('edge_t', 10, 1e10, log=True)

        matt_gqm.fit( data, force_dict_training=True, **lbfgs_pars)
        LL = matt_gqm.eval_models(data[data.val_blks], null_adjusted=False)[0]

        gqm_model = Model(matt_gqm, LLsNULL_cc-LL, trial)
        matt_gqms_temp.append(gqm_model)

        return LLsNULL_cc-LL

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.HyperbandPruner())

    study.optimize(objective, n_trials=40)

    matt_gqms.append(matt_gqms_temp[study.best_trial.number])

    with open(gqm_filename, 'wb') as f:
        matt_gqms[cc].trial = study
        pickle.dump(matt_gqms[cc], f)

    print(study.best_trial.number, study.best_params)
    
    
