# Dataset preprocessor to prepare the data for training
import numpy as np
import torch
from time import time
import scipy.io as sio
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
sys.path.append('../lib')

from NTdatasets.generic import GenericDataset
from NTdatasets.cumming.monocular import MultiDataset
import NTdatasets.conway.cloud_datasets as datasets

import NDNT.utils as utils # some other utilities
import ColorDataUtils.ConwayUtils as CU
import ColorDataUtils.EyeTrackingUtils as ETutils

import runner2 as r
import model as m
import experiment as exp
import plot

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
#data.draw_stim_locations(top_corner = new_tc, L=NX)

data.assemble_stimulus(top_corner=[new_tc[0], new_tc[1]], L=NX, fixdot=0, shifts=-shifts)


goodfix = np.where(ETmetrics[:,1] < 0.80)[0]
valfix = torch.zeros([ETmetrics.shape[0], 1], dtype=torch.float32)
valfix[goodfix] = 1.0
# Test base-level performance (full DFs and then modify DFs)
#DFsave = deepcopy(data2.dfs)  # this is also in data.dfs
data.dfs_out *= valfix


# load drift model
drift_weights = []
for cc in range(0, len(vallam)):
    e = exp.load('color_drift_cc' + str(cc), experiment_location='../experiments')
    drift_weights.append(e.trials[0].model.NDN.networks[0].layers[0].weight)


trainer_params = r.TrainerParams(num_lags=num_lags,
                                 device="cuda:1", # use the second GPU
                                 max_epochs=3, # just for testing
                                 batch_size=20,
                                 history_size=100,
                                 max_iter=500,
                                 include_MUs=True,
                                 init_num_samples=0,
                                 bayes_num_steps=0,
                                 num_initializations=1,
                                 block_sample=True,
                                 trainer_type=r.TrainerType.lbfgs)

for cc in range(0, 1): # len(vallam)
    drift_dims = [1, 1, 1, data.Xdrift.shape[1]]
    inp_drift = m.Input(covariate='Xdrift', input_dims=drift_dims)

    drift_layer = m.Layer(
        weights=drift_weights[cc],
        freeze_weights=True,
        NLtype=m.NL.linear,
        num_filters=1,
        bias=False,
        reg_vals={'d2t': r.Sample(default=Dreg, typ=r.RandomType.float, values=[Dreg], start=0.00001, end=0.1),
                  'bcs': {'d2t': 0}})
    drift_net = m.Network(layers=[drift_layer],
                          name='drift')

    inp_stim = m.Input(covariate='stim', input_dims=data.stim_dims)

    glm_layer = m.TemporalLayer(
        NLtype=m.NL.linear,
        num_filters=1,
        bias=False,
        initialize_center=True,
        num_lags=num_lags,
        reg_vals={'d2x': r.Sample(default=20, typ=r.RandomType.int, values=[20], start=10, end=30),
                  'd2t': r.Sample(default=1, typ=r.RandomType.int, values=[1], start=1, end=4),
                  'l1': r.Sample(default=0.1, typ=r.RandomType.float, values=[0.1], start=0.01, end=1.0),
                  'glocalx': r.Sample(default=10.0, typ=r.RandomType.float, values=[10.0], start=5.0, end=15.0),
                  'edge_t': r.Sample(default=10, typ=r.RandomType.int, values=[10], start=5, end=15)})
    glm_net = m.Network(layers=[glm_layer],
                        name='glm')

    gqm_layer = m.TemporalLayer(
        NLtype=m.NL.square,
        num_filters=2,
        bias=False,
        initialize_center=True,
        num_lags=num_lags,
        reg_vals={'d2x': r.Sample(default=20, typ=r.RandomType.int, values=[20], start=10, end=30),
                  'd2t': r.Sample(default=1, typ=r.RandomType.int, values=[1], start=1, end=4),
                  'l1': r.Sample(default=0.1, typ=r.RandomType.float, values=[0.1], start=0.01, end=1.0),
                  'glocalx': r.Sample(default=10.0, typ=r.RandomType.float, values=[10.0], start=5.0, end=15.0),
                  'edge_t': r.Sample(default=10, typ=r.RandomType.int, values=[10], start=5, end=15)})
    gqm_net = m.Network(layers=[gqm_layer],
                        name='gqm')

    inp_drift.to(drift_net)
    inp_stim.to(glm_net)
    inp_stim.to(gqm_net)

    output_glm = m.Output(num_neurons=1)
    m.Add(networks=[drift_net, glm_net], bias=True).to(output_glm)
    glm_model = m.Model(output_glm,
                        name='GLM',
                        create_NDN=False, verbose=True)

    # set the cells to a single cell
    data.set_cells([vallam[cc]])

    runner = r.Runner(experiment_name='color_glm_cc' + str(cc),
                      dataset_expt=[fn], # name of the experiment
                      dataset=data,
                      dataset_on_gpu=False,
                      model_template=glm_model,
                      trainer_params=trainer_params,
                      overwrite=False,
                      trial_params={'cc': cc})
    runner.run()
    
