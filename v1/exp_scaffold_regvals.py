import torch

# NDN tools
import NDNT.utils as utils # some other utilities
from NTdatasets.cumming.monocular import MultiDataset
from NDNT.modules.layers import *
from NDNT.networks import *

import experiment as exp
import model_factory as mf
import model as m

device = torch.device("cuda:1")
dtype = torch.float32

# load sample dataset to construct the model appropriately
datadir = './Mdata/'
num_lags = 10
expts = ['expt04']
data = MultiDataset(
    datadir=datadir, filenames=expts, include_MUs=False,
    time_embed=True, num_lags=num_lags )


# create the Model
conv_layer0 = m.ConvolutionalLayer(
    norm_type=m.Norm.none,
    NLtype=m.NL.relu,
    bias=False,
    initialize_center=True,
    reg_vals=[
        {'d2xt': 0.0001, 'center': None, 'bcs':{'d2xt':1} },
        {'d2xt': 0.001, 'center': None, 'bcs':{'d2xt':1} }
    ],
    num_filters=8,
    filter_dims=21,
    window='hamming',
    output_norm='batch',
    num_inh=4
)
conv_layer1 = m.ConvolutionalLayer().like(conv_layer0)
conv_layer1.params['num_filters'] = [8]
conv_layer1.params['num_inh'] = [4]
conv_layer1.params['filter_dims'] = [9]
conv_layer2 = m.ConvolutionalLayer().like(conv_layer0)
conv_layer2.params['num_filters'] = [4]
conv_layer2.params['num_inh'] = [2]
conv_layer2.params['filter_dims'] = [9]

readout_layer0 = m.Layer(
    norm_type=m.Norm.none,
    NLtype=m.NL.softplus,
    bias=True,
    reg_vals=[
        {'glocalx': 0.001}],
    pos_constraint=True
)

inp_stim = m.Input(covariate='stim', input_dims=data.stim_dims)

scaffold_net = m.Network(layers=[conv_layer0, conv_layer1, conv_layer2],
                         network_type=m.NetworkType.scaffold,
                         name='scaffold')
readout_net = m.Network(layers=[readout_layer0],
                        name='readout')
output_11 = m.Output(num_neurons=data.NC)

inp_stim.to(scaffold_net)
scaffold_net.to(readout_net)
readout_net.to(output_11)
model_template = m.Model(output_11)


# create ADAM params
adam_pars = utils.create_optimizer_params(
    optimizer_type='AdamW', batch_size=2000, num_workers=0,
    learning_rate=0.01, early_stopping_patience=4,
    optimize_graph=False, weight_decay = 0.1)
adam_pars['device'] = device


# run experiment
exps_to_try = [['expt04'],
               ['expt04', 'expt05'],
               ['expt04', 'expt05', 'expt06'],
               ['expt04', 'expt05', 'expt06', 'expt07'],
               ['expt04', 'expt05', 'expt06', 'expt07', 'expt08']]
exp_regvals = exp.Experiment('scaffold_conL0_regvals_comb', model_template, datadir, exps_to_try,
                             num_lags=num_lags, fit_params=adam_pars, overwrite=True)

exp_regvals.run()
