import torch
import sys

# NDN tools
import NDNT.utils as utils # some other utilities
from NTdatasets.cumming.monocular import MultiDataset
from NDNT.modules.layers import *
from NDNT.networks import *

import experiment as exp
import model as m

device = torch.device("cuda:1")
dtype = torch.float32

# load sample dataset to construct the model appropriately
datadir = './Mdata/'
num_lags = 10
expts = ['expt04']
dataset = MultiDataset(
    datadir=datadir, 
    filenames=expts, 
    include_MUs=False,
    time_embed=True, 
    num_lags=num_lags )

# for each data
# load sample dataset to construct the model appropriately
datadir = './Mdata/'
num_lags = 10
expts = [['expt04'],
         ['expt04', 'expt05'],
         ['expt04', 'expt05', 'expt06'],
         ['expt04', 'expt05', 'expt06', 'expt07']]

adam_pars = utils.create_optimizer_params(
    optimizer_type='AdamW', 
    batch_size=2000, 
    num_workers=0,
    learning_rate=0.01, 
    early_stopping_patience=4,
    optimize_graph=False, 
    weight_decay = 0.1)
adam_pars['device'] = device


# create the Model
def generate_trial(prev_trials):
    trial_idx = 0
    for expt in expts:
        for num_filters in [4,6,8]: # try a few filter amounts
            convolutional_layer = m.ConvolutionalLayer(
                num_filters=num_filters,
                num_inh=num_filters//2,
                filter_dims=21,
                window='hamming',
                NLtype=m.NL.relu, 
                norm_type=m.Norm.unit,
                bias=False,
                initialize_center=True,
                output_norm='batch',
                reg_vals={'d2xt': 0.01, 'l1':0.0001, 'center':0.01, 'bcs':{'d2xt':1}  })
            readout_layer = m.Layer(
                pos_constraint=True, # because we have inhibitory subunits on the first layer
                norm_type=m.Norm.none,
                NLtype=m.NL.softplus,
                bias=True,
                initialize_center=True,
                reg_vals={'glocalx': 0.01})
            inp_stim = m.Input(covariate='stim', input_dims=dataset.stim_dims)
            nim_net = m.Network(layers=[convolutional_layer, readout_layer], name='CNIM')
            output_11 = m.Output(num_neurons=dataset.NC)
            inp_stim.to(nim_net)
            nim_net.to(output_11)
            model = m.Model(output_11, verbose=True)
            
            print('Loading dataset for', expt)
            dataset_params = {
                'datadir': datadir,
                'filenames': expt,
                'include_MUs': False,
                'time_embed': True,
                'num_lags': num_lags
            }
            expt_dataset = MultiDataset(**dataset_params)
            expt_dataset.set_cells() # TODO: what does this do??
    
            eval_params = {
                'null_adjusted': True
            }
    
            # update model based on the provided params
            # modify the model_template.output to match the data.NC before creating
            print('Updating model output neurons to:', expt_dataset.NC)
            model.update_num_neurons(expt_dataset.NC)
    
            trial_info = exp.TrialInfo(name='CNIM_NF'+str(num_filters)+'_'+'+'.join(expt),
                                       description='CNIM train on the expt dataset with num_filters=NF in the first layer',
                                       dataset_params=dataset_params,
                                       dataset_class=MultiDataset,
                                       fit_params=adam_pars,
                                       eval_params=eval_params)
    
            trial = exp.Trial(trial_info=trial_info,
                              model=model,
                              dataset=expt_dataset)
            trial_idx += 1
            yield trial


# run the experiment
experiment = exp.Experiment(name='exp_CNIM',
                            description='Convolutional Nonlinear Input Model',
                            generate_trial=generate_trial,
                            experiment_location='experiments',
                            overwrite=exp.Overwrite.overwrite)
experiment.run(device)
