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
    datadir=datadir, filenames=expts, include_MUs=False,
    time_embed=True, num_lags=num_lags )

# for each data
# load sample dataset to construct the model appropriately
datadir = './Mdata/'
num_lags = 10
expts = [['expt04'], 
         ['expt05'], 
         ['expt06'],
         ['expt04', 'expt05'],
         ['expt05', 'expt06'],
         ['expt06', 'expt07'],
         ['expt04', 'expt07'],
         ['expt05', 'expt06', 'expt07']]

adam_pars = utils.create_optimizer_params(
    optimizer_type='AdamW', batch_size=2000, num_workers=0,
    learning_rate=0.01, early_stopping_patience=4,
    optimize_graph=False, weight_decay = 0.1)
adam_pars['device'] = device

lbfgs_pars = utils.create_optimizer_params(
    optimizer_type='lbfgs',
    tolerance_change=1e-10,
    tolerance_grad=1e-10,
    batch_size=2000,
    history_size=100,
    max_iter=2000)
lbfgs_pars['device'] = device


# create the Model
glm_layer0 = m.Layer(
    NLtype=m.NL.linear,
    bias=True,
    initialize_center=True,
    reg_vals={'d2xt': 0.02, 'bcs':{'d2xt': 1}})
inp_stim = m.Input(covariate='stim', input_dims=dataset.stim_dims)
glm_net = m.Network(layers=[glm_layer0], name='GLM')
output_11 = m.Output(num_neurons=dataset.NC)
inp_stim.to(glm_net)
glm_net.to(output_11)
model = m.Model(output_11, verbose=True)

print('=======')
model.NDN.list_parameters()
#sys.exit()


def generate_trial(prev_trials):
    trial_idx = 0
    for expt in expts:
        print('prev_trials', len(prev_trials))

        print('Loading dataset for', expt)
        dataset_params = {
            'datadir': datadir,
            'filenames': expt,
            'include_MUs': False,
            'time_embed': True,
            'num_lags': num_lags
        }
        expt_dataset = MultiDataset(**dataset_params)
        expt_dataset.set_cells()

        eval_params = {
            'null_adjusted': True
        }

        # update model based on the provided params
        # modify the model_template.output to match the data.NC before creating
        print('Updating model output neurons to:', expt_dataset.NC)
        model.update_num_neurons(expt_dataset.NC)

        # track the specific parameters going into this trial
        trial_params = {
            'expt': expt
        }
        
        trial_info = exp.TrialInfo(name='GLM_'+'+'.join(expt),
                                   description='GLM trained on different datasets',
                                   trial_params=trial_params,
                                   dataset_params=dataset_params,
                                   dataset_class=MultiDataset,
                                   fit_params=lbfgs_pars,
                                   eval_params=eval_params)

        trial = exp.Trial(trial_info=trial_info,
                          model=model,
                          dataset=expt_dataset)
        trial_idx += 1
        yield trial


# run the experiment
experiment = exp.Experiment(name='exp_GLM',
                            description='Simple GLM test',
                            generate_trial=generate_trial,
                            experiment_location='experiments',
                            overwrite=exp.Overwrite.overwrite)
experiment.run(device)
