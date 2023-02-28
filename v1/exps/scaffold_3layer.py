import sys
sys.path.insert(0, '../lib')
sys.path.insert(0, '../')

import torch
import copy
import itertools as it
import numpy as np

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
datadir = '../Mdata/'
num_lags = 10
dataset = MultiDataset(
    datadir=datadir,
    filenames=['expt04'],
    include_MUs=False,
    time_embed=True,
    num_lags=num_lags)

adam_pars = utils.create_optimizer_params(
    optimizer_type='AdamW',
    batch_size=2000,
    num_workers=0,
    learning_rate=0.01,
    early_stopping_patience=4,
    optimize_graph=False,
    weight_decay = 0.1)
    #max_epochs=1)
adam_pars['device'] = device


def cnim_scaffold(num_filters, num_inh_percent, reg_vals, kernel_widths):
    conv_layer0 = m.ConvolutionalLayer(
        num_filters=num_filters[0],
        num_inh=int(num_filters[0]*num_inh_percent),
        filter_dims=kernel_widths[0],
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        reg_vals=reg_vals)
    conv_layer1 = m.ConvolutionalLayer().like(conv_layer0)
    conv_layer1.params['num_filters'] = num_filters[1]
    conv_layer1.params['num_inh'] = int(num_filters[1]*num_inh_percent)
    conv_layer1.params['filter_dims'] = kernel_widths[1]
    conv_layer2 = m.ConvolutionalLayer().like(conv_layer0)
    conv_layer2.params['num_filters'] = num_filters[2]
    conv_layer2.params['num_inh'] = int(num_filters[2]*num_inh_percent)
    conv_layer2.params['filter_dims'] = kernel_widths[2]

    readout_layer0 = m.Layer(
        pos_constraint=True, # because we have inhibitory subunits on the first layer
        norm_type=m.Norm.none,
        NLtype=m.NL.softplus,
        bias=True,
        initialize_center=True,
        reg_vals={'glocalx': 0.01}
    )

    inp_stim = m.Input(covariate='stim', input_dims=[1,36,1,10])

    core_net = m.Network(layers=[conv_layer0, conv_layer1, conv_layer2],
                             network_type=m.NetworkType.scaffold,
                             name='core')
    readout_net = m.Network(layers=[readout_layer0],
                            name='readout')
    # this is set as a starting point, but updated on each iteration
    output_11 = m.Output(num_neurons=11)

    inp_stim.to(core_net)
    core_net.to(readout_net)
    readout_net.to(output_11)
    return m.Model(output_11, verbose=True)


# TODO: also use multiunits (MUs)
# TODO: also initialize the models a few times to compare across initializations
# TODO: also try TCNIM (vs just spatial convolutions with different filter widths and heights)
# ------------------------------
# TODO: create a new regularization method penalizing the earlier weights more,
#       forcing it to learn more about the more recent information
# TODO: also add in rotation and translation invariance into the Trainer and Model
# TODO: also update DataLoader to sample across experiments more evenly
# parameters to iterate over
experiment_name = 'cnim_scaffold_3_layer_v4'
experiment_desc = 'Try freezing the weights'
expts = [['expt04', 'expt05'], ['expt04', 'expt05', 'expt06']]
copy_weightses = [True] # we have to copy the weights if we are freezing them
freeze_weightses = [True, False]
num_filterses = [[24, 20, 16]]
num_inh_percents = [0.75]
kernel_widthses = [[21, 21, 21]]
reg_valses = [{'d2xt': 0.01, 'l1': 0.0001, 'center': 0.01, 'bcs': {'d2xt': 1}}]
models = [{'cnim_scaffold': cnim_scaffold}]


# grid search through the desired parameters 
grid_search = it.product(num_filterses, num_inh_percents, kernel_widthses, reg_valses, copy_weightses, freeze_weightses, models)
print('====================================')
print('RUNNING', len(list(grid_search)), 'EXPERIMENTS')
print('====================================')
# regenerate this since we used up the iterations by getting the length...
grid_search = it.product(num_filterses, num_inh_percents, kernel_widthses, reg_valses, copy_weightses, freeze_weightses, models)


def generate_trial(prev_trials):
    trial_idx = 0
    for num_filters, num_inh_percent, kernel_widths, reg_vals, copy_weights, freeze_weights, model in grid_search:
        print('==========================================')
        print(num_filters, num_inh_percent, kernel_widths, reg_vals, copy_weights, freeze_weights, list(model.keys())[0])
        modelstr = list(model.keys())[0]
        modelfunc = list(model.values())[0]

        # make the model
        model = modelfunc(num_filters, num_inh_percent, reg_vals, kernel_widths)

        for expt in expts:
            print('Loading dataset for', expt)
            dataset_params = {
                'datadir': datadir,
                'filenames': expt,
                'include_MUs': False,
                'time_embed': True,
                'num_lags': num_lags
            }
            expt_dataset = MultiDataset(**dataset_params)
            expt_dataset.set_cells() # specify which cells to use (use all if no params provided)

            eval_params = {
                'null_adjusted': True
            }

            # update model based on the provided params
            # modify the model_template.output to match the data.NC before creating
            print('Updating model output neurons to:', expt_dataset.NC)
            model.update_num_neurons(expt_dataset.NC)

            if copy_weights and len(prev_trials) > 0:
                prev_trial = prev_trials[-1]
                try:
                    # freeze weights if set as well
                    if freeze_weights:
                        model.freeze_weights(network_names=['core'])
                    model.use_weights_from(prev_trial.model)
                except TypeError as error:
                    print(error) # eat error and continue (it is expected)
            else:
                print(':: NO WEIGHTS YET ::')

            # track the specific parameters going into this trial
            trial_params = {
                'null_adjusted_LL': True,
                'num_filters': ','.join([str(a) for a in num_filters]),
                'num_inh_percent': num_inh_percent,
                'expt': '+'.join(expt),
                'kernel_widths': ','.join([str(a) for a in kernel_widths]),
                'copy_weights': copy_weights,
                'freeze_weights': freeze_weights,
                'modelstr': modelstr
            }
            # add individual reg_vals to the trial_params
            for k,v in reg_vals.items():
                trial_params[k] = v

            trial_info = exp.TrialInfo(name=modelstr+str(trial_idx),
                                       description=modelstr+' with specified parameters',
                                       trial_params=trial_params,
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
experiment = exp.Experiment(name=experiment_name,
                            description=experiment_desc,
                            generate_trial=generate_trial,
                            experiment_location='../experiments',
                            overwrite=exp.Overwrite.overwrite)
experiment.run(device)
