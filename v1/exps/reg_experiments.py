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
from NTdatasets.generic import GenericDataset

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

fit_pars = utils.create_optimizer_params(
    optimizer_type='AdamW',
    batch_size=2000,
    num_workers=0,
    learning_rate=0.01,
    early_stopping_patience=4,
    optimize_graph=False,
    weight_decay = 0.1)
#max_epochs=1)
fit_pars['device'] = device
fit_pars['verbose'] = True
fit_pars['is_multiexp'] = True # to use the experiment_sampler


def cnim_scaffold(num_filters, num_inh_percent, reg_vals, kernel_widths, kernel_heights):
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
    conv_layer1.params['reg_vals'] = {'activity': reg_vals['activity']}
    conv_layer2 = m.ConvolutionalLayer().like(conv_layer0)
    conv_layer2.params['num_filters'] = num_filters[2]
    conv_layer2.params['num_inh'] = int(num_filters[2]*num_inh_percent)
    conv_layer2.params['filter_dims'] = kernel_widths[2]
    conv_layer2.params['reg_vals'] = {'activity': reg_vals['activity']}

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


# ------------------------------
# TODO: decrease receptive field size in the later layers (3 or 5)
# TODO: also add in rotation and translation invariance into the Trainer and Model
# TODO: try copying weights again and record when it is working better,
#       not sure why it wasn't working better in my latest experiments
#       FINISH UNIT TESTS
# TODO: reinitialize the models multiple times, and record these during experiments
# TODO: also try TCNIM (vs just spatial convolutions with different filter widths and heights)
#       see if convolutional filters over shorter time lags improve the fit or not
#  fix: "Given groups=1, weight of size [40, 1, 21, 3], expected input[2000, 32, 56, 8] to have 1 channels, but got 32 channels instead"
# TODO: create a new regularization method penalizing the earlier weights more,
#       forcing it to learn more about the more recent information (recency regularization)
# parameters to iterate over
experiment_name = 'reg_experiment_10'
experiment_desc = 'Comparing activity reg and weight regs and only act reg on the deeper layers.'
expts = [['expt04']]
          # 'expt01', 'expt02', 'expt03', 'expt04'
          # 'expt05', 'expt06', 'expt07', 'expt08',
          # 'expt09', 'expt10', 'expt11', 'expt12',
          # 'expt13', 'expt14', 'expt15', 'expt16',
          # 'expt17', 'expt18', 'expt19', 'expt20', 'expt21']]
copy_weightses = [False]
freeze_weightses = [False]
include_MUses = [False]
is_multiexps = [False]
batch_sizes = [6000]
num_filterses = [[16, 8, 8]]
num_inh_percents = [0.5]
kernel_widthses = [[21, 11, 5]]
kernel_heightses = [[3, 3, 3]]
# remove the l1 regularization for now and try different orders of magnitude for the activity regularization
# TODO: try nonneg loss instead of relu, see how that affects things
reg_valses = [{'activity': 1.0,  'd2xt': 0.01, 'center': 0.01, 'bcs': {'d2xt': 1}},
              {'activity': 0.75, 'd2xt': 0.01, 'center': 0.01, 'bcs': {'d2xt': 1}},
              {'activity': 0.5,  'd2xt': 0.01, 'center': 0.01, 'bcs': {'d2xt': 1}},
              {'activity': 0.25, 'd2xt': 0.01, 'center': 0.01, 'bcs': {'d2xt': 1}},
              {'activity': 0.1,  'd2xt': 0.01, 'center': 0.01, 'bcs': {'d2xt': 1}},
              {'activity': 0.0,  'd2xt': 0.01, 'center': 0.01, 'bcs': {'d2xt': 1}}]

models = [{'cnim_scaffold': cnim_scaffold}]


# grid search through the desired parameters 
grid_search = it.product(num_filterses, num_inh_percents, kernel_widthses, kernel_heightses, reg_valses, copy_weightses, freeze_weightses, include_MUses, is_multiexps, batch_sizes, models)
print('====================================')
print('RUNNING', len(list(grid_search)), 'EXPERIMENTS')
print('====================================')
# regenerate this since we used up the iterations by getting the length...
grid_search = it.product(num_filterses, num_inh_percents, kernel_widthses, kernel_heightses, reg_valses, copy_weightses, freeze_weightses, include_MUses, is_multiexps, batch_sizes, models)


# why are the LLs infinite for some neurons when include_MUs?
# the infinite ones are the ones that are not included in the list of SUs.
from torch.utils.data.dataset import Subset
def eval_function(model, dataset, device):
    # get just the single units from the dataset
    # make a dataset from these and use that as val_ds
    #val_ds = GenericDataset(dataset[dataset.val_inds], device=device)
    val_ds = Subset(dataset, dataset.val_inds)
    return model.NDN.eval_models(val_ds, null_adjusted=True)

def generate_trial(prev_trials):
    trial_idx = 0
    for num_filters, num_inh_percent, kernel_widths, kernel_heights, reg_vals, copy_weights, freeze_weights, include_MUs, is_multiexp, batch_size, model in grid_search:
        print('==========================================')
        print(num_filters, num_inh_percent, kernel_widths, kernel_heights, reg_vals, copy_weights, freeze_weights, include_MUs, is_multiexp, batch_size, list(model.keys())[0])
        modelstr = list(model.keys())[0]
        modelfunc = list(model.values())[0]

        fit_pars['is_multiexp'] = is_multiexp
        fit_pars['batch_size'] = batch_size

        # make the model
        model = modelfunc(num_filters, num_inh_percent, reg_vals, kernel_widths, kernel_heights)

        for expt in expts:
            print('Loading dataset for', expt)
            dataset_params = {
                'datadir': datadir,
                'filenames': expt,
                'include_MUs': include_MUs,
                'time_embed': True,
                'num_lags': num_lags
            }
            expt_dataset = MultiDataset(**dataset_params)
            expt_dataset.set_cells() # specify which cells to use (use all if no params provided)

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
                    print(':: TYPE ERROR ::', end=' ----> ')
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
                'kernel_heights': ','.join([str(a) for a in kernel_heights]),
                'copy_weights': copy_weights,
                'freeze_weights': freeze_weights,
                'include_MUs': include_MUs,
                'is_multiexp': is_multiexp,
                'batch_size': batch_size,
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
                                       fit_params=fit_pars)

            trial = exp.Trial(trial_info=trial_info,
                              model=model,
                              dataset=expt_dataset,
                              eval_function = eval_function)
            trial_idx += 1
            yield trial


# run the experiment
experiment = exp.Experiment(name=experiment_name,
                            description=experiment_desc,
                            generate_trial=generate_trial,
                            experiment_location='../experiments',
                            overwrite=exp.Overwrite.overwrite)
experiment.run(device)
