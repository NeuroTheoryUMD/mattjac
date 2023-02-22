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
adam_pars['device'] = device


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

    scaffold_net = m.Network(layers=[conv_layer0, conv_layer1, conv_layer2],
                             network_type=m.NetworkType.scaffold,
                             name='scaffold')
    readout_net = m.Network(layers=[readout_layer0],
                            name='readout')
    # this is set as a starting point, but updated on each iteration
    output_11 = m.Output(num_neurons=11)

    inp_stim.to(scaffold_net)
    scaffold_net.to(readout_net)
    readout_net.to(output_11)
    return m.Model(output_11, verbose=True)


def tcnim_scaffold(num_filters, num_inh, reg_vals, kernel_width, kernel_height):
    tconv_layer0 = m.TemporalConvolutionalLayer(
        num_filters=num_filters,
        num_inh=num_inh,
        filter_dims=[1,kernel_width,1,kernel_height], # [C, w, h, t]
        window='hamming',
        padding='spatial',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        reg_vals=reg_vals)
    tconv_layer1 = m.ConvolutionalLayer().like(tconv_layer0)
    tconv_layer1.params['num_filters'] = num_filters
    tconv_layer1.params['num_inh'] = num_filters//2
    tconv_layer1.params['filter_dims'] = 9
    tconv_layer2 = m.ConvolutionalLayer().like(tconv_layer0)
    tconv_layer2.params['num_filters'] = 4
    tconv_layer2.params['num_inh'] = 2
    tconv_layer2.params['filter_dims'] = 9

    readout_layer0 = m.Layer(
        pos_constraint=True, # because we have inhibitory subunits on the first layer
        norm_type=m.Norm.none,
        NLtype=m.NL.softplus,
        bias=True,
        initialize_center=True,
        reg_vals={'glocalx': 0.01}
    )

    inp_stim = m.Input(covariate='stim', input_dims=[1,36,1,10])

    scaffold_net = m.Network(layers=[tconv_layer0, tconv_layer1, tconv_layer2],
                             network_type=m.NetworkType.scaffold,
                             name='scaffold')
    readout_net = m.Network(layers=[readout_layer0],
                            name='readout')
    output_11 = m.Output(num_neurons=11)

    inp_stim.to(scaffold_net)
    scaffold_net.to(readout_net)
    readout_net.to(output_11)
    return m.Model(output_11, verbose=True)


# parameters to iterate over
# TODO: also initialize the models a few times to compare across initializations
# TODO: also copy the weights
# TODO: also freeze the weights
# TODO: also use multiunits (MUs)
# TODO: also try TCNIM
# TODO: create a new regularization method penalizing the earlier weights more, forcing it to learn more about the more recent information
# TODO: also add in rotation and translation invariance into the Trainer and Model
# TODO: also update DataLoader to sample across experiments more evenly
experiment_name = 'cnim_scaffold_3_layer_v2'
experiment_desc = 'Compare scaffold across experiments, num_filters, and copying weights'
expts = [['expt04', 'expt05', 'expt06', 'expt07']]
copy_weightses = [True, False]
num_filterses = [[16, 16, 16], [24, 20, 16], [16, 20, 24]]
num_inh_percents = [0.25, 0.5, 0.75]
kernel_widthses = [[21, 21, 21]]
kernel_heightses = [[5, 5, 5]]
reg_valses = [{'d2xt': 0.01, 'l1': 0.0001, 'center': 0.01, 'bcs': {'d2xt': 1}}]
models = [{'cnim_scaffold': cnim_scaffold}]


# grid search through the desired parameters 
grid_search = it.product(num_filterses, num_inh_percents, kernel_widthses, kernel_heightses, reg_valses, models)
print('====================================')
print('RUNNING', len(list(grid_search)), 'EXPERIMENTS')
print('====================================')
# regenerate this since we used up the iterations by getting the length...
grid_search = it.product(num_filterses, num_inh_percents, kernel_widthses, kernel_heightses, reg_valses, models)


def generate_trial(prev_trials):
    trial_idx = 0
    for num_filters, num_inh_percent, kernel_widths, kernel_heights, reg_vals, model in grid_search:
        print('==========================================')
        print(num_filters, num_inh_percent, kernel_widths, kernel_heights, reg_vals, list(model.keys())[0])
        modelstr = list(model.keys())[0]
        modelfunc = list(model.values())[0]

        # make the model
        model = modelfunc(num_filters, num_inh_percent, reg_vals, kernel_widths, kernel_heights)

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

            # TODO: update this for the scaffold network
            if copy_weights and len(prev_trials) > 0:
                prev_trial = prev_trials[-1]
                # skip if this is the first of the new batch of sizes
                if prev_trial.trial_info.trial_params['num_filters'] == num_filters:
                    # copy the weights from the prev_trials[-1], into the current model
                    prev_model = prev_trial.model
                    prev_net0_layer0_weights = prev_model.NDN.networks[0].layers[0].weight
                    prev_net0_layer1_weights = prev_model.NDN.networks[0].layers[1].weight

                    print(':: WEIGHTS ::\n')
                    print(prev_model.NDN.networks[0].layers[0].weight.shape, end='-->')
                    print(model.NDN.networks[0].layers[0].weight.shape, end='\n')
                    print(prev_model.NDN.networks[0].layers[1].weight.shape, end='-->')
                    print(model.NDN.networks[0].layers[1].weight.shape)

                    # copy weights of first, conv, layer
                    model.NDN.networks[0].layers[0].weight = copy.deepcopy(prev_net0_layer0_weights)

                    # copy weights of last, readout, layer
                    # https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308/2
                    # clone the weights first
                    with torch.no_grad(): # have to make the tensors temporarily not require grad to do this
                        curr_net0_layer1_weights = model.NDN.networks[0].layers[1].weight
                        curr_net0_layer1_weights[:,:prev_net0_layer1_weights.shape[1]] = copy.deepcopy(prev_net0_layer1_weights)
                        model.NDN.networks[0].layers[1].weight = curr_net0_layer1_weights
                else:
                    print('NUM_FILTERS NOT THE SAME')
            else:
                print(':: NO WEIGHTS YET ::')


            # track the specific parameters going into this trial
            trial_params = {
                #'copy_weights': copy_weights,
                'null_adjusted_LL': True,
                'num_filters': ','.join([str(a) for a in num_filters]),
                'num_inh_percent': num_inh_percent,
                'expt': '+'.join(expt),
                'kernel_widths': ','.join([str(a) for a in kernel_widths]),
                'kernel_heights': ','.join([str(a) for a in kernel_heights]),
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
