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



def cnim(num_filters, num_inh, reg_vals, kernel_width, kernel_height):
    convolutional_layer = m.ConvolutionalLayer(
        num_filters=num_filters,
        num_inh=num_inh,
        filter_dims=kernel_width,
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        reg_vals=reg_vals)
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
    return m.Model(output_11, verbose=True)


def cnim_scaffold(num_filterses, num_inh_percent, reg_vals, kernel_widths, kernel_heights):
    conv_layer0 = m.ConvolutionalLayer(
        num_filters=num_filterses[0],
        num_inh=int(num_filterses[0]*num_inh_percent),
        filter_dims=kernel_widths[0],
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        reg_vals=reg_vals)
    conv_layer1 = m.ConvolutionalLayer().like(conv_layer0)
    conv_layer1.params['num_filters'] = num_filterses[1]
    conv_layer1.params['num_inh'] = int(num_filterses[1]*num_inh_percent),
    conv_layer1.params['filter_dims'] = kernel_widths[1]
    conv_layer2 = m.ConvolutionalLayer().like(conv_layer0)
    conv_layer2.params['num_filters'] = num_filterses[2]
    conv_layer2.params['num_inh'] = int(num_filterses[2]*num_inh_percent),
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


def tcnim(num_filters, num_inh, reg_vals, kernel_width, kernel_height):
    tconv_layer = m.TemporalConvolutionalLayer(
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
    readout_layer = m.Layer(
        pos_constraint=True, # because we have inhibitory subunits on the first layer
        norm_type=m.Norm.none,
        NLtype=m.NL.softplus,
        bias=True,
        initialize_center=True,
        reg_vals={'glocalx': 0.01})
    inp_stim = m.Input(covariate='stim', input_dims=dataset.stim_dims)
    tcnim_net = m.Network(layers=[tconv_layer, readout_layer], name='TCNIM')
    output_11 = m.Output(num_neurons=dataset.NC)
    inp_stim.to(tcnim_net)
    tcnim_net.to(output_11)
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
experiment_name = 'cnim_scaffold_3_layer'
experiment_desc = 'Compare scaffold across experiments, num_filters, and copying weights'
expts = [['expt04', 'expt05'],
         ['expt04', 'expt05', 'expt06'],
         ['expt04', 'expt05', 'expt06', 'expt07']]
copy_weights = [True, False]
num_filterses = [16, 20, 24]
#kernel_heights = [5]
reg_vals = {'d2xt': 0.01, 'l1': 0.0001, 'center': 0.01, 'bcs': {'d2xt': 1}}
#modelfuncs = [cnim, cnim_scaffold, tcnim, tcnim_scaffold]
#modelstrs  = ['cnim', 'cnim_scaffold', 'tcnim', 'tcnim_scaffold']
modelfuncs = [cnim_scaffold]
modelstrs  = ['cnim_scaffold']


# grid search through the desired parameters 
grid_search = it.product(num_filterses, modelstrs)

def generate_trial(prev_trials):
    trial_idx = 0
    for num_filters, kernel_height, reg_vals, modelstr in grid_search:
        modelfunc = modelfuncs[modelstrs.index(modelstr)] # get the func at the index of the modelstr
        
        print('==========================================')
        print(num_filters, kernel_height, reg_vals, modelstr)
        
        # make the model
        num_inh = num_filters // 2
        model = modelfunc(num_filters, num_inh, reg_vals, kernel_height)
        
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
        
            # if copy_weight and len(prev_trials) > 0:
            #     prev_trial = prev_trials[-1]
            #     # skip if this is the first of the new batch of sizes
            #     if prev_trial.trial_info.trial_params['num_filters'] == num_filters:
            #         
            #         
            #         # copy the weights from the prev_trials[-1], into the current model
            #         prev_model = prev_trial.model
            #         prev_net0_layer0_weights = prev_model.NDN.networks[0].layers[0].weight
            #         prev_net0_layer1_weights = prev_model.NDN.networks[0].layers[1].weight
            # 
            #         print(':: WEIGHTS ::\n')
            #         print(prev_model.NDN.networks[0].layers[0].weight.shape, end='-->')
            #         print(model.NDN.networks[0].layers[0].weight.shape, end='\n')
            #         print(prev_model.NDN.networks[0].layers[1].weight.shape, end='-->')
            #         print(model.NDN.networks[0].layers[1].weight.shape)
            # 
            #         # copy weights of first, conv, layer
            #         model.NDN.networks[0].layers[0].weight = copy.deepcopy(prev_net0_layer0_weights)
            # 
            #         # copy weights of last, readout, layer
            #         # https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308/2
            #         # clone the weights first
            #         with torch.no_grad(): # have to make the tensors temporarily not require grad to do this
            #             curr_net0_layer1_weights = model.NDN.networks[0].layers[1].weight
            #             curr_net0_layer1_weights[:,:prev_net0_layer1_weights.shape[1]] = copy.deepcopy(prev_net0_layer1_weights)
            #             model.NDN.networks[0].layers[1].weight = curr_net0_layer1_weights
            #     else:
            #         print('NUM_FILTERS NOT THE SAME')
            # else:
            #     print(':: NO WEIGHTS YET ::')
            # # TODO: try with freezing the weights and not
        
        
            # track the specific parameters going into this trial
            trial_params = {
                #'copy_weights': copy_weights,
                'num_filters': num_filters,
                'expt': '+'.join(expt),
                'kernel_height': kernel_height, 
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
