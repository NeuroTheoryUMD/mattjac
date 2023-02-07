import torch
import sys
import copy

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
    num_lags=num_lags)

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
    for num_filters in [8, 12, 16]:
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
        
        for copy_weights in [True, False]:
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
                # TODO: try with freezing the weights and not


                # track the specific parameters going into this trial
                trial_params = {
                    'copy_weights': copy_weights,
                    'num_filters': num_filters,
                    'expt': '+'.join(expt)
                }

                trial_info = exp.TrialInfo(name='CNIM'+str(trial_idx),
                                           description='CNIM train on the dataset, and copy the weights over from the previous trial',
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
experiment = exp.Experiment(name='exp_CNIM_copyweights',
                            description='Convolutional Nonlinear Input Model, copying weights from previous datasets over',
                            generate_trial=generate_trial,
                            experiment_location='experiments',
                            overwrite=exp.Overwrite.overwrite)
experiment.run(device)
