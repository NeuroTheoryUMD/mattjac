# collect common experiment running code here

import sys
sys.path.insert(0, '../lib')
sys.path.insert(0, '../')

import os
import torch
import copy
import pprint
import random
import itertools as it
import numpy as np

from torch.utils.data.dataset import Subset
from enum import Enum

# NDN tools
import NDNT.utils as utils # some other utilities
from NTdatasets.cumming.monocular import MultiDataset
from NDNT.modules.layers import *
from NDNT.networks import *
from NTdatasets.generic import GenericDataset

import model as m
import experiment as exp


# common hyperparameters to use across all models
HYPER_PARAMETERS = {
    "num_filterses": None,
    "num_inh_percents": None,
    "kernel_widthses": None,
    "kernel_heightses": None,
    "num_iters": None,
    "reg_valses": None,
    "include_MUses": None,
    "is_multiexps": None,
    "batch_sizes": None,
    "num_lagses": None,
    "learning_rate": [0.01],
    "weight_decay": [0.1],
    "early_stopping_patience": [4]
}

class SearchStrategy(Enum):
    grid_search = 0
    gradient_descent = 1


class Sample:
    def __init__(self, start, end, num_samples=3):
        self.start = start
        self.end = end
        self.num_samples = num_samples


def update_model_regvals(model, reg_vals_keys, reg_vals_vals):
    for idx, key in enumerate(reg_vals_keys):
        ni, li, k, bk = key
        if bk is None:  # handle the general case
            model.networks[ni].layers[li].params['reg_vals'][k] = reg_vals_vals[idx]
        else:  # handle the boundary condition special case
            model.networks[ni].layers[li].params['reg_vals'][k][bk] = reg_vals_vals[idx]
            
def get_model_regvals(model_template):
    # extract the reg_vals (with value as Sample), and put them inside a vector and keep track of the keys
    reg_val_keys = []
    reg_val_vals = []
    for ni, network in enumerate(model_template.networks):
        for li, layer in enumerate(network.layers):
            for k, v in layer.params['reg_vals'].items():
                if isinstance(v, Sample): # handle the general case
                    reg_val_keys.append((ni, li, k, None))
                    reg_val_vals.append(v)
                elif isinstance(v, dict): # handle the boundary condition special case
                    for bk, bv in v.items():
                        if isinstance(bv, Sample):
                            reg_val_keys.append((ni, li, k, bk))
                            reg_val_vals.append(bv)
    return reg_val_keys, reg_val_vals


def get_model_regvals_vals(model, reg_val_keys):
    # get the reg_vals_vals from the model given the reg_val_keys
    reg_val_vals = []
    for idx, key in enumerate(reg_val_keys):
        ni, li, k, bk = key
        if bk is None:
            reg_val_vals.append(model.networks[ni].layers[li].params['reg_vals'][k])
        else:
            reg_val_vals.append(model.networks[ni].layers[li].params['reg_vals'][k][bk])
    return reg_val_vals


class HyperparameterSampler:
    def __init__(self, model_template, num_samples=2, learning_rate=0.01, max_num_samples=10):
        self.model_template = model_template
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.max_num_samples = max_num_samples
        self.current_idx = 0
        self.models = []
        self.reg_val_keys = []
        reg_val_vals = []
        
        # extract the reg_vals (with value as Sample), and put them inside a vector and keep track of the keys
        self.reg_val_keys, template_reg_val_vals = get_model_regvals(model_template)

        # populate the reg_val_vals with the vals from the model template samples
        for i in range(num_samples):
            reg_val_val = []
            for v in template_reg_val_vals:
                if isinstance(v, Sample):
                    # draw a random sample from the start and end values
                    reg_val_val.append(np.random.uniform(v.start, v.end, 1))
            reg_val_vals.append(reg_val_val)

        # create the initial models
        for reg_val_val in reg_val_vals:
            self.add_model_with_reg_vals(reg_val_val)

    
    def update_models(self, prev_trials):
        # get best and worst models based on the previous trials np.mean(trial.LLs))
        best_model = prev_trials[np.argmax([np.mean(trial.LLs) for trial in prev_trials])].model
        worst_model = prev_trials[np.argmin([np.mean(trial.LLs) for trial in prev_trials])].model
        
        # get the reg_val_vals from the best and worst models
        best_model_reg_val_vals = get_model_regvals_vals(best_model, self.reg_val_keys)
        worst_model_reg_val_vals = get_model_regvals_vals(worst_model, self.reg_val_keys)
        
        # populate the self.reg_val_vals with the best reg_val_vals plus a small amount in the direction of the gradient
        reg_val_val = []
        for i in range(len(self.reg_val_keys)):
            reg_val_val.append(best_model_reg_val_vals[i] + self.learning_rate * (best_model_reg_val_vals[i] - worst_model_reg_val_vals[i]))
        self.add_model_with_reg_vals(reg_val_val)
    
    def add_model_with_reg_vals(self, reg_val_vals):
        model = copy.deepcopy(self.model_template)
        update_model_regvals(model, self.reg_val_keys, reg_val_vals)
        self.models.append(model)

    def has_next(self):
        return self.current_idx < self.max_num_samples

    def get_next(self, prev_trials):
        # update the models with the previous trials
        if prev_trials is not None and len(prev_trials) >= self.num_samples:
            self.update_models(prev_trials)
        
        # return the next model
        model = self.models[self.current_idx]
        self.current_idx += 1
        return model
    
        

# The hyperparameter walker should focus on walking over reg vals and optimizer params
# the model templates can be defined in advance and the runner should not care about how the models are structured
# as there are too many different ways to structure the models,
# and it is not worth the effort to make the runner flexible enough to handle all of them.
class HyperparameterWalker:
    def __init__(self, model_template):
        self.current_idx = 0
        self.models = []

        # walk over the network, extract the reg_vals (with value as Sample), and put them inside a 
        # flat list with the network, layer, param name and the reg_val
        # walk over the network, extract the reg_vals (with value as Sample), and put them inside a 
        # flat list with the network, layer, param name and the reg_val
        reg_vals_keys = []
        reg_vals_vals = []
        for ni, network in enumerate(model_template.networks):
            for li,layer in enumerate(network.layers):
                if 'reg_vals' in layer.params:
                    for k,v in layer.params['reg_vals'].items():
                        if k == 'bcs': # handle the boundary condition special case
                            for bk,bv in v.items():
                                reg_vals_keys.append((ni, li, k, bk))
                                if type(bv) is Sample:
                                    reg_vals_vals.append(np.linspace(bv.start, bv.end, bv.num_samples).tolist())
                                else:
                                    reg_vals_vals.append([bv]) # just copy the value over as a list
                        elif type(v) is Sample:
                            # set the reg_vals to be a list of samples of different orders of magnitude between the start and end
                            reg_vals_keys.append((ni, li, k, None))
                            reg_vals_vals.append(np.linspace(v.start, v.end, v.num_samples).tolist())
                        else:
                            reg_vals_keys.append((ni, li, k, None))
                            reg_vals_vals.append([v]) # just copy the value over as a list
        
        # get the combinations of reg_vals
        grid_search = it.product(*reg_vals_vals)

        # now, we have a dict of reg_vals, with each value being a list of reg_vals to sample from
        # we want to create a model for each combination of reg_vals
        for reg_vals in grid_search:
            model = copy.deepcopy(model_template)
            # get the network, layer and keys for each reg_val
            for i, reg_val in enumerate(reg_vals):
                ni, li, k, bk = reg_vals_keys[i]
                if bk is None:
                    model.networks[ni].layers[li].params['reg_vals'][k] = reg_val
                else:
                    model.networks[ni].layers[li].params['reg_vals'][k][bk] = reg_val
            model.update_NDN() # update the NDN with the new reg_vals
            self.models.append(model)

        print('Total number of models:', len(self.models))

    def has_next(self):
        return self.current_idx < len(self.models)

    def get_next(self, prev_trials):
        model = self.models[self.current_idx]
        self.current_idx += 1
        return model


class TrainerParams:
    def __init__(self,
                 batch_size=2000,
                 num_lags=10,
                 num_initializations=1,
                 max_epochs=None,
                 learning_rate=0.01,
                 weight_decay=0.1,
                 early_stopping_patience=4,
                 device = torch.device("cuda:1"),
                 include_MUs=False,
                 is_multiexp=False):
        self.batch_size = batch_size
        self.num_lags = num_lags
        self.num_initializations = num_initializations
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.include_MUs = include_MUs
        self.is_multiexp = is_multiexp


class Runner:
    def __init__(self,
                 experiment_name,
                 dataset_expts,
                 model_templates,
                 search_strategy=SearchStrategy.grid_search,
                 trainer_params=TrainerParams(),
                 experiment_desc='',
                 experiment_location='../experiments/',
                 datadir='../Mdata/',
                 overwrite=False):
        self.experiment_name = experiment_name
        self.dataset_expts = dataset_expts
        self.model_templates = model_templates
        self.search_strategy = search_strategy
        self.trainer_params = trainer_params
        self.experiment_desc = experiment_desc
        self.experiment_location = experiment_location
        self.datadir = datadir

        # default to this
        self.hyperparameter_walker = None
        self.initial_trial_idx = 0
        self.expt_idx = 0
        self.model_idx = 0
        self.experiment_finished = False

        # check if the experiment folder exists
        if overwrite:
            self.overwrite_mode = exp.Overwrite.overwrite
        else:
            self.overwrite_mode = exp.Overwrite.append
            # check if the experiment is finished (e.g. there is a 'finished' file in the directory)
            if os.path.exists(os.path.join(self.experiment_location, self.experiment_name, 'finished')):
                self.experiment_finished = True
                return
            # check the experiment folder for existing trials, and get the latest trial number
            try:
                existing_experiment = exp.load(self.experiment_name, experiment_location=self.experiment_location, datadir=self.datadir)
                latest_trial = existing_experiment.trials[-1]
                # set the current trial index to the last trial index of the existing experiment + 1
                self.initial_trial_idx = latest_trial.trial_info.trial_params['trial_idx'] + 1
                self.hyperparameter_walker = latest_trial.hyperparameter_walker
                # increment the current_idx of the hyperparameter walker by 1
                self.hyperparameter_walker.current_idx += 1
                self.expt_idx = latest_trial.trial_info.expt_idx
                self.model_idx = latest_trial.trial_info.model_idx
            except ValueError:
                # no existing experiment found
                pass

        # make the experiment
        self.experiment = exp.Experiment(name=self.experiment_name,
                                    description=self.experiment_desc,
                                    generate_trial=lambda prev_trials: self.generate_trial(prev_trials),
                                    experiment_location=self.experiment_location,
                                    overwrite=self.overwrite_mode)


    def generate_trial(self, prev_trials):
        trial_idx = self.initial_trial_idx
        expt_idx = self.expt_idx
        model_idx = self.model_idx
        for model_template in self.model_templates[model_idx:]:
            for expt in self.dataset_expts[expt_idx:]:
                if self.hyperparameter_walker is None:
                    if self.search_strategy == SearchStrategy.grid_search:
                        hyperparameter_walker = HyperparameterWalker(model_template)
                    elif self.search_strategy == SearchStrategy.gradient_descent:
                        hyperparameter_walker = HyperparameterSampler(model_template)
                else:
                    hyperparameter_walker = self.hyperparameter_walker
                    print('Using existing hyperparameter walker, Models left:', len(hyperparameter_walker.models) - hyperparameter_walker.current_idx)
                while hyperparameter_walker.has_next():
                    model = hyperparameter_walker.get_next(prev_trials)
                    fit_pars = utils.create_optimizer_params(
                        optimizer_type='AdamW',
                        num_workers=0,
                        optimize_graph=False,
                        batch_size=self.trainer_params.batch_size,
                        learning_rate=self.trainer_params.learning_rate,
                        early_stopping_patience=self.trainer_params.early_stopping_patience,
                        weight_decay=self.trainer_params.weight_decay)
                    fit_pars['is_multiexp'] = self.trainer_params.is_multiexp
                    fit_pars['device'] = self.trainer_params.device
                    fit_pars['verbose'] = True
                    if self.trainer_params.max_epochs is not None:
                        fit_pars['max_epochs'] = self.trainer_params.max_epochs

                    # skip until trial_idx
                    if trial_idx < self.initial_trial_idx:
                        print('skipping trial', trial_idx)
                        trial_idx += 1
                        continue

                    print('Loading dataset for', expt)
                    dataset_params = {
                        'time_embed': True,
                        'datadir': self.datadir,
                        'filenames': expt,
                        'include_MUs': self.trainer_params.include_MUs,
                        'num_lags': self.trainer_params.num_lags
                    }
                    expt_dataset = MultiDataset(**dataset_params)
                    expt_dataset.set_cells() # specify which cells to use (use all if no params provided)

                    # modify the model_template.output to match the data.NC before creating
                    print('Updating model output neurons to:', expt_dataset.NC)
                    model.update_num_neurons(expt_dataset.NC)

                    trial_params = {'trial_idx': trial_idx,
                                    'model_name': model.name,
                                    'expt': '+'.join(expt)}

                    # print the trial_params
                    print('=== TRIAL', trial_idx, '===')
                    pprint.pprint(trial_params)

                    trial_info = exp.TrialInfo(name=model.name+str(trial_idx),
                                               description=model.name,
                                               trial_params=trial_params,
                                               dataset_params=dataset_params,
                                               dataset_class=MultiDataset,
                                               fit_params=fit_pars,
                                               expt_idx=expt_idx,
                                               model_idx=model_idx)

                    trial = exp.Trial(trial_info=trial_info,
                                      model=model,
                                      dataset=expt_dataset,
                                      hyperparameter_walker=hyperparameter_walker,
                                      eval_function=lambda model,dataset,device: model.NDN.eval_models(Subset(dataset, dataset.val_inds), null_adjusted=True))
                    yield trial
                    trial_idx += 1
                expt_idx += 1
            model_idx += 1


    def run(self):
        if self.experiment_finished:
            print('Experiment', self.experiment_name, 'is already finished. Exiting.')
            return
        
        self.experiment.run(self.trainer_params.device)
