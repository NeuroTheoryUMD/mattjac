# collect common experiment running code here
import sys
sys.path.insert(0, '../lib')
sys.path.insert(0, '../')

import os
import torch
import copy
import pprint
import pickle
import pprint
import random
import itertools as it
import numpy as np

from enum import Enum
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
from scipy.optimize import minimize
from torch.utils.data.dataset import Subset

# NDN tools
import NDNT.utils as utils # some other utilities
from NTdatasets.cumming.monocular import MultiDataset
from NDNT.modules.layers import *
from NDNT.networks import *
from NTdatasets.generic import GenericDataset

import model as m
import experiment as exp


from copy import deepcopy

class Sample:
    def __init__(self, typ, start, end, values):
        self.typ = typ
        self.start = start
        self.end = end
        self.values = values
        

def update_model_params(model, param_keys, param_vals):
    for idx, key in enumerate(param_keys):
        ni, li, param_key, sub_key = key
        if param_key == 'reg_vals':  # handle reg_vals as a special case
            if sub_key is None:  # handle the general case
                model.networks[ni].layers[li].params[param_key] = param_vals[idx]
            else:  # handle the boundary condition special case
                model.networks[ni].layers[li].params[param_key][sub_key] = param_vals[idx]
        else:  # handle all other parameters
            model.networks[ni].layers[li].params[param_key] = param_vals[idx]

def get_model_params_with_keys(model_template, param_keys):
    # get the param values from the model template for the given param keys
    param_vals = []
    for idx, key in enumerate(param_keys):
        ni, li, param_key, sub_key = key
        if param_key == 'reg_vals':
            if sub_key is None:
                param_vals.append(model_template.networks[ni].layers[li].params[param_key])
            else:
                param_vals.append(model_template.networks[ni].layers[li].params[param_key][sub_key])
        else:
            param_vals.append(model_template.networks[ni].layers[li].params[param_key])
    return param_vals

def get_model_params(model_template):
    param_keys = []
    param_vals = []
    for ni, network in enumerate(model_template.networks):
        for li, layer in enumerate(network.layers):
            for param_key, param_val in layer.params.items():
                if param_key == 'reg_vals':
                    if isinstance(param_val, dict):
                        for sub_key, sub_val in param_val.items():
                            param_keys.append((ni, li, param_key, sub_key))
                            param_vals.append(sub_val)
                    else:
                        param_keys.append((ni, li, param_key, None))
                        param_vals.append(param_val)
                else:
                    param_keys.append((ni, li, param_key, None))
                    param_vals.append(param_val)
    return param_keys, param_vals

class HyperparameterBayesianOptimization:
    def __init__(self, model_template, init_num_samples=2, 
                 learning_rate=0.1, bayes_num_steps=10, num_initializations=1):
        self.model_template = model_template
        self.init_num_samples = init_num_samples
        self.learning_rate = learning_rate
        self.bayes_num_steps = bayes_num_steps
        self.num_initializations = num_initializations
        self.current_idx = 0
        self.models = []

        kernel = Matern()
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

        self.param_keys, self.template_param_vals = get_model_params(model_template)
        print("TEMPLATE PARAM VALS: ")
        print(self.template_param_vals)
        
        # store default values for each param
        self.default_param_vals = copy.deepcopy(self.template_param_vals)
        for idx, v in enumerate(self.template_param_vals):
            if isinstance(v, Sample):
                # add the default value
                self.default_param_vals[idx] = v.values[0]

        # Create models for each param val
        # keep track of the keys for the sample params
        self.sample_param_keys = []
        self.sample_param_vals = []
        self.sample_param_key_to_idx = {}
        
        self.num_fixed_params = None
        for idx, v in enumerate(self.template_param_vals):
            if isinstance(v, Sample):
                # add the values to the value_list_param_vals, if they exist
                # otherwise, add the default value
                # set the number of fixed params
                if self.num_fixed_params is None:
                    self.num_fixed_params = len(v.values)
                elif len(v.values) != self.num_fixed_params:
                    raise ValueError("All ValueLists must have the same length")

                # add the sample param key
                self.sample_param_keys.append(self.param_keys[idx])
                
                # add the sample param key to idx mapping
                self.sample_param_key_to_idx[self.param_keys[idx]] = idx
                
                self.sample_param_vals.append(v)

        # make the models for the fixed lists of values
        for model_idx in range(self.num_fixed_params):
            model_params = copy.deepcopy(self.template_param_vals)
            for idx, param in enumerate(self.template_param_vals):
                if isinstance(param, Sample):
                    model_params[idx] = param.values[model_idx]
            # add the model a number of times equal to the number of initializations
            for _ in range(num_initializations):
                self.add_model_with_param_vals(self.param_keys, model_params)

        # make the initial models for the Samples
        for i in range(init_num_samples):
            model_params = copy.deepcopy(self.template_param_vals)
            for idx, param in enumerate(self.template_param_vals):
                if isinstance(param, Sample):
                    sample_val = np.random.uniform(param.start, param.end, 1)[0]
                    model_params[idx] = sample_val
            # add the model a number of times equal to the number of initializations
            for _ in range(num_initializations):
                self.add_model_with_param_vals(self.param_keys, model_params)
                
        self.total_num_samples = len(self.models) + self.bayes_num_steps

    def update_models(self, prev_trials):
        # Retrieve hyperparameters and performances of previous trials
        X = np.array([get_model_params_with_keys(trial.model, self.sample_param_keys) for trial in prev_trials])
        y = np.array([np.mean(trial.LLs) for trial in prev_trials])
    
        # Normalize the hyperparameters to [0, 1] range
        # we only want to change the hyperparameters that are Samples
        bounds = [(v.start, v.end) for v in self.sample_param_vals]
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)
    
        # Fit the Gaussian Process on normalized X
        self.gp.fit(X_normalized, y)
    
        # Define the acquisition function (Expected Improvement)
        def acquisition(x):
            mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
            if sigma == 0:
                return -np.inf
            else:
                gamma = (mu - np.max(y)) / sigma
                return -1 * (mu - np.max(y)) - 0.01 * (sigma * (norm.cdf(gamma) + gamma * norm.pdf(gamma)))
    
        # Step 2: Find the hyperparameters that perform best on the surrogate
        res = minimize(acquisition, x0=np.random.uniform(size=len(bounds)), bounds=[(0, 1)] * len(bounds), method='L-BFGS-B')
    
        # The new model's hyperparameters (denormalize back to the original range)
        sample_param_vals = scaler.inverse_transform(res.x.reshape(1, -1))[0]
    
        # copy the param_keys and replace the sample param keys with the sample param vals
        param_vals = copy.deepcopy(self.default_param_vals)
        for i,k in enumerate(self.sample_param_keys):
            idx = self.sample_param_key_to_idx[k] # get the index of the sample param
            assert isinstance(self.template_param_vals[idx], Sample)
            typ = self.template_param_vals[idx].typ # get the type of the sample param
            # cast the sample param val to the correct type (int, float, etc.)
            param_vals[idx] = typ(sample_param_vals[i])
    
        # add the new model
        for _ in range(self.num_initializations):
            self.add_model_with_param_vals(self.param_keys, param_vals)

    def add_model_with_param_vals(self, param_keys, param_vals):
        model = copy.deepcopy(self.model_template)
        update_model_params(model, param_keys, param_vals)
        model.update_NDN(verbose=True)
        self.models.append(model)

    def has_next(self):
        return self.current_idx < self.total_num_samples

    def get_next(self, prev_trials):
        # update the models with the previous trials
        if len(prev_trials) >= self.init_num_samples + self.num_fixed_params:
            self.update_models(prev_trials)

        # return the next model
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
                 device="cuda:1",
                 include_MUs=False,
                 is_multiexp=False,
                 init_num_samples=2,
                 bayes_num_steps=10):
        self.batch_size = batch_size
        self.num_lags = num_lags
        self.num_initializations = num_initializations
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device(device)
        self.include_MUs = include_MUs
        self.is_multiexp = is_multiexp
        self.init_num_samples = init_num_samples
        self.bayes_num_steps = bayes_num_steps

class Runner:
    def __init__(self, 
                 experiment_name, 
                 model_template, 
                 dataset_expt, 
                 trainer_params,
                 trial_params,
                 experiment_desc='', 
                 experiment_location='../experiments/',
                 datadir='../Mdata/',
                 overwrite=False,
                 initial_trial_idx=0):
        self.experiment_name = experiment_name
        self.model_template = model_template
        self.dataset_expt = dataset_expt
        self.datadir = datadir
        self.trainer_params = trainer_params
        self.trial_params = trial_params
        self.initial_trial_idx = initial_trial_idx
        self.experiment_desc = experiment_desc
        self.experiment_location = experiment_location
        self.hyperparameter_walker = None
        self.experiment_finished = False  # or determine based on some condition

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
                # TODO: populate the HP walker when loading the experiment
                self.experiment = exp.load(self.experiment_name, experiment_location=self.experiment_location, datadir=self.datadir)
                # populate the generate_trial function, since it is not serialized
                self.experiment.generate_trial = lambda prev_trials: self.generate_trial(prev_trials)
                self.hyperparameter_walker = self.experiment.hyperparameter_walker
                # get the latest trial number
                self.initial_trial_idx = len(self.experiment.trials)
            except ValueError:
                # no existing experiment found
                # make the experiment
                self.experiment = exp.Experiment(name=self.experiment_name,
                                                 description=self.experiment_desc,
                                                 generate_trial=lambda prev_trials: self.generate_trial(prev_trials),
                                                 hyperparameter_walker=self.hyperparameter_walker,
                                                 experiment_location=self.experiment_location,
                                                 overwrite=self.overwrite_mode)

    def generate_trial(self, prev_trials):
        trial_idx = self.initial_trial_idx
        model_template = self.model_template
    
        # create the hyperparameter walker if it doesn't exist
        if self.hyperparameter_walker is None:
            self.hyperparameter_walker = HyperparameterBayesianOptimization(model_template,
                                                                            init_num_samples=self.trainer_params.init_num_samples,
                                                                            bayes_num_steps=self.trainer_params.bayes_num_steps,
                                                                            num_initializations=self.trainer_params.num_initializations)
        
        while self.hyperparameter_walker.has_next():
            # TODO: combine the runner and the experiment classes
            #       so that we can use the experiment class to run the model
            #       and this will make it easier to save and restore the state
            # save the current hyperparameter_walker state
            with open(os.path.join(self.experiment_location, self.experiment_name, 'hyperparameter_walker.pickle'), 'wb') as f:
                pickle.dump(self.hyperparameter_walker, f)
    
            model = self.hyperparameter_walker.get_next(prev_trials)
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
    
            print('Loading dataset for', self.dataset_expt)
            dataset_params = {
                'time_embed': True,
                'datadir': self.datadir,
                'filenames': self.dataset_expt,
                'include_MUs': self.trainer_params.include_MUs,
                'num_lags': self.trainer_params.num_lags
            }
            expt_dataset = MultiDataset(**dataset_params)
            expt_dataset.set_cells()  # specify which cells to use (use all if no params provided)
    
            # modify the model_template.output to match the data.NC before creating
            print('Updating model output neurons to:', expt_dataset.NC)
            model.update_num_neurons(expt_dataset.NC)
    
            trial_params = {'trial_idx': trial_idx,
                            'model_name': model.name,
                            'expt': '+'.join(self.dataset_expt)}
            
            # extend the trial_params with the self.trial_params
            trial_params.update(self.trial_params)
    
            # print the trial_params
            print('=== TRIAL', trial_idx, '===')
            pprint.pprint(trial_params)

            trial_info = exp.TrialInfo(name=model.name + str(trial_idx),
                                       description=model.name,
                                       trial_params=trial_params,
                                       dataset_params=dataset_params,
                                       dataset_class=MultiDataset,
                                       fit_params=fit_pars)
            
            trial = exp.Trial(trial_info=trial_info,
                              model=model,
                              dataset=expt_dataset,
                              eval_function=lambda model, dataset, device: model.NDN.eval_models(
                                  Subset(dataset, dataset.val_inds), null_adjusted=True))
            yield trial
            trial_idx += 1
    
        # reset the hyperparameter_walker before moving to the next experiment or model_template
        self.hyperparameter_walker = None

    def run(self):
        if self.experiment_finished:
            print('Experiment', self.experiment_name, 'is already finished. Exiting.')
            return

        self.experiment.run(self.trainer_params.device)
