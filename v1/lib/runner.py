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

from enum import Enum
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
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


# common hyperparameters to use across all models
HYPER_PARAMETERS = {
class SearchStrategy(Enum):
    grid = 0
    bayes = 1


class ValueList:
    def __init__(self, values):
        self.values = values

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
                # if the value is a dict, then it is a boundary condition
                if isinstance(v, dict):
                    for bk, bv in v.items():
                        reg_val_keys.append((ni, li, k, bk))
                        reg_val_vals.append(bv)
                else:
                    # add the key and value to the list
                    reg_val_keys.append((ni, li, k, None))
                    reg_val_vals.append(v)

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
    return np.array(reg_val_vals).squeeze()


class HyperparameterBayesianOptimization:
    def __init__(self, model_template, init_num_samples=2, learning_rate=0.1, max_num_samples=10):
        self.model_template = model_template
        self.init_num_samples = init_num_samples
        self.learning_rate = learning_rate
        self.max_num_samples = max_num_samples
        self.len_prev_trials = -1
        self.current_idx = 0
        self.models = []
        self.reg_val_keys = []
        reg_val_vals = []

        # Step 1: Build a surrogate probability model of the objective function
        kernel = Matern()
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

        # extract the reg_vals (with value as Sample), and put them inside a vector and keep track of the keys
        self.reg_val_keys, self.template_reg_val_vals = get_model_regvals(model_template)

        # populate the reg_val_vals with the vals from the model template samples
        for i in range(init_num_samples):
            reg_val_val = []
            for v in self.template_reg_val_vals:
                if isinstance(v, Sample):
                    # draw a random sample from the start and end values
                    reg_val_val.append(np.random.uniform(v.start, v.end, 1))
            reg_val_vals.append(reg_val_val)

        # now handle the ValueList case
        num_fixed_vals = None
        for v in self.template_reg_val_vals:
            if isinstance(v, ValueList):
                if num_fixed_vals is None:
                    num_fixed_vals = len(v.values)
                else:
                    assert num_fixed_vals == len(v.values), "All ValueList must have the same number of values"
                # add all the values in the ValueList
                reg_val_vals.extend(v.values)
            else:
                reg_val_vals.append(v)

        # create the initial models
        for reg_val_val in reg_val_vals:
            self.add_model_with_reg_vals(reg_val_val)

    def update_models(self, prev_trials):
        # Retrieve hyperparameters and performances of previous trials
        X = np.array([get_model_regvals_vals(trial.model, self.reg_val_keys) for trial in prev_trials])
        y = np.array([np.mean(trial.LLs) for trial in prev_trials])

        self.gp.fit(X, y)

        # Define the acquisition function (Expected Improvement)
        def acquisition(x):
            mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
            if sigma == 0:
                return -np.inf
            else:
                gamma = (mu - np.max(y)) / sigma
                return -1 * (mu - np.max(y)) - 0.01 * (sigma * (norm.cdf(gamma) + gamma * norm.pdf(gamma)))

        # Step 2: Find the hyperparameters that perform best on the surrogate
        bounds = [(v.start, v.end) for v in self.template_reg_val_vals]
        res = minimize(acquisition, x0=np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds]), bounds=bounds, method='L-BFGS-B')

        # Step 3: Apply these hyperparameters to the true objective function is done outside this function
        # Step 4: Update the surrogate model incorporating the new results
        # This is done automatically as we refit the Gaussian Process with new data at each iteration

        # The new model's hyperparameters
        reg_val_val = res.x

        # add the new model
        self.add_model_with_reg_vals(reg_val_val)

    def add_model_with_reg_vals(self, reg_val_vals):
        model = copy.deepcopy(self.model_template)
        update_model_regvals(model, self.reg_val_keys, reg_val_vals)
        self.models.append(model)

    def has_next(self):
        return self.current_idx < self.max_num_samples

    def get_next(self, prev_trials):
        self.len_prev_trials += 1
        print('length of prev_trials: ', self.len_prev_trials)
        # update the models with the previous trials
        if self.len_prev_trials >= self.init_num_samples:
            self.update_models(prev_trials)

        # return the next model
        print('models length: ', len(self.models), ' current_idx: ', self.current_idx)
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
                 bayes_init_num_samples=2,
                 bayes_max_num_samples=10):
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
        self.bayes_init_num_samples = bayes_init_num_samples
        self.bayes_max_num_samples = bayes_max_num_samples


class Runner:
    def __init__(self,
                 experiment_name,
                 dataset_expts,
                 model_templates,
                 search_strategy=SearchStrategy.grid,
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
                self.experiment = exp.load(self.experiment_name, experiment_location=self.experiment_location, datadir=self.datadir)
                # populate the generate_trial function, since it is not serialized
                self.experiment.generate_trial = lambda prev_trials: self.generate_trial(prev_trials)
                latest_trial = self.experiment.trials[-1]
                # set the current trial index to the last trial index of the existing experiment + 1
                self.initial_trial_idx = latest_trial.trial_info.trial_params['trial_idx'] + 1
                self.hyperparameter_walker = latest_trial.hyperparameter_walker
                self.expt_idx = latest_trial.trial_info.expt_idx
                self.model_idx = latest_trial.trial_info.model_idx
            except ValueError:
                # no existing experiment found
                # make the experiment
                self.experiment = exp.Experiment(name=self.experiment_name,
                                                 description=self.experiment_desc,
                                                 generate_trial=lambda prev_trials: self.generate_trial(prev_trials),
                                                 experiment_location=self.experiment_location,
                                                 overwrite=self.overwrite_mode)

    def generate_trial(self, prev_trials):
        trial_idx = self.initial_trial_idx
        expt_idx = self.expt_idx # NOTE: this will break recovery if we want to try different numbers of experiments
        model_idx = self.model_idx
        for model_template in self.model_templates[model_idx:]:
            print('Model:', model_template.name, self.hyperparameter_walker)
            expt_idx = 0 # reset the expt_idx
            for expt in self.dataset_expts[expt_idx:]:
                print('HERE1')
                # TODO: this is a terrible hack,
                #       move this into the experiment and refactor the relationship between experiment and runner
                model_expt_prev_trials = []
                init_prev_trial_len = len(prev_trials)
    
                if self.hyperparameter_walker is None:
                    print('HERE')
                    if self.search_strategy == SearchStrategy.grid:
                        hyperparameter_walker = HyperparameterGridSearch(model_template)
                    elif self.search_strategy == SearchStrategy.bayes:
                        hyperparameter_walker = HyperparameterBayesianOptimization(model_template,
                                                                                   init_num_samples=self.trainer_params.bayes_init_num_samples,
                                                                                   max_num_samples=self.trainer_params.bayes_max_num_samples)
                else:
                    hyperparameter_walker = self.hyperparameter_walker
                    if self.hyperparameter_walker.len_prev_trials > 0:
                        model_expt_prev_trials.extend(prev_trials[:-self.hyperparameter_walker.len_prev_trials])
                    print('Using existing hyperparameter walker, Models left:',
                          len(hyperparameter_walker.models) - hyperparameter_walker.current_idx)
                while hyperparameter_walker.has_next():
                    if len(prev_trials) > init_prev_trial_len:
                        # if there are new trials, add them to the model_expt_prev_trials
                        model_expt_prev_trials.append(prev_trials[-1])
    
                    model = hyperparameter_walker.get_next(model_expt_prev_trials)
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
                # reset the hyperparameter_walker before moving to the next experiment or model_template
                self.hyperparameter_walker = None
                expt_idx += 1
            model_idx += 1
    
    
    def run(self):
        if self.experiment_finished:
            print('Experiment', self.experiment_name, 'is already finished. Exiting.')
            return
    
        self.experiment.run(self.trainer_params.device)
