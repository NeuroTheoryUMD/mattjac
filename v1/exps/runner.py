# collect common experiment running code here

import sys
sys.path.insert(0, '../lib')
sys.path.insert(0, '../')

import torch
import copy
import itertools as it
import numpy as np

from torch.utils.data.dataset import Subset

# NDN tools
import NDNT.utils as utils # some other utilities
from NTdatasets.cumming.monocular import MultiDataset
from NDNT.modules.layers import *
from NDNT.networks import *
from NTdatasets.generic import GenericDataset

import experiment as exp


class HyperparameterWalker:
    def __init__(self,
                 num_filterses,
                 num_inh_percents,
                 kernel_widthses,
                 kernel_heightses,
                 reg_valses,
                 include_MUses,
                 is_multiexps,
                 batch_sizes):
        # initialize the hyperparameters
        # for now, just grid search through the space and come up with something better after
        self.params = list(it.product(num_filterses,
                                 num_inh_percents,
                                 kernel_widthses,
                                 kernel_heightses,
                                 reg_valses,
                                 include_MUses,
                                 is_multiexps,
                                 batch_sizes))
        print('num trials', len(list(self.params)))
    
    def walk(self):
        for p in self.params:
            yield p


class Runner:
    def __init__(self,
                 experiment_name,
                 dataset_expts,
                 num_lags,
                 model_templates,
                 hyperparameter_walker,
                 experiment_desc='',
                 experiment_location='../experiments/',
                 num_initializations=1,
                 include_MUs=False,
                 batch_size=2000,
                 max_epochs=None,
                 device = torch.device("cuda:1"),
                 datadir='../Mdata/'):
        self.experiment_name = experiment_name
        self.experiment_desc = experiment_desc
        self.experiment_location = experiment_location
        self.dataset_expts = dataset_expts
        self.num_lags = num_lags
        self.model_templates = model_templates
        self.hyperparameter_walker = hyperparameter_walker
        self.num_initializations = num_initializations
        self.include_MUs = include_MUs
        self.batch_size = batch_size
        self.device = device
        self.datadir = datadir
        
        self.fit_pars = utils.create_optimizer_params(
            optimizer_type='AdamW',
            batch_size=2000,
            num_workers=0,
            learning_rate=0.01,
            early_stopping_patience=4,
            optimize_graph=False,
            weight_decay = 0.1)
        self.fit_pars['device'] = device
        self.fit_pars['verbose'] = True
        if max_epochs is not None:
            self.fit_pars['max_epochs'] = max_epochs
            
        # check the experiment folder for existing trials, and get the latest trial number
        try:
            existing_experiment = exp.load(self.experiment_name, experiment_location=self.experiment_location, datadir=self.datadir)
            # set the current trial index to the last trial index of the existing experiment + 1
            self.trial_idx = existing_experiment.trials[-1].trial_info.trial_params['trial_idx'] + 1
        except ValueError:
            self.trial_idx = 0


    def generate_trial(self, prev_trials):
        trial_idx = self.trial_idx
        for model_template in self.model_templates:
            for i, (num_filters, num_inh_percent, kernel_widths, kernel_heights, reg_vals, include_MUs, is_multiexp, batch_size) in enumerate(self.hyperparameter_walker.walk()):
                # skip until i >= trial_idx
                if i < trial_idx:
                    print('skipping trial', i)
                    continue
                
                print(num_filters, num_inh_percent, kernel_widths, kernel_heights, reg_vals, include_MUs, is_multiexp, batch_size)
                
                # get the model from the model_template
                model = model_template(num_filters, num_inh_percent, reg_vals, kernel_widths, kernel_heights)
                
                self.fit_pars['is_multiexp'] = is_multiexp
                self.fit_pars['batch_size'] = batch_size
                
                for expt in self.dataset_expts:
                    for run in range(self.num_initializations):
                        print('Loading dataset for', expt)
                        dataset_params = {
                            'datadir': self.datadir,
                            'filenames': expt,
                            'include_MUs': include_MUs,
                            'time_embed': True,
                            'num_lags': self.num_lags
                        }
                        expt_dataset = MultiDataset(**dataset_params)
                        expt_dataset.set_cells() # specify which cells to use (use all if no params provided)
        
                        # update model based on the provided params
                        # modify the model_template.output to match the data.NC before creating
                        print('Updating model output neurons to:', expt_dataset.NC)
                        model.update_num_neurons(expt_dataset.NC)
        
                        # track the specific parameters going into this trial
                        trial_params = {
                            'null_adjusted_LL': True,
                            'num_filters': ','.join([str(a) for a in num_filters]),
                            'num_inh_percent': num_inh_percent,
                            'expt': '+'.join(expt),
                            'kernel_widths': ','.join([str(a) for a in kernel_widths]),
                            'kernel_heights': ','.join([str(a) for a in kernel_heights]),
                            'include_MUs': include_MUs,
                            'is_multiexp': is_multiexp,
                            'batch_size': batch_size,
                            'trial_idx': trial_idx,
                            'run': run
                        }
                        # add individual reg_vals to the trial_params
                        for k,v in reg_vals.items():
                            trial_params[k] = v
                        
                        print('model name:', model.name)
        
                        trial_info = exp.TrialInfo(name=model.name+str(trial_idx)+'.'+str(run),
                                                   description=model.name,
                                                   trial_params=trial_params,
                                                   dataset_params=dataset_params,
                                                   dataset_class=MultiDataset,
                                                   fit_params=self.fit_pars)
        
                        trial = exp.Trial(trial_info=trial_info,
                                          model=model,
                                          dataset=expt_dataset,
                                          eval_function=lambda model,dataset,device: model.NDN.eval_models(Subset(dataset, dataset.val_inds), null_adjusted=True))
                        yield trial
                    trial_idx += 1
    
    
    def run(self):
        # run the experiment
        experiment = exp.Experiment(name=self.experiment_name,
                                    description=self.experiment_desc,
                                    generate_trial=lambda prev_trials: self.generate_trial(prev_trials),
                                    experiment_location=self.experiment_location,
                                    overwrite=exp.Overwrite.append)
        experiment.run(self.device)
