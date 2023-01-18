import os
import shutil
import pickle
import json
import torch
import numpy as np


# utility functions

# make the var a list only if it is not already a list
def as_list(var):
    if isinstance(var, list):
        return var
    else:
        return list(var)


# keep around, but maybe delete if it is no longer useful
class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(JsonEncoder, self).default(obj)



# contains data and model to fit and return log-likelihoods for
class Trial:
    def __init__(self, name, model, data, fit_params, directory):
        self.name = name
        self.model = model
        self.data = data
        self.fit_params = fit_params
        self.directory = directory
    
    def run(self):
        # make the trial folder to save the results to
        trial_directory = os.path.join(self.directory, self.name)
        os.mkdir(trial_directory)
        
        # fit model
        self.model.fit(self.data, **self.fit_params)
        
        # eval model
        LLs = self.model.eval_models(self.data[self.data.val_inds], 
                                     null_adjusted=True)
        
        print(LLs)
        
        # save model
        with open(os.path.join(trial_directory, 'model.pickle'), 'wb') as f:
            pickle.dump(self.model, f)
            
        # save params
        with open(os.path.join(trial_directory, 'fit_params.pickle'), 'wb') as f:
            pickle.dump(self.fit_params, f)
        
        # save LLs
        with open(os.path.join(trial_directory, 'LLs.pickle'), 'wb') as f:
            pickle.dump(list(LLs), f)
        
        # save data info
        data_info = {
            'covariates': self.data.covariates,
            'stim_dims': self.data.stim_dims,
            'NC': self.data.NC
        }
        with open(os.path.join(trial_directory, 'data_info.pickle'), 'wb') as f:
            pickle.dump(data_info, f)

        # return LLs
        return LLs



# creates experiment for a single model architecture 
# given the desired params and data to test as different trials
class Experiment:
    def __init__(self, name, model, folder='experiments', overwrite=False):
        self.directory = os.path.join(folder, name)
        experiment_exists = os.path.exists(self.directory)
        if overwrite:
            if experiment_exists: # delete the previous experiment
                shutil.rmtree(self.directory, ignore_errors=True)
            os.makedirs(self.directory)
        else: # don't overwrite
            assert not experiment_exists, "experiment \""+name+"\" already exists"
            # make experiment directory
            os.makedirs(self.directory)
        
        # experiment model    
        self.model = model
        # experiment params
        self.exparams = {}
    
    def run(self):
        # make the trials given the params
        trials = []
        for di, data in enumerate(self.exparams['data']):
            # create models based on the provided params
            models = self.model.build(data)
            for mi, model in enumerate(models):
                for fi, fit_params in enumerate(self.exparams['fit_params']):
                    trial_name = 'm'+str(mi)+'d'+str(di)+'f'+str(fi)
                    trials.append(Trial(trial_name, model, data, fit_params, 
                                        directory=self.directory))
        
        # run each trial
        for trial in trials:
            trial.run()
            
        return trials

    
    # experiment configuration
    def with_data(self, *data):
        self.exparams['data'] = as_list(data)
        return self

    def with_fit_params(self, *fit_params):
        self.exparams['fit_params'] = as_list(fit_params)
        return self
