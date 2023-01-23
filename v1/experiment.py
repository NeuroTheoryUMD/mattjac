import os
import shutil
import pickle
import json
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import model_factory as mf


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
        self.LLs = []
    
    # TODO: make fit and eval separate?
    
    def run(self):
        # make the trial folder to save the results to
        trial_directory = os.path.join(self.directory, self.name)
        os.mkdir(trial_directory)
        
        # fit model
        assert self.model.NDN is not None
        self.model.NDN.fit(self.data, **self.fit_params)
        
        # eval model
        self.LLs = self.model.NDN.eval_models(self.data[self.data.val_inds], 
                                     null_adjusted=True)
        
        # save model
        with open(os.path.join(trial_directory, 'model.pickle'), 'wb') as f:
            pickle.dump(self.model, f)
            
        # save params
        with open(os.path.join(trial_directory, 'fit_params.pickle'), 'wb') as f:
            pickle.dump(self.fit_params, f)
        
        # save LLs
        with open(os.path.join(trial_directory, 'LLs.pickle'), 'wb') as f:
            pickle.dump(list(self.LLs), f)
        
        # save data info
        data_info = {
            'covariates': self.data.covariates,
            'stim_dims': self.data.stim_dims,
            'NC': self.data.NC
        }
        with open(os.path.join(trial_directory, 'data_info.pickle'), 'wb') as f:
            pickle.dump(data_info, f)

        # return LLs
        return self.LLs



# creates experiment for a single model architecture 
# given the desired params and data to test as different trials
class Experiment:
    def __init__(self, name, model_template, data, fit_params, folder='experiments', overwrite=False):
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
        
        if not isinstance(data, list):
            data = [data]
        if not isinstance(fit_params, list):
            fit_params = [fit_params]
                    
        # experiment model_template    
        self.model_template = model_template
        # experiment params
        self.exparams = {
            'data': data,
            'fit_params': fit_params
        }
        # experiment tials
        self.trials = []
    
    def run(self, verbose=False):
        # make the trials given the params
        self.trials = []
        for di, data in enumerate(self.exparams['data']):
            # create models based on the provided params
            print('Creating models')
            models = mf.create_models(self.model_template, verbose)
            print('Created', len(models), 'models')
            for mi, model in tqdm.tqdm(enumerate(models)):
                for fi, fit_params in enumerate(self.exparams['fit_params']):
                    trial_name = 'm'+str(mi)+'d'+str(di)+'f'+str(fi)
                    self.trials.append(Trial(trial_name, model, data, fit_params, 
                                        directory=self.directory))

        print('==== Running', len(self.trials), 'trials ====')
        # run each trial
        for ti, trial in enumerate(self.trials):
            print('==== Trial', ti, '-->', trial.name, '====')
            trial.run()
            
        return self.trials
    
    
    def plot_LLs(self):
        plt.figure()
        for trial in self.trials:
            plt.plot(trial.LLs, label=trial.name)
        plt.legend()
        plt.show()

    def __getitem__(self, idx):
        return self.trials[idx]
    
    def __len__(self):
        return len(self.trials)


    # TODO: print a table (DataFrame) of comparing LLs over parameter combinations tried per dataset
    