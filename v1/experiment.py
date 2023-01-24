import os
import glob
import shutil
import pickle
import json
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import model_factory as mf

from NTdatasets.generic import GenericDataset
from NTdatasets.cumming.monocular import MultiDataset


# loading functions
def _load_data(datadir,
                expnames,
                num_lags,
                include_MUs=False,
                time_embed=True):
    return MultiDataset(
        datadir=datadir, filenames=expnames, include_MUs=include_MUs,
        time_embed=time_embed, num_lags=num_lags)

# unpickle the pickles and make a Trial object
def _load_trial(trial_name, expname, folder, lazy=True): # lazy=True to lazy load the dataset
    trial_directory = os.path.join(folder, expname, trial_name)
    
    # load model
    with open(os.path.join(trial_directory, 'model.pickle'), 'rb') as f:
        model = pickle.load(f)

    # load params
    with open(os.path.join(trial_directory, 'fit_params.pickle'), 'rb') as f:
        fit_params = pickle.load(f)

    # load LLs
    with open(os.path.join(trial_directory, 'LLs.pickle'), 'rb') as f:
        LLs = np.array(pickle.load(f))

    # load data_loc
    with open(os.path.join(trial_directory, 'data_loc.pickle'), 'rb') as f:
        data_loc = pickle.load(f)
        
    trial = Trial(name=trial_name,
                  model=model,
                  datadir=data_loc['datadir'],
                  expnames=data_loc['expnames'],
                  num_lags=data_loc['num_lags'],
                  fit_params=fit_params,
                  folder=folder)
    trial.LLs = LLs
    
    return trial

def load(expname, folder='experiments'): # load experiment
    exp_dir = os.path.join(folder, expname)
    with open(os.path.join(exp_dir, 'exp_params.pickle'), 'rb') as f:
        exp_params = pickle.load(f)
    
    experiment = Experiment(expname,
                            model_template=exp_params['model_template'],
                            datadir=exp_params['datadir'],
                            list_of_expnames=exp_params['list_of_expnames'],
                            num_lags=exp_params['num_lags'],
                            fit_params=exp_params['fit_params'],
                            load=True)
    
    # loop over trials in folder and deserialize them into Trial objects
    for trial_name in os.listdir(exp_dir):
        # skip the root directory (the experiment directory)
        if os.path.basename(trial_name) == expname:
            continue
        # skip the exp_params file in the folder
        if trial_name == 'exp_params.pickle':
            continue
        trial = _load_trial(trial_name, expname, folder)
        experiment.trials.append(trial)

    return experiment


# contains data and model to fit and return log-likelihoods for
class Trial:
    def __init__(self, name, model, datadir, expnames, num_lags, fit_params, folder):
        self.name = name
        self.model = model
        self.data = None # this is used for storage in memory, but it is not saved
        self.fit_params = fit_params
        self.folder = folder
        self.trial_directory = os.path.join(self.folder, self.name)
        self.data_loc = {
            'datadir': datadir,
            'expnames': expnames,
            'num_lags': num_lags
        }
        self.LLs = []
    
    # TODO: make fit and eval separate?
    
    def run(self):
        if self.data is None: # be lazy
            self.data = _load_data(**self.data_loc)
        
        # fit model
        assert self.model.NDN is not None
        self.model.NDN.fit(self.data, **self.fit_params)
        
        # eval model
        self.LLs = self.model.NDN.eval_models(self.data[self.data.val_inds], 
                                     null_adjusted=True)

        # make the trial folder to save the results to
        os.mkdir(self.trial_directory)

        # save model
        with open(os.path.join(self.trial_directory, 'model.pickle'), 'wb') as f:
            pickle.dump(self.model, f)
            
        # save params
        with open(os.path.join(self.trial_directory, 'fit_params.pickle'), 'wb') as f:
            pickle.dump(self.fit_params, f)
        
        # save LLs
        with open(os.path.join(self.trial_directory, 'LLs.pickle'), 'wb') as f:
            pickle.dump(list(self.LLs), f)
        
        # save data_loc
        with open(os.path.join(self.trial_directory, 'data_loc.pickle'), 'wb') as f:
            pickle.dump(self.data_loc, f)

        # return LLs
        return self.LLs



# creates experiment for a single model architecture 
# given the desired params and data to test as different trials
class Experiment:
    def __init__(self, name, model_template, datadir, list_of_expnames, num_lags, fit_params, folder='experiments', overwrite=False, load=False):
        self.folder = os.path.join(folder, name)
        if not load: # TODO: this is also hacky...
            experiment_exists = os.path.exists(self.folder)
            if overwrite:
                if experiment_exists: # delete the previous experiment
                    shutil.rmtree(self.folder, ignore_errors=True)
                os.makedirs(self.folder)
            else: # don't overwrite
                assert not experiment_exists, "experiment \""+name+"\" already exists"
                # make experiment folder
                os.makedirs(self.folder)
        
        if not isinstance(fit_params, list):
            fit_params = [fit_params]
            
        # save the experiment params to the experiment folder
        exp_dir = os.path.join(folder, name)
        exp_params = {
            'model_template': model_template,
            'datadir': datadir,
            'list_of_expnames': list_of_expnames,
            'num_lags': num_lags, 
            'fit_params': fit_params
        }
        with open(os.path.join(exp_dir, 'exp_params.pickle'), 'wb') as f:
            pickle.dump(exp_params, f)
        
        
        # experiment model_template
        self.model_template = model_template
        # experiment params
        self.datadir = datadir
        self.list_of_expnames = list_of_expnames
        self.num_lags = num_lags
        self.fit_params = fit_params
        
        # experiment tials
        self.trials = []
    
    def run(self, verbose=False):
        # make the trials given the params
        self.trials = []
        
        # for each data
        for di, expnames in enumerate(self.list_of_expnames):
            print('Loading dataset for', expnames)
            # create models based on the provided params
            # load the data
            data = _load_data(datadir=self.datadir,
                              expnames=expnames,
                              num_lags=self.num_lags)

            # modify the model_template.output to match the data.NC before creating
            print('Updating model output neurons to:', data.NC)
            self.model_template.output.update_num_neurons(data.NC)
            
            models_to_try = mf.create_models(self.model_template)
            print('Trying', len(models_to_try), 'models')
            for mi, model in tqdm.tqdm(enumerate(models_to_try)):
                for fi, fit_params in enumerate(self.fit_params):
                    trial_name = 'm'+str(mi)+'d'+str(di)+'f'+str(fi)
                    # TODO: pass in the correct params that trial needs now
                    trial = Trial(trial_name, model, self.datadir,
                                  expnames, self.num_lags,
                                  fit_params, folder=self.folder)
                    print('Fitting model', mi)
                    trial.run()
                    self.trials.append(trial)
            
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
    