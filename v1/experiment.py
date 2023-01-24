import os
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
                filenames,
                num_lags,
                include_MUs=False,
                time_embed=True):
    return MultiDataset(
        datadir=datadir, filenames=filenames, include_MUs=include_MUs,
        time_embed=time_embed, num_lags=num_lags)

# unpickle the pickles and make a Trial object
def _load_trial(self, name, lazy=True): # lazy=True to lazy load the dataset
    # TODO: load the trial
        

    # load model
    with open(os.path.join(self.trial_directory, 'model.pickle'), 'rb') as f:
        self.model = pickle.load(f)

    # load params
    with open(os.path.join(self.trial_directory, 'fit_params.pickle'), 'rb') as f:
        self.fit_params = pickle.load(f)

    # load LLs
    with open(os.path.join(self.trial_directory, 'LLs.pickle'), 'rb') as f:
        self.LLs = np.array(pickle.load(f))

    # load metadata
    with open(os.path.join(self.trial_directory, 'metadata.pickle'), 'rb') as f:
        self.metadata = pickle.load(f)

    # load data_loc
    with open(os.path.join(self.trial_directory, 'data_loc.pickle'), 'rb') as f:
        self.data_loc = pickle.load(f)

    # load dataset
    if not lazy:
        self.data = _load_data(**self.data_loc)
    
    return None

def load(name, folder='experiments'): # load experiment
    # loop over trials in folder and deserialize them into Trial objects
    
    
    return None


# contains data and model to fit and return log-likelihoods for
class Trial:
    def __init__(self, name, model, datadir, expnames, num_lags, fit_params, directory):
        self.name = name
        self.model = model
        self.data = None # this is used for storage in memory, but it is not saved
        self.metadata = {'name': name}
        self.fit_params = fit_params
        self.directory = directory
        self.trial_directory = os.path.join(self.directory, self.name)
        self.data_loc = {
            'datadir': datadir,
            'filenames': expnames,
            'num_lags': num_lags
        }
        self.LLs = []
    
    # TODO: make fit and eval separate?
    
    def run(self):
        # make the trial folder to save the results to
        os.mkdir(self.trial_directory)
        
        if self.data is None: # be lazy
            self.data = _load_data(**self.data_loc)
        
        # fit model
        assert self.model.NDN is not None
        self.model.NDN.fit(self.data, **self.fit_params)
        
        # eval model
        self.LLs = self.model.NDN.eval_models(self.data[self.data.val_inds], 
                                     null_adjusted=True)

        # save model
        with open(os.path.join(self.trial_directory, 'model.pickle'), 'wb') as f:
            pickle.dump(self.model, f)
            
        # save params
        with open(os.path.join(self.trial_directory, 'fit_params.pickle'), 'wb') as f:
            pickle.dump(self.fit_params, f)
        
        # save LLs
        with open(os.path.join(self.trial_directory, 'LLs.pickle'), 'wb') as f:
            pickle.dump(list(self.LLs), f)
            
        # save metadata
        with open(os.path.join(self.trial_directory, 'metadata.pickle'), 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # save data_loc
        with open(os.path.join(self.trial_directory, 'data_loc.pcikle'), 'wb') as f:
            pickle.dump(self.data_loc, f)

        # return LLs
        return self.LLs



# creates experiment for a single model architecture 
# given the desired params and data to test as different trials
class Experiment:
    def __init__(self, name, model_template, datadir, list_of_expnames, num_lags, fit_params, folder='experiments', overwrite=False):
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
        
        if not isinstance(fit_params, list):
            fit_params = [fit_params]
                    
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
        
        for di, expnames in enumerate(self.list_of_expnames):
            # create models based on the provided params
            # load the data
            data = _load_data(datadir=self.datadir,
                               filenames=expnames,
                               num_lags=self.num_lags)
            # modify the model_template.output to match the data.NC before creating
            print('Updating model neurons to:', data.NC)
            self.model_template.output.update_num_neurons(data.NC)
            
            print('Creating models')
            models = mf.create_models(self.model_template, verbose)
            print('Created', len(models), 'models')
            for mi, model in tqdm.tqdm(enumerate(models)):
                for fi, fit_params in enumerate(self.fit_params):
                    trial_name = 'm'+str(mi)+'d'+str(di)+'f'+str(fi)
                    # TODO: pass in the correct params that trial needs now
                    trial = Trial(trial_name, model, self.datadir,
                                  expnames, self.num_lags, 
                                  fit_params, directory=self.directory)
                    self.trials.append(trial)

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
    