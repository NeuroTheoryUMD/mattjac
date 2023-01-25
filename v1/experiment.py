import os
import shutil
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import model as mod
from enum import Enum


# loading functions
# unpickle the pickles and make a Trial object
def _load_trial(trial_name, experiment_folder, lazy=True): # lazy=True to lazy load the dataset
    trial_directory = os.path.join(experiment_folder, trial_name)
    
    # load model
    with open(os.path.join(trial_directory, 'model.pickle'), 'rb') as f:
        model = pickle.load(f)

    # load LLs
    with open(os.path.join(trial_directory, 'LLs.pickle'), 'rb') as f:
        LLs = np.array(pickle.load(f))

    # load trial_info
    with open(os.path.join(trial_directory, 'trial_info.pickle'), 'rb') as f:
        trial_info = pickle.load(f)
        
    # load the dataset
    dataset = None
    if not lazy:
        dataset = trial_info.dataset_class(**trial_info.dataset_params)
        
    trial = Trial(trial_info=trial_info,
                  model=model,
                  dataset=dataset)
    trial.LLs = LLs
    return trial

def load(expname, experiment_location='experiments', lazy=False): # load experiment
    experiment_folder = os.path.join(experiment_location, expname)
    with open(os.path.join(experiment_folder, 'exp_params.pickle'), 'rb') as f:
        exp_params = pickle.load(f)
    
    experiment = Experiment(name=expname,
                            description=exp_params['description'],
                            generate_trial=None,
                            experiment_location=experiment_location,
                            overwrite=Overwrite.overwrite)
    
    # loop over trials in folder and deserialize them into Trial objects
    for trial_name in os.listdir(experiment_folder):
        # skip the root directory (the experiment directory)
        if os.path.basename(trial_name) == expname:
            continue
        # skip the exp_params file in the folder
        if trial_name == 'exp_params.pickle':
            continue
        trial = _load_trial(trial_name, experiment_folder, lazy=lazy)
        experiment.trials.append(trial)

    return experiment


# contains metadata about the trial,
# so we can keep this and the model and data separate
class TrialInfo:
    def __init__(self, 
                 name:str,
                 description:str,
                 dataset_params:dict,
                 dataset_class,
                 fit_params:dict,
                 eval_params:dict):
        self.name = name
        self.description = description
        self.dataset_params = dataset_params
        self.dataset_class = dataset_class
        self.fit_params = fit_params
        self.eval_params = eval_params


# contains data and model to fit and return log-likelihoods for
class Trial:
    def __init__(self, 
                 trial_info:TrialInfo,
                 model:mod.Model,
                 dataset):
        self.trial_info = trial_info
        self.model = model
        self.dataset = dataset # this is used for storage in memory, but it is not saved
        self.LLs = []
    
    def run(self, experiment_folder):
        trial_directory = os.path.join(experiment_folder, self.trial_info.name)
        
        # fit model
        assert self.model.NDN is not None
        self.model.NDN.fit(self.dataset, **self.trial_info.fit_params)
        
        # eval model
        self.LLs = self.model.NDN.eval_models(self.dataset[self.dataset.val_inds], 
                                              **self.trial_info.eval_params)

        # make the trial folder to save the results to
        os.mkdir(trial_directory)

        # save model
        with open(os.path.join(trial_directory, 'model.pickle'), 'wb') as f:
            pickle.dump(self.model, f)
            
        # save trial_info
        with open(os.path.join(trial_directory, 'trial_info.pickle'), 'wb') as f:
            pickle.dump(self.trial_info, f)
        
        # save LLs
        with open(os.path.join(trial_directory, 'LLs.pickle'), 'wb') as f:
            pickle.dump(list(self.LLs), f)


# creates experiment for a single model architecture 
# given the desired params and data to test as different trials
class Overwrite(Enum):
    none = 0
    append = 1
    overwrite = 2
    
class Experiment:
    def __init__(self, 
                 name:str,
                 description:str,
                 generate_trial,
                 experiment_location:str,
                 overwrite:Overwrite=Overwrite.none):
        self.name = name
        self.description = description
        self.experiment_folder = os.path.join(experiment_location, name)
        self.generate_trial = generate_trial
        self.overwrite = overwrite
        
        # if we don't specify how to overwrite
        if self.overwrite == Overwrite.none:
            assert not os.path.exists(self.experiment_folder), 'experiment_folder already exists and overwrite was not specified'
        
        # experiment trials
        self.trials = []
    
    def run(self):
        experiment_exists = os.path.exists(self.experiment_folder)
        # make the dirs if it doesn't currently exist
        if not experiment_exists: # make new directory if it doesn't exist
            os.makedirs(self.experiment_folder) # make the new experiment
        else: # overwrite the previous directory if asked to
            if self.overwrite == Overwrite.overwrite:
                shutil.rmtree(self.experiment_folder, ignore_errors=True)
                os.makedirs(self.experiment_folder) # make the new experiment
            # else: it will automatically append
        
        # save the experiment params to the experiment folder
        exp_params = {
            'name': self.name,
            'description': self.description,
            'experiment_folder': self.experiment_folder
        }
        with open(os.path.join(self.experiment_folder, 'exp_params.pickle'), 'wb') as f:
            pickle.dump(exp_params, f)
        
        # for each Trial
        # pass in the previous trials into the next one
        for trial in self.generate_trial(self.trials):
            trial.run(self.experiment_folder)
            self.trials.append(trial)
    
    
    def plot_LLs(self, figsize=(15,5)):
        # get maximum num_neurons for the experiment
        max_num_neurons = 0
        for trial in self.trials:
            if trial.model.output.num_neurons > max_num_neurons:
                max_num_neurons = trial.model.output.num_neurons
        
        df = pd.DataFrame({
            'Neuron': ['N'+str(n) for n in range(max_num_neurons)],
        })
        for trial in self.trials:
            df[trial.name] = np.concatenate((trial.LLs, np.zeros(max_num_neurons-len(trial.LLs), dtype=np.float32)))
        fig, ax1 = plt.subplots(figsize=figsize)
        tidy = df.melt(id_vars='Neuron').rename(columns=str.title)
        ax = sns.barplot(x='Neuron', y='Value', hue='Variable', data=tidy, ax=ax1)
        ax.set(xlabel='Neuron', ylabel='Log-Likelihood')
        plt.legend(title='Model')
        sns.despine(fig)
        
    def __getitem__(self, trial_name):
        for trial in self.trials:
            if trial.name == trial_name:
                return trial
        return None
    
    def __len__(self):
        return len(self.trials)


    # TODO: print a table (DataFrame) of comparing LLs over parameter combinations tried per dataset
    