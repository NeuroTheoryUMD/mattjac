import sys
sys.path.insert(0, '../') # to have access to NTdatasets

import os
import glob
import shutil
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import model as mod
from enum import Enum
# https://docs.python.org/3/library/typing.html

# to be able to load the Tensorboard events to see the loss
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# NTdataset
from NTdatasets.generic import GenericDataset


# enums
# creates experiment for a single model architecture 
# given the desired params and data to test as different trials
class Overwrite(Enum):
    none = 0
    append = 1
    overwrite = 2
    
class Loss(Enum):
    loss = 'Loss/Loss'
    train = 'Loss/Train'
    reg = 'Loss/Reg'
    train_epoch = 'Loss/Train (Epoch)'
    val_epoch = 'Loss/Validation (Epoch)'


# loading functions
# unpickle the pickles and make a Trial object
def _load_trial(trial_name, experiment_folder, datadir=None, lazy=True): # lazy=True to lazy load the dataset
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
    
    # set the datadir in the trial_info.dataset_params
    if datadir is not None:
        trial_info.dataset_params['datadir'] = datadir
    
    # load the dataset
    dataset = None
    if not lazy:
        dataset = trial_info.dataset_class(**trial_info.dataset_params)
        
    trial = Trial(trial_info=trial_info,
                  model=model,
                  dataset=dataset,
                  eval_function=None)
    # set the results properties so we have access to them
    trial.LLs = LLs
    trial.trial_directory = trial_directory
    trial.ckpts_directory = os.path.join(trial_directory, 'checkpoints')
    return trial

def load(expname, experiment_location, datadir=None, lazy=True): # load experiment    
    experiment_folder = os.path.join(experiment_location, expname)
    with open(os.path.join(experiment_folder, 'exp_params.pickle'), 'rb') as f:
        exp_params = pickle.load(f)
    
    experiment = Experiment(name=expname,
                            description=exp_params['description'],
                            generate_trial=None,
                            experiment_location=experiment_location,
                            datadir=datadir,
                            overwrite=Overwrite.overwrite)
    
    # loop over trials in folder and deserialize them into Trial objects
    trials = []
    for trial_name in os.listdir(experiment_folder):
        # skip the root directory (the experiment directory)
        if os.path.basename(trial_name) == expname:
            continue
        # skip the exp_params file in the folder
        if trial_name == 'exp_params.pickle':
            continue
        # skip hidden folders
        if trial_name.startswith('.'):
            continue
        # finally, load the trial folder
        try:
            trial = _load_trial(trial_name=trial_name, 
                                experiment_folder=experiment_folder, 
                                datadir=datadir, 
                                lazy=lazy)
        except Exception as e:
            print("Error loading trial", trial_name, "...skipping")
            print(e)
            continue
        trials.append(trial)
    experiment.trials = trials # make sure to use the setter

    return experiment


# contains metadata about the trial,
# so we can keep this and the model and data separate
class TrialInfo:
    def __init__(self, 
                 name:str,
                 description:str,
                 trial_params:dict,
                 dataset_params:dict,
                 dataset_class,
                 fit_params:dict):
        self.name = name
        self.description = description
        self.trial_params = trial_params
        self.dataset_params = dataset_params
        self.dataset_class = dataset_class
        self.fit_params = fit_params


# contains data and model to fit and return log-likelihoods for
class Trial:
    def __init__(self, 
                 trial_info:TrialInfo,
                 model:mod.Model,
                 dataset,
                 eval_function):
        self.name = trial_info.name
        self.description = trial_info.description
        self.trial_params = trial_info.trial_params
        self.trial_info = trial_info
        self.model = model
        self.eval_function = eval_function # not serialized
        self._dataset = dataset # not serialized
        
        # these are initially null until the Trial is trained or loaded
        self.trial_directory = None
        self.ckpts_directory = None
        self.LLs = []
        self._losses = None # lazy loaded

    def _get_dataset(self):
        if self._dataset is None:
            print('lazy loading dataset')
            self._dataset = self.trial_info.dataset_class(**self.trial_info.dataset_params)
        return self._dataset

    def losses(self, loss_type):
        # lazy load losses as well
        if self._losses is None:
            print('lazy loading losses')
            # look for a file prefixed with "events.out" to get the filename
            # 'experiments/exp_NIM_test/NIM_expt04/checkpoints/
            # M011_NN/version1/events.out.tfevents.1674778321.beast.155528.0'
            # we need to ignore the versioning scheme in the checkpoints directory
            # walk through the hierarchy until we get to the events file
            subfolders = []
            for root,dirs,files in os.walk(self.ckpts_directory, topdown=True):
                if len(dirs) > 0:
                    subfolders.append(dirs[0])
            assert not len(subfolders) < 1, 'Trial ['+self.trial_info.name+'] no subfolders found in the checkpoints directory'
            assert not len(subfolders) > 2, 'Trial ['+self.trial_info.name+'] more than 2 subfolders found in the checkpoints directory'
            events_directory = os.path.join(self.ckpts_directory, *subfolders)
            event_filenames = glob.glob(os.path.join(events_directory, 'events.out.*'))
            assert not len(event_filenames) < 1, 'Trial ['+self.trial_info.name+'] no event file found in the checkpoints directory'
            assert not len(event_filenames) > 1, 'Trial ['+self.trial_info.name+'] more than 1 event file found in the checkpoints directory'
            event_filename = event_filenames[0]
            event_acc = EventAccumulator(event_filename)
            event_acc.Reload()
            # Show all tags in the log file -- print(event_acc.Tags())
            # get wall clock, number of steps and value for a scalar 'Accuracy'
            loss_w_times, loss_step_nums, loss_losses = zip(*event_acc.Scalars(Loss.loss.value))
            train_w_times, train_step_nums, train_losses = zip(*event_acc.Scalars(Loss.train.value))
            reg_w_times, reg_step_nums, reg_losses= zip(*event_acc.Scalars(Loss.reg.value))
            train_epoch_w_times, train_epoch_step_nums, train_epoch_losses = zip(*event_acc.Scalars(Loss.train_epoch.value))
            val_epoch_times, val_epoch_step_nums, val_epoch_losses = zip(*event_acc.Scalars(Loss.val_epoch.value))
            self._losses = {
                Loss.loss: loss_losses,
                Loss.train: train_losses,
                Loss.reg: reg_losses,
                Loss.train_epoch: train_epoch_losses,
                Loss.val_epoch: val_epoch_losses
            }
        return self._losses[loss_type]
    
    # define property to allow lazy loading
    dataset = property(_get_dataset)

    def run(self, device, experiment_folder):
        self.trial_directory = os.path.join(experiment_folder, self.trial_info.name)
        self.ckpts_directory  = os.path.join(self.trial_directory, 'checkpoints')
        
        # fit model
        assert self.model.NDN is not None

        force_dict_training = False
        # we need to force_dict_training if we are using lbfgs
        if self.trial_info.fit_params['optimizer_type'] == 'lbfgs':
            force_dict_training = True
        assert self.dataset.train_inds is not None, 'Trial ['+self.trial_info.name+']dataset is missing train_inds'
        assert self.dataset.val_inds is not None, 'Trial ['+self.trial_info.name+']dataset is missing val_inds'

        # train_ds = GenericDataset(self.dataset[self.dataset.train_inds], device=device)
        # val_ds = GenericDataset(self.dataset[self.dataset.val_inds], device=device)
        # self.model.NDN.fit_dl(train_ds, val_ds, save_dir=self.ckpts_directory, force_dict_training=force_dict_training, **self.trial_info.fit_params)

        self.model.NDN.fit(self.dataset, save_dir=self.ckpts_directory, force_dict_training=force_dict_training, **self.trial_info.fit_params)
        
        # eval model
        self.LLs = self.eval_function(self.model, self.dataset, device)

        # creating the checkpoints automatically creates the trial_directory,
        # but, let's just confirm here
        assert os.path.exists(self.trial_directory), 'Trial ['+self.trial_info.name+'] trial_directory is missing, training with checkpoints should have created it'

        # save model
        with open(os.path.join(self.trial_directory, 'model.pickle'), 'wb') as f:
            pickle.dump(self.model, f)
            
        # save trial_info
        with open(os.path.join(self.trial_directory, 'trial_info.pickle'), 'wb') as f:
            pickle.dump(self.trial_info, f)
        
        # save LLs
        with open(os.path.join(self.trial_directory, 'LLs.pickle'), 'wb') as f:
            pickle.dump(list(self.LLs), f)

    
class Experiment:
    def __init__(self, 
                 name:str,
                 description:str,
                 generate_trial,
                 experiment_location:str,
                 datadir:str=None,
                 overwrite:Overwrite=Overwrite.append):
        self.name = name
        self.description = description
        self.experiment_folder = os.path.join(experiment_location, name)
        self.datadir = datadir
        self.generate_trial = generate_trial
        self.overwrite = overwrite
        
        # if we don't specify how to overwrite
        if self.overwrite == Overwrite.none:
            assert not os.path.exists(self.experiment_folder), 'experiment_folder already exists and overwrite was not specified'
        
        # experiment trials
        self._trials = []
    
    def run(self, device):
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
        trials = []
        for trial in self.generate_trial(trials):
            assert trial is not None, 'generate_trial() callback must return a valid trial'
            trial.run(device, self.experiment_folder)
            trials.append(trial)
        assert len(trials) > 0, 'no trials were run'
        self.trials = trials # make sure to call the setter this way
    
    def plot_LLs(self, trials=None, figsize=(15,5)):
        # default to using all trials if not are specifically provided
        if trials is None:
            trials = [trial for trial in self.trials]
            
        if len(trials) == 0:
            print("No trials found to plot")
            return
        
        # get maximum num_neurons for the experiment
        max_num_neurons = 0
        for trial in trials:
            if trial.model.output.num_neurons > max_num_neurons:
                max_num_neurons = trial.model.output.num_neurons
        
        df = pd.DataFrame({
            'Neuron': ['N'+str(n) for n in range(max_num_neurons)],
        })
        
        # get the trials and order them by the desired order
        for trial in trials:
            df[trial.name] = np.concatenate((trial.LLs, np.zeros(max_num_neurons-len(trial.LLs), dtype=np.float32)))
        fig, ax1 = plt.subplots(figsize=figsize)
        tidy = df.melt(id_vars='Neuron').rename(columns=str.title)
        ax = sns.barplot(x='Neuron', y='Value', hue='Variable', data=tidy, ax=ax1)
        ax.set(xlabel='Neuron', ylabel='Log-Likelihood')
        plt.legend(title='Model')
        sns.despine(fig)
        
    def plot_losses(self, trials=None, loss_type=Loss.val_epoch, figsize=(15,5)):
        # default to using all trials if not are specifically provided
        if trials is None:
            trials = [trial for trial in self.trials]

        if len(trials) == 0:
            print("No trials found to plot")
            return
        
        plt.figure(figsize=figsize)
        for trial in trials:
            plt.plot(trial.losses(loss_type), label=trial.name)
        plt.legend()
        plt.show()
        
    def trials_where(self, union=False, **kwargs):
        assert len(self._trials) > 0, 'trials is empty'
        cond = '&' # default to intersection of params
        if union: # but, support union of params as well
            cond = '|'
        
        i = 0
        query = ""
        for k,v in kwargs.items():
            if i > 0: query += cond
            # use double quotes for query and single for strings inside the query
            if isinstance(v,str):
                query += k+"=='"+v+"'"
            else:
                query += k+"=="+str(v)
            i += 1

        # query and return the matching trial objects
        trials = [trial for trial in self.trials.query(query)['trial']]
        return trials
    
    def _set_trials(self, trials):
        # get all trial_params keys
        dfs = []
        for trial in trials:
            # the value going into a dataframe must be a list
            df = pd.DataFrame({k:[v] for k,v in trial.trial_info.trial_params.items()})
            df['name'] = trial.name
            df['trial'] = trial
            df['mean_LLs'] = np.mean(trial.LLs)
            dfs.append(df)
        
        # concatenate the individual DFs
        self._trials = pd.concat(dfs)
        self._trials = self.trials.sort_values(by='name') # sort alphabetically by name
    
    def _get_trials(self):
        return self._trials
    
    # property to encapsulate getting and setting trials
    trials = property(fget=_get_trials, fset=_set_trials)
