import pickle
import torch
import numpy as np

import NTdatasets.HN.HNdatasets as datasets


def load_data(_filename, _num_lags=12):
    # load the data from the file
    data = datasets.HNdataset(filename=_filename,
                              datadir='../../data/hn/',
                              drift_interval=90)
    data.prepare_stim(which_stim='left', num_lags=_num_lags)
    return data


def load_model(_filename):
    # load the model
    with open(_filename, 'rb') as f:
        model = pickle.load(f)
    return model


def enpickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def depickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def construct_R_matrix(data):
    # determine R matrix shape (trials X time X neurons)
    num_trials = len(data.block_inds)
    max_trial_length = max([len(data.block_inds[i]) for i in range(len(data.block_inds))])
    num_neurons = data.robs.shape[1]

    # get the spikes for the neurons
    robs = data.robs.numpy()

    # construct R matrix (trials X time X neurons)
    R = np.zeros((num_trials, max_trial_length, num_neurons))
    for i, block_inds in enumerate(data.block_inds): # go through each trial
        trial_length = robs[block_inds].shape[0]
        if trial_length < max_trial_length:
            # add a number of empty rows to R for the missing timepoints in the trial
            # number of missing timepoints = max_trial_length - trial_length
            R[i, :, :] = np.vstack([data.robs[block_inds].numpy(),
                                    np.zeros((max_trial_length - trial_length, num_neurons))])
        else:
            R[i, :, :] = robs[block_inds]
    return R


def construct_Z_matrix(R, model, num_latents=3):
    # determine Z matrix shape (trials X time X latents)
    num_trials = R.shape[0]
    max_trial_length = R.shape[1]
    
    # construct Z matrix (trials X time X latents)
    Z = np.zeros((num_trials, max_trial_length, num_latents))

    for trial in range(num_trials):
        z = model.networks[1].layers[0](torch.from_numpy(R[trial, :, :]).float()) # need to cast back to float
        z = [z_i.detach().numpy() for z_i in z]  # convert each to a numpy array
        Z[trial, :, :] = z
    return Z


# TODO: this is also probably unnecessary
def calc_behavior_centroids(latents, behaviors, behavior_categories):
    category_to_centroid = {}
    for behavior_category in behavior_categories: # e.g. -1, 1
        # filter the latents by the current category
        latents_for_category = latents[np.where(behaviors == behavioral_category)]
        # get mean of latents for this category, across columns
        mean_latents_for_category = np.mean(latents_for_category, axis=0)
        category_to_centroid[behavior_category] = mean_latents_for_category
    return category_to_centroid

