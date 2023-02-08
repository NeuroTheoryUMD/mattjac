import numpy as np


def r2(data, model, test_inds=None):
    # run the stimulus through the model and compare with the robs
    # run the stim forward through the model
    if test_inds is not None: # use only the provided test indices
        y_pred = (model({'stim': data.stim[test_inds]})  * data.dfs[test_inds]).detach().numpy()
        y_test = (data.robs[test_inds] * data.dfs[test_inds]).detach().numpy()
    else: # use all the data
        y_pred = (model({'stim': data.stim}) * data.dfs).detach().numpy()
        y_test = (data.robs * data.dfs).detach().numpy()
    
    return 1 - np.sum((y_pred - y_test)**2, axis=0) / np.sum((y_test - np.mean(y_test))**2, axis=0)

