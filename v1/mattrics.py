from sklearn.metrics import r2_score


def r2(data, model, test_inds=None, variance_weighted=False):
    # run the stimulus through the model and compare with the robs
    # run the stim forward through the model
    if test_inds: # use only the provided test indices
        y_pred = model({'stim': data.stim[test_inds]}).detach().numpy()
        y_test = data.robs[test_inds].detach().numpy()
    else: # use all the data
        y_pred = model({'stim': data.stim}).detach().numpy()
        y_test = data.robs.detach().numpy()
    
    # set the flag
    if variance_weighted: 
        multioutput = 'variance_weighted'
    else:
        multioutput = 'raw_values'
    
    return r2_score(y_test, y_pred, multioutput=multioutput)

