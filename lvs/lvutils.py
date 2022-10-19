# utilities for simplifying common operations for analyzing latent variables

import numpy as np
from sklearn.linear_model import LogisticRegression


def centroid_vector(Z, Y):
    # take means across trials and time
    classes = np.unique(Y)
    if len(classes) > 2:
        assert "Y can only have 2 classes"

    # get the indices for each class
    Y0 = np.where(Y == classes[0])
    Y1 = np.where(Y == classes[1])

    # calculate the centroids
    Y1_mean = np.mean(np.mean(Z[Y0], axis=1), axis=0)
    Y2_mean = np.mean(np.mean(Z[Y1], axis=1), axis=0)

    # calculate vector between the centroids
    vector = Y1_mean - Y2_mean
    # normalize
    vector /= np.sqrt(vector @ vector.T)
    # reshape to a row vector
    vector = np.array([vector])
    return vector


def hyperplane_vector(Z, Y):
    # fit logistic regression model between Y binary classes
    Y_trial_model = LogisticRegression(random_state=0).fit(np.mean(Z, axis=1), Y)
    coef = Y_trial_model.coef_

    # calculate hyperplane vector
    vector = (coef / np.sqrt(coef @ coef.T))
    return vector

