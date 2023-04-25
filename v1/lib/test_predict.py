import sys
# not best practice, but makes it easier to import from subdirectory
sys.path.insert(0, './')

import pytest
import pickle
import torch
import numpy as np
import predict
from predict import Results
from NTdatasets.cumming.monocular import MultiDataset

# Load the pre-trained model and dataset
with open('model_regexp04_cnimscaf3.pickle', 'rb') as f:
    model = pickle.load(f)

expt = ['expt01', 'expt02', 'expt03', 'expt04',
          'expt05', 'expt06', 'expt07', 'expt08',
          'expt09', 'expt10', 'expt11', 'expt12']
dataset_params = {
    'datadir': '../Mdata/',
    'filenames': expt,
    'include_MUs': False,
    'time_embed': True,
    'num_lags': 10
}
dataset = MultiDataset(**dataset_params)


# Test cases
def test_predict():
    results = predict.predict(model, dataset=dataset)

    assert isinstance(results, Results)
    assert results.inps is not None
    assert results.outputs is not None
    assert results.jacobians is not None
    assert results.robs is not None
    assert results.pred is not None
    assert results.r2 is not None
    assert results.jacobian is not None


def test_results_attributes():
    results = predict.predict(model, dataset=dataset)

    assert hasattr(results, 'inps')
    assert hasattr(results, 'outputs')
    assert hasattr(results, 'jacobians')
    assert hasattr(results, 'robs')
    assert hasattr(results, 'pred')
    assert hasattr(results, 'r2')
    assert hasattr(results, 'jacobian')


def test_results_shapes():
    results = predict.predict(model, dataset=dataset)

    assert results.inps.shape == dataset['stim'].shape
    assert results.robs.shape == dataset['robs'].shape
    assert results.pred.shape == dataset['robs'].shape
    assert len(results.jacobian) == dataset['stim'].shape[0]
