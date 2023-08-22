# place to track and interact with all the models we have trained

import sys
sys.path.append('../')

import os
import pickle
from dataclasses import dataclass
from NDNT.modules.layers import *
from NDNT.networks import *

model_dir = '../models'

def load_model(fname):
    with open(os.path.join(model_dir, fname), 'rb') as f:
        return pickle.load(f)


# 3-layer CNNs
cnn_0715_1x = None
cnn_0722_1x = None
cnn_0801_1x = None
cnn_0808_1x = None
cnn_0715_2x = None
cnn_0722_2x = None
cnn_0808_2x = None
cnn_core = None # 'cnns_multi_08' # on ['J220715','J220722','J220801','J220808']

# 7-layer iterative models
iter_core = load_model('cnns_multi_06/cnn_0.pkl') # # on ['J220715','J220722','J220801','J220808']
#iter_0715_1x = load_model('cnns_multi_06_1x_0715/cnn_0.pkl')
