# place to track and interact with all the models we have trained

import sys
sys.path.append('../')

import os
import io
import pickle
import numpy as np
from dataclasses import dataclass
from NDNT.modules.layers import *
from NDNT.networks import *

model_dir = '../models'

class MPSUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='mps')
        else: return super().find_class(module, name)

class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

# need a class to load the pickles
class Model:
    def __init__(self, ndn_model, LLs, trial):
        self.ndn = ndn_model
        self.LLs = LLs
        self.trial = trial

def load_model(fname):
    with open(os.path.join(model_dir, fname), 'rb') as f:
        return CPUUnpickler(f).load()


# 3-layer CNNs
cnn_0715_1x = load_model('cnns_multi_08_1x_0715/cnn_0.pkl')
cnn_0722_1x = load_model('cnns_multi_08_1x_0722/cnn_0.pkl')
cnn_0801_1x = load_model('cnns_multi_08_1x_0801/cnn_0.pkl')
cnn_0808_1x = load_model('cnns_multi_08_1x_0808/cnn_0.pkl')
cnn_0715_2x = load_model('cnns_multi_08_2x_0715/cnn_0.pkl')
cnn_0722_2x = load_model('cnns_multi_08_2x_0722/cnn_0.pkl')
cnn_0801_2x = load_model('cnns_multi_08_2x_0801/cnn_0.pkl')
cnn_0808_2x = load_model('cnns_multi_08_2x_0808/cnn_0.pkl')
cnn_core    = load_model('cnns_multi_08/cnn_0.pkl') # on ['J220715','J220722','J220801','J220808']

# 7-layer iterative models
iter_core    = load_model('cnns_multi_06/cnn_0.pkl') # on ['J220715','J220722','J220801','J220808']
iter_0715_1x = load_model('cnns_multi_06_1x_0715/cnn_0.pkl')
iter_0722_1x = load_model('cnns_multi_06_1x_0722/cnn_0.pkl')
iter_0801_1x = load_model('cnns_multi_06_1x_0801/cnn_0.pkl')
iter_0808_1x = load_model('cnns_multi_06_1x_0808/cnn_0.pkl')
iter_0715_2x = load_model('cnns_multi_06_2x_0715/cnn_0.pkl')
iter_0722_2x = load_model('cnns_multi_06_2x_0722/cnn_0.pkl')
iter_0801_2x = load_model('cnns_multi_06_2x_0801/cnn_0.pkl')
iter_0808_2x = load_model('cnns_multi_06_2x_0808/cnn_0.pkl')

if __name__ == "__main__":
    sigfigs = 4
    print('Model       \t', 'mean LL')
    print('Iter')
    print('iter_core   \t', np.round(np.mean(iter_core.LLs), sigfigs))
    print('iter_0715_1x\t', np.round(np.mean(iter_0715_1x.LLs), sigfigs))
    print('iter_0722_1x\t', np.round(np.mean(iter_0722_1x.LLs), sigfigs))
    print('iter_0801_1x\t', np.round(np.mean(iter_0801_1x.LLs), sigfigs))
    print('iter_0808_1x\t', np.round(np.mean(iter_0808_1x.LLs), sigfigs))
    print('iter_0715_2x\t', np.round(np.mean(iter_0715_2x.LLs), sigfigs))
    print('iter_0722_2x\t', np.round(np.mean(iter_0722_2x.LLs), sigfigs))
    print('iter_0801_2x\t', np.round(np.mean(iter_0801_2x.LLs), sigfigs))
    print('iter_0808_2x\t', np.round(np.mean(iter_0808_2x.LLs), sigfigs))
    print()
    print('CNN')
    print('cnn_core   \t', np.round(np.mean(cnn_core.LLs), sigfigs))
    print('cnn_0715_1x\t', np.round(np.mean(cnn_0715_1x.LLs), sigfigs))
    print('cnn_0722_1x\t', np.round(np.mean(cnn_0722_1x.LLs), sigfigs))
    print('cnn_0801_1x\t', np.round(np.mean(cnn_0801_1x.LLs), sigfigs))
    print('cnn_0808_1x\t', np.round(np.mean(cnn_0808_1x.LLs), sigfigs))
    print('cnn_0715_2x\t', np.round(np.mean(cnn_0715_2x.LLs), sigfigs))
    print('cnn_0722_2x\t', np.round(np.mean(cnn_0722_2x.LLs), sigfigs))
    print('cnn_0801_2x\t', np.round(np.mean(cnn_0801_2x.LLs), sigfigs))
    print('cnn_0808_2x\t', np.round(np.mean(cnn_0808_2x.LLs), sigfigs))

    
    
