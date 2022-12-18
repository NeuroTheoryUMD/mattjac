#%% Define Stuff

# From:
# https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py
# Explanation of backward functions
# https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944

import os
import sys
# not best practice, but makes it easier to import from subdirectory
sys.path.insert(0, '../lib')

import matplotlib.pyplot as plt
from tqdm import tqdm

import math
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import numpy as np

import utils as ut

# use Apple Silicon GPU (Metal Performance Shaders)
device = torch.device("mps")

# load the data
data = ut.load_data('ki_0503_V2b_py.mat')
# determine minimim trial size to truncate the trial lengths to
# it is 117 in this case
min_trial_size = min([len(block_ind) for block_ind in data.block_inds])

# params for training
num_epochs = 20
batch_size = 16  # use a factor of 848 to avoid padding and stuff
learning_rate = 1e-3

# utility to get the batch of robs for a trial and format it correctly
def get_batch(start_block_idx, batch_size, normalize=True):
    """
    Returns the batch from the starting block index (trial index),
    up to batch_size subsequent trials afterward (stacked in an array). 
    :param start_block_idx: starting block (trial) index
    :param batch_size: number of subsequent trials to stack
    :return: batch_size X 3627 array
    """
    # create the empty batch tensor
    batch = torch.zeros((batch_size, min_trial_size*data.NC)).to(device)
    # go through the timepoints for the block and accumulate the robs
    #print(start_block_idx, start_block_idx+batch_size)
    for row_idx, block_idx in enumerate(range(start_block_idx, start_block_idx+batch_size)):
        batch[row_idx, :] = data[data.block_inds[block_idx]]['robs'][0:117,:].reshape(3627)

    # normalize the batch, if told to
    if normalize:
        batch = F.normalize(batch, dim=0)
    return batch

# our AutoEncoder class
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # 3627 is the number of timepoints in a trial * the number of cells
            nn.Linear(3627, 2048),
            # use 128 latents
            nn.Linear(2048, 128))
        self.decoder = nn.Sequential(
            nn.Linear(128, 2048),
            nn.Linear(2048, 3627))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
#%% Training Loop
model = AutoEncoder().to(device)
criterion = nn.MSELoss()
#optimizer = torch.optim.LBFGS(model.parameters(), history_size=10, max_iter=4)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

losses = []
for epoch in tqdm(range(num_epochs)):
    # iterate over batches (927 batches)
    for batch_idx, start_block_idx in enumerate(range(0, len(data.block_inds), batch_size)):
        # get the batch
        batch = get_batch(start_block_idx, batch_size, normalize=True)
        # forward
        output = model(batch) # predict
        loss = criterion(output, batch) # calculate the loss
        # backward
        optimizer.zero_grad() # zero the gradients
        loss.backward() # calculate the gradient
        optimizer.step() # update x with the gradient
    # store losses
    losses.append(loss.data.cpu())

# show the losses at the end of training
plt.figure()
plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

#%% get the latents and plot the 2D manifold with UMAP
batch = get_batch(0, 848)
latents = model.encoder(batch).cpu().detach().numpy() # latents

from sklearn.preprocessing import StandardScaler
import umap

reducer = umap.UMAP()
scaled_latents = StandardScaler().fit_transform(latents)

embedding = reducer.fit_transform(scaled_latents)
print(embedding.shape)

scat = plt.scatter(
    embedding[:,0],
    embedding[:,1],
    c=data.TRcued) # color by cued trials
plt.colorbar(scat) # show the colorbar (like a legend, but for colors)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the latents')

# calculate r2 between predicted and actual robs
#reals = batch
#preds = model.decoder(torch.tensor(latents).to(device))

#mean_reals = torch.mean(reals, axis=0)

# I don't think this is right
#r2 = torch.sum((reals - preds)**2, axis=0) / torch.sum((reals - mean_reals)**2, axis=0)
#print(r2)
# r2 = sumSquaredRegression / totalSumOfSquares
# sumSquaredRegression = sum((actual - predicted)**2)
# totalSumOfSquared = sum((actual - mean_actual)**)


#%% from Problem Set 7
# r2 = 1 - (var(ratesR-predR') / var(ratesR))

dt = 1.0/120.0  # bin time

# get all robs
reals = get_batch(0, 848, normalize=False)
batch = get_batch(0, 848, normalize=True)
# encode and decode to see if the autoencoder reproduces the input correctly
latents = model.encoder(batch)
preds = model.decoder(latents)

# reshape the data into trials x time x robs
reals = batch.reshape(848,117,31).cpu().detach().numpy()
preds = preds.reshape(848,117,31).cpu().detach().numpy()

# calculate firing rates from the reals and preds
# calculate the trial-averaged rates (divide by dt to get it in terms of Hz
# NOTE: this is incorrect, as we cannot try to predict the Hz from the normalized robs,
#       we need to make sure real robs and pred robs are scaled correctly
ratesR = np.mean(reals[:,:,0], axis=0) / dt
ratesP = np.mean(preds[:,:,0], axis=0) / dt

# calculate r2
r2 = 1 - (np.var(ratesR-ratesP) / np.var(ratesR))

# the first 3 points or so are not very good, so remove them from the plot
plt.plot(ratesP[3:], label='pred rates')
plt.plot(ratesR[3:], label='real rates')
plt.legend()

print('r2 = '+str(r2))
#%%
