#%% Define Stuff
__author__ = 'SherlockLiao'

import os

import sys
# not best practice, but makes it easier to import from subdirectory
sys.path.insert(0, '../lib')

import matplotlib.pyplot as plt

import math
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import numpy as np

import utils as ut

device = torch.device("mps")

data = ut.load_data('ki_0503_V2b_py.mat')

num_epochs = 25
batch_size = 16  # use a nice factor of 848 to avoid padding and stuff
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataloader = DataLoader(data, batch_size=batch_size, num_workers=4, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # 3627 is the number of timepoints in a trial * the number of cells
            nn.Linear(3627, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            # use 3 latents
            nn.Linear(64, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 3627),
            # to allow for negative values in the output img
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
#%% Training Loop
min_trial_size = min([len(block_ind) for block_ind in data.block_inds])

model = autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

losses = []
for epoch in range(num_epochs):
    print('epochs '+str(epoch))
    # iterate over batches (927 batches)
    for batch_idx, start_block_idx in enumerate(range(0, len(data.block_inds), batch_size)):
        #print('batch '+str(batch_idx))
        # create the empty batch tensor
        batch = torch.zeros((batch_size, min_trial_size*data.NC)).to(device)
        # go through the timepoints for the block and accumulate the robs
        #print(start_block_idx, start_block_idx+batch_size)
        for row_idx, block_idx in enumerate(range(start_block_idx, start_block_idx+batch_size)):
            batch[row_idx, :] = data[data.block_inds[block_idx]]['robs'][0:117,:].reshape(3627)
            
        # ===================forward=====================
        output = model(batch)
        loss = criterion(output, batch)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print(loss.data.cpu())
    losses.append(loss.data.cpu())

#%%
