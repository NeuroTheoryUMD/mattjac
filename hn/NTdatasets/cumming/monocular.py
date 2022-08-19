
import os
from tkinter import N
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset

import NDNT.utils as utils

#from NDNT.utils import download_file, ensure_dir
from copy import deepcopy
import h5py

class MultiDataset(Dataset):
    """
    MULTIDATASET can load batches from multiple datasets
    """

    def __init__(self,
        sess_list,
        dirname,
        preload=False,
        num_lags=1,
        time_embed=True,
        includeMUs=False,
        device=None):

        self.dirname = dirname
        self.sess_list = sess_list
        self.num_lags = num_lags

        # get hdf5 file handles
        self.fhandles = [h5py.File(os.path.join(dirname, sess + '.hdf5'), 'r') for sess in self.sess_list]

        # build index map
        self.file_index = [] # which file the block corresponds to
        self.block_inds = []
        self.NTfile = []

        self.unit_ids = []
        self.includeMUs = includeMUs
        self.num_units, self.num_sus, self.num_mus, self.sus = [], [], [], []
        self.dims_file = []

        if (device is not None) and (not preload):
            preload = True
            print("Warning: switching preload to True so device argument is meaningful.")
        
        self.preload = preload
        self.device = device

        self.NT = 0
        self.NC = 0
        self.num_blocks = 0
        self.block_assign = []
        self.block_grouping = []
        nfiles = 0
        self.cells_out = None

        for f, fhandle in enumerate(self.fhandles):
            NTfile = fhandle['robs'].shape[0]
            NCfile = fhandle['robs'].shape[1]
            NMUfile = fhandle['robsMU'].shape[1]
            self.num_sus.append(NCfile)
            self.num_mus.append(NMUfile)
            self.sus = self.sus + list(range(self.NC, self.NC+NCfile))

            self.dims_file.append(fhandle['stim'].shape[1])
            if includeMUs:
                NCfile += NMUfile
            self.unit_ids.append(self.NC + np.asarray(range(NCfile)))
            
            self.num_units.append(NCfile)
            self.NTfile.append(NTfile)

            # Pull blocks from data_filters
            blocks = (np.sum(fhandle['dfs'][:,:], axis=1)==0).astype(np.float32)
            blocks[0] = 1 # set invalid first sample
            blocks[-1] = 1 # set invalid last sample

            blockstart = np.where(np.diff(blocks)==-1)[0]
            blockend = np.where(np.diff(blocks)==1)[0]
            nblocks = len(blockstart)

            for b in range(nblocks):
                self.file_index.append(f)
                self.block_inds.append(self.NT + np.arange(blockstart[b], blockend[b]))
            
            # Assign each block to a file
            self.block_assign = np.concatenate(
                (self.block_assign, nfiles*np.ones(NTfile, dtype=int)), axis=0)
            self.block_grouping.append( self.num_blocks+np.arange(nblocks, dtype=int) )
            self.NT += NTfile
            self.NC += NCfile
            self.num_blocks += nblocks
            nfiles += 1

        # Set overall dataset variables
        NX = np.unique(np.asarray(self.dims_file)) # assumes they're all the same
        assert len(NX) == 1, 'problems'
        self.dims = [NX[0], 1, 1]
        if time_embed:
            self.dims[-1] = num_lags

        if preload:
            self.stim = np.zeros([self.NT, np.prod(self.dims)], dtype=np.float32)
            self.robs = np.zeros([self.NT, self.NC], dtype=np.float32)
            self.dfs = np.zeros([self.NT, self.NC], dtype=np.float32)
            tcount, ccount = 0, 0
            for f, fhandle in enumerate(self.fhandles):
                NT = fhandle['robs'].shape[0]
                NC = fhandle['robs'].shape[1]
                trange = range(tcount, tcount+NT)
                crange = range(ccount, ccount+NC)
                
                # Stimulus
                if not time_embed:
                    self.stim[trange, :] = np.array(self.fhandles[f]['stim'], dtype='float32')
                else:
                    # Time embed stimulus -- simple way
                    idx = np.arange(NT)
                    tmp = np.array(self.fhandles[f]['stim'], dtype='float32')
                    self.stim[trange, :] = np.reshape( 
                        np.transpose(
                            tmp[np.arange(NT)[:,None]-np.arange(self.num_lags), :], 
                            [0,2,1]),
                        [NT, -1])

                # Robs and DFs
                robs_tmp = np.zeros([NT, self.NC], dtype=np.float32)
                dfs_tmp = np.zeros([NT, self.NC], dtype=np.float32)
                robs_tmp[:, crange] = np.array(self.fhandles[f]['robs'], dtype='float32')
                dfs_tmp[:, crange] = np.array(self.fhandles[f]['dfs'], dtype='float32')
                if includeMUs:
                    NMU = fhandle['robsMU'].shape[1]
                    crange = range(ccount+NC, ccount+NC+NMU)
                    NC += NMU
                    robs_tmp[:, crange] = np.array(self.fhandles[f]['robsMU'], dtype='float32')
                    dfs_tmp[:, crange] = np.array(self.fhandles[f]['dfsMU'], dtype='float32')

                self.robs[trange, :] = deepcopy(robs_tmp)
                self.dfs[trange, :] = deepcopy(dfs_tmp)
                tcount += NT
                ccount += NC

            # Convert data to tensor
            self.to_tensor()

        self.sus = np.array(self.sus, dtype=np.int64)
        # Set average rates across dataset (so can be quickly accessed)
        self.avRs = None  # need to set to None so can will pre-calculate
        self.avRs = self.avrates()

        # Set up default cross-validation config
        self.crossval_setup()
    # END MultiDataset.__init__

    def to_tensor(self, device=None):
        if device is None:
            if self.device is None:
                device = torch.device("cpu")
            else:
                device = self.device

        if type(self.robs) != torch.Tensor:
            self.stim = torch.tensor(self.stim, dtype=torch.float32, device=device)
            self.robs = torch.tensor(self.robs, dtype=torch.float32, device=device)
            self.dfs = torch.tensor(self.dfs, dtype=torch.float32, device=device)
        else: 
            # Simply move to device:
            self.stim = self.stim.to( device )
            self.robs = self.robs.to( device )
            self.dfs = self.dfs.to( device )
    # END MultiDataset.to_tensor

    def __getitem__(self, index):
        
        if isinstance(index, int) or isinstance(index, np.int64):
            index = np.array([index], dtype=np.int64)
        elif type(index) is slice:
            index = list(range(index.start or 0, index.stop or self.NT, index.step or 1))
        if isinstance(index, list):
            index = np.array(index, dtype=np.int64)
    
        if self.preload:
            stim = self.stim[index, :]
            robs = self.robs[index, :]
            dfs = self.dfs[index, :]
        else:
            stim = []
            robs = []
            dfs = []
            for ii in index:
                inds = self.block_inds[ii]
                NT = len(inds)
                f = self.file_index[ii]

                """ Stim """
                stim_tmp = torch.tensor(self.fhandles[f]['stim'][inds,:], dtype=torch.float32)

                """ Spikes: needs padding so all are B x NC """ 
                robs_tmp = torch.tensor(self.fhandles[f]['robs'][inds,:], dtype=torch.float32)
                NCbefore = int(np.asarray(self.num_units[:f]).sum())
                NCafter = int(np.asarray(self.num_units[f+1:]).sum())
                robs_tmp = torch.cat(
                    (torch.zeros( (NT, NCbefore), dtype=torch.float32),
                    robs_tmp,
                    torch.zeros( (NT, NCafter), dtype=torch.float32)),
                    dim=1)

                """ Datafilters: needs padding like robs """
                dfs_tmp = torch.tensor(self.fhandles[f]['dfs'][inds,:], dtype=torch.float32)
                dfs_tmp[:self.num_lags,:] = 0 # invalidate the filter length
                dfs_tmp = torch.cat(
                    (torch.zeros( (NT, NCbefore), dtype=torch.float32),
                    dfs_tmp,
                    torch.zeros( (NT, NCafter), dtype=torch.float32)),
                    dim=1)

                stim.append(stim_tmp)
                robs.append(robs_tmp)
                dfs.append(dfs_tmp)

            stim = torch.cat(stim, dim=0)
            robs = torch.cat(robs, dim=0)
            dfs = torch.cat(dfs, dim=0)

        if self.cells_out is not None:
            cells_out = np.array(self.cells_out, dtype=np.int64)
            assert len(cells_out) > 0, "DATASET: cells_out must be a non-zero length"
            assert np.max(cells_out) < self.robs.shape[1],  "DATASET: cells_out must be a non-zero length"
            return {'stim': stim, 'robs': robs[:, cells_out], 'dfs': dfs[:, cells_out]}
        else:
            return {'stim': stim, 'robs': robs, 'dfs': dfs}
    # END MultiDataset.__get_item__

    def __len__(self):
        return self.NT
    
    ###### Additional functions that might not be useful yet #####
    def avrates( self, inds=None ):
        """
        Calculates average firing probability across specified inds (or whole dataset)
        -- Note will respect datafilters
        -- will return precalc value to save time if already stored
        """
        if inds is None:
            inds = range(self.NT)
        if len(inds) == self.NT:
            # then calculate across whole dataset
            if self.avRs is not None:
                if self.cells_out is None:
                    return self.avRs
                if len(self.cells_out) == len(self.avRs):
                    # then precalculated and do not need to do
                    return self.avRs

        # Otherwise calculate across all data
        if self.preload:
            Reff = (self.dfs * self.robs).sum(dim=0).cpu()
            Teff = self.dfs.sum(dim=0).clamp(min=1e-6).cpu()
            if self.cells_out is not None:
                if len(self.cells_out) > 0:
                    return (Reff[self.cells_out]/Teff[self.cells_out]).detach().numpy()
            return (Reff/Teff).detach().numpy()
        else:
            print('Still need to implement avRs without preloading')
            return None
    # END MultiDatasset.avrates()

    def subset( self, indxs=None, train=True, val=False, device=None ):
        assert self.preload, "Need preloaded data for this to work"

        self.fhandles = None
        if indxs is None:
            if train:
                assert not val, "Either train or val must be false."
                indxs = self.train_inds
            else:
                assert val, "Either train or val must be true"
                indxs = self.val_inds
        
        # Crop all preloaded data down to indices
        self.val_inds = None
        self.train_inds = None
        self.stim = self.stim[indxs, ...]
        self.robs = self.robs[indxs, :]
        self.dfs = self.dfs[indxs, :]
        self.NT = len(indxs)
    # End Subset

    def shift_stim_fixation( self, stim, shift):
        """Simple shift by integer (rounded shift) and zero padded. Note that this is not in 
        is in units of number of bars, rather than -1 to +1. It assumes the stim
        has a batch dimension (over a fixation), and shifts the whole stim by the same amount."""
        sh = round(shift)
        shstim = stim.new_zeros(*stim.shape)
        if sh < 0:
            shstim[:, -sh:] = stim[:, :sh]
        elif sh > 0:
            shstim[:, :-sh] = stim[:, sh:]
        else:
            shstim = deepcopy(stim)

        return shstim
    # END MultiDatasetFix.shift_stim_fixation

    def crossval_setup(self, folds=5, random_gen=False, test_set=False):
        """This sets the cross-validation indices up We can add featuers here. Many ways to do this
        but will stick to some standard for now. It sets the internal indices, which can be read out
        directly or with helper functions. Perhaps helper_functions is the best way....
        
        Inputs: 
            random_gen: whether to pick random fixations for validation or uniformly distributed
            test_set: whether to set aside first an n-fold test set, and then within the rest n-fold train/val sets
        Outputs:
            None: sets internal variables test_inds, train_inds, val_inds
        """

        test_fixes = []
        tfixes = []
        vfixes = []
        for ee in range(len(self.block_grouping)):
            blk_inds = self.block_grouping[ee]
            vfix1, tfix1 = self.fold_sample(len(blk_inds), folds, random_gen=random_gen)
            if test_set:
                test_fixes += list(blk_inds[vfix1])
                vfix2, tfix2 = self.fold_sample(len(tfix1), folds, random_gen=random_gen)
                vfixes += list(blk_inds[tfix1[vfix2]])
                tfixes += list(blk_inds[tfix1[tfix2]])
            else:
                vfixes += list(blk_inds[vfix1])
                tfixes += list(blk_inds[tfix1])

        val_blks = np.array(vfixes, dtype='int64')
        train_blks = np.array(tfixes, dtype='int64')
        
        # Assign indexes based on validation by block
        self.train_inds = []
        self.test_inds = []
        self.val_inds = []
        for bb in train_blks:
            self.train_inds += list(self.block_inds[bb])
        for bb in val_blks:
            self.val_inds += list(self.block_inds[bb])
        if test_set:
            test_blks = np.array(test_fixes, dtype='int64')
            for bb in test_blks:
                self.test_inds += list(self.block_inds[bb])

        # finally convert to np.array
        self.train_inds = np.array(self.train_inds)
        self.val_inds = np.array(self.val_inds)
        self.test_inds = np.array(self.test_inds)
    # END MultiDatasetFix.crossval_setup

    def fold_sample( self, num_items, folds, random_gen=False):
        """This really should be a general method not associated with self"""
        if random_gen:
            num_val = int(num_items/folds)
            tmp_seq = np.random.permutation(num_items)
            val_items = np.sort(tmp_seq[:num_val])
            rem_items = np.sort(tmp_seq[num_val:])
        else:
            offset = int(folds//2)
            val_items = np.arange(offset, num_items, folds, dtype='int32')
            rem_items = np.delete(np.arange(num_items, dtype='int32'), val_items)
        return val_items, rem_items



def get_stim_url(id):
    
    urlpath = {
            'expt01': 'https://www.dropbox.com/s/mn70kyohmp3kjnl/expt01.mat?dl=1',
            'expt02':'https://www.dropbox.com/s/pods4w89tbu2x57/expt02.mat?dl=1',
            'expt03': 'https://www.dropbox.com/s/p08375vcunrf9rh/expt03.mat?dl=1',
            'expt04': 'https://www.dropbox.com/s/zs1vcaz3sm01ncn/expt04.mat?dl=1',
            'expt05': 'https://www.dropbox.com/s/f3mpp3mlsrhof8k/expt05.mat?dl=1',
            'expt06': 'https://www.dropbox.com/s/saqjo7yibc6y8ut/expt06.mat?dl=1',
            'expt07': 'https://www.dropbox.com/s/op0rw7obzfvnm53/expt07.mat?dl=1',
            'expt08': 'https://www.dropbox.com/s/fwmtdegmlcdk9wo/expt08.mat?dl=1',
            'expt09': 'https://www.dropbox.com/s/yo8xo58ldiyrktm/expt09.mat?dl=1',
            'expt10': 'https://www.dropbox.com/s/k2zldzv7zfe7x06/expt10.mat?dl=1',
            'expt11': 'https://www.dropbox.com/s/rsc7h4njqntts39/expt11.mat?dl=1',
            'expt12': 'https://www.dropbox.com/s/yf1mm805j53yaj2/expt12.mat?dl=1',
            'expt13': 'https://www.dropbox.com/s/gidll8bgg5uie8h/expt13.mat?dl=1',
            'expt14': 'https://www.dropbox.com/s/kfof5m08g1v3rfe/expt14.mat?dl=1',
            'expt15': 'https://www.dropbox.com/s/zpcc7a2iy9bmkjd/expt15.mat?dl=1',
            'expt16': 'https://www.dropbox.com/s/b19kwdwy18d14hl/expt16.mat?dl=1',
        }
    
    if id not in urlpath.keys():
        raise ValueError('Stimulus URL not found')
    
    return urlpath[id]

def download_set(sessname, fpath):
    
    ensure_dir(fpath)

    # Download the data set
    url = get_stim_url(sessname)
    fout = os.path.join(fpath, sessname + '.mat')
    download_file(url, fout)

# --- Define data-helpers
def time_in_blocks(block_inds):
    num_blocks = block_inds.shape[0]
    #print( "%d number of blocks." %num_blocks)
    NT = 0
    for nn in range(num_blocks):
        NT += block_inds[nn,1]-block_inds[nn,0]+1
    return NT


def make_block_inds( block_lims, gap=20, separate = False):
    block_inds = []
    for nn in range(block_lims.shape[0]):
        if separate:
            block_inds.append(np.arange(block_lims[nn,0]-1+gap, block_lims[nn,1]), dtype='int')
        else:
            block_inds = np.concatenate( 
                (block_inds, np.arange(block_lims[nn,0]-1+gap, block_lims[nn,1], dtype='int')), axis=0)
    return block_inds


def monocular_data_import( datadir, exptn, num_lags=20 ):

    from copy import deepcopy

    time_shift = 1
    filename = exptn + '.mat'
    matdata = sio.loadmat( os.path.join(datadir,filename) )

    sus = matdata['goodSUs'][:,0] - 1  # switch from matlab indexing
    print('SUs:', sus)
    NC = len(sus)
    layers = matdata['layers'][0,:]
    block_list = matdata['block_inds'] # note matlab indexing
    stim_all = Utils.shift_mat_zpad(matdata['stimulus'], time_shift, 0)
    NTtot, NX = stim_all.shape
    DFs_all = deepcopy(matdata['data_filters'][:,sus])
    Robs_all = deepcopy(matdata['binned_SU'][:,sus])
    
    # Break up into train and test blocks
    # Assemble train and test indices based on BIlist
    NBL = block_list.shape[0]
    Xb = np.arange(2, NBL, 5)  # Every fifth trial is cross-validation
    Ub = np.array(list(set(list(range(NBL)))-set(Xb)), dtype='int')
    
    used_inds = make_block_inds( block_list, gap=num_lags )
    Ui, Xi = Utils.generate_xv_folds( len(used_inds) )
    TRinds, TEinds = used_inds[Ui].astype(int), used_inds[Xi].astype(int)

    Eadd_info = {
        'cortical_layer':layers, 'used_inds': used_inds, 
        'TRinds':TRinds, 'TEinds': TEinds, #'TRinds': Ui, 'TEinds': Xi, 
        'block_list': block_list, 'TRblocks': Ub, 'TEblocks': Xb}
    return stim_all, Robs_all, DFs_all, Eadd_info