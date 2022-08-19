import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import NDNT.utils as utils
#from NDNT.utils import download_file, ensure_dir
from copy import deepcopy
import h5py
from NTdatasets.sensory_base import SensoryBase

class HNdataset(SensoryBase):

    def __init__(self, filename=None, which_stim='left', **kwargs):

        # call parent constructor
        super().__init__(filename, **kwargs)
        print(self.datadir, filename)
        matdat = sio.loadmat(self.datadir+filename)
        print('Loaded ' + filename)
        #matdat = sio.loadmat('Data/'+exptname+'py.mat')
        self.disp_list = matdat['disp_list'][:,0]
        self.stimlist = np.unique(matdat['stimL'])
        self.Nstim = len(self.stimlist)

        self.TRcued = matdat['cued'][:,0] # Ntr 
        self.TRchoice = matdat['choice'][:,0] # Ntr 
        self.TRsignal = matdat['signal']  # Ntr x 2 (sorted by RF)
        self.TRstrength = matdat['strength']  # Ntr x 2 (sorted by RF)
        #self.TRstim = matdat['cued_stim']  # Ntr x 4 (sorted by cued, then uncued)
        # Detect disparities used for decision (indexed by stimulus number)
        #decision_stims = np.where(matdat['disp_list'] == np.unique(matdat['cued_stim'][:,0]))[0]

        self.TRstim = np.multiply(self.TRsignal, self.TRstrength)  # stim strength and direction combined
        
        ### Process neural data
        self.robs = torch.tensor( matdat['Robs'], dtype=torch.float32 )
        self.NT, self.NC = self.robs.shape
        # Make datafilters
        self.dfs = torch.zeros( [self.NT, self.NC], dtype=torch.float32 )
        self.used_inds = matdat['used_inds'][:,0] - 1
        self.dfs[self.used_inds, :] = 1.0
        #modvars = matdat['moduvar']

        # High resolution stimuli: note these are already one-hot matrices
        self.stimL = matdat['stimL']
        self.stimR = matdat['stimR']

        # Saccade info
        self.Xsacc = torch.tensor( matdat['Xsacc'], dtype=torch.float32 )
        #saccdirs = matdat['sacc_dirs']

        # Make block_inds
        blks = matdat['blks']
        self.Ntr = blks.shape[0]
        self.Nframes = np.min(np.diff(blks))
        for bb in range(self.Ntr):
            self.block_inds.append( np.arange(blks[bb,0]-1, blks[bb,1], dtype=np.int64) )

        self.CHnames = [None]*self.NC
        for cc in range(self.NC):
            self.CHnames[cc] = matdat['CHnames'][0][cc][0]
        #expt_info = {'exptname':filename, 'CHnames': CHname, 'blks':blks, 'dec_stims': decision_stims, 
        #            'DispList': dislist, 'StimList': stimlist, #'Xsacc': Xsacc, 'sacc_dirs': saccdirs, 
        #            'stimL': stimL, 'stimR':stimR, 'Robs':Robs, 'used_inds': used_inds}
    
        twin = np.arange(25,self.Nframes, dtype=np.int64)
        self.Rtr = np.zeros([self.Ntr, self.NC], dtype='float32')
        for ii in range(self.Ntr):
            self.Rtr[ii,:] = torch.sum(self.robs[twin+blks[ii,0], :], axis=0)

        print("%d frames, %d units, %d trials with %d frames each"%(self.NT, self.NC, self.Ntr, self.Nframes))
    
        # Generate cross-validation
        use_random = False
        # Cued and uncued trials
        trC = np.where(self.TRcued > 0)[0]
        trU = np.where(self.TRcued < 0)[0]
        # zero-strength trials
        tr0 = np.where(self.TRstrength[:,0] == 0)[0]
        # sort by cued/uncued
        tr0C = np.where((self.TRstrength[:,0] == 0) & (self.TRcued > 0))[0]
        tr0U = np.where((self.TRstrength[:,0] == 0) & (self.TRcued < 0))[0]
        # for purposes of cross-validation, do the same for non-zero-strength trials
        trXC = np.where((self.TRstrength[:,0] != 0) & (self.TRcued > 0))[0]
        trXU = np.where((self.TRstrength[:,0] != 0) & (self.TRcued < 0))[0]

        # Assign train and test indices sampled evenly from each subgroup (note using default 4-fold)
        Ut0C, Xt0C = self.train_test_assign( tr0C, use_random=use_random )
        Ut0U, Xt0U = self.train_test_assign( tr0U, use_random=use_random )
        UtXC, XtXC = self.train_test_assign( trXC, use_random=use_random )
        UtXU, XtXU = self.train_test_assign( trXU, use_random=use_random )

        # Putting together for larger groups
        Ut0 = np.sort( np.concatenate( (Ut0C, Ut0U), axis=0 ) )
        Xt0 = np.sort( np.concatenate( (Xt0C, Xt0U), axis=0 ) )
        UtC = np.sort( np.concatenate( (Ut0C, UtXC), axis=0 ) )
        XtC = np.sort( np.concatenate( (Xt0C, XtXC), axis=0 ) )
        UtU = np.sort( np.concatenate( (Ut0U, UtXU), axis=0 ) )
        XtU = np.sort( np.concatenate( (Xt0U, XtXU), axis=0 ) )

        Ut = np.sort( np.concatenate( (Ut0, UtXC, UtXU), axis=0 ) )
        Xt = np.sort( np.concatenate( (Xt0, XtXC, XtXU), axis=0 ) )
        
        self.trs = {'c':trC, 'u':trU, '0':tr0, '0c': tr0C, '0u': tr0U}
        self.Utr = {'all':Ut, '0':Ut0, 'c':UtC, 'u':UtU, '0c':Ut0C, '0u':Ut0U}
        self.Xtr = {'all':Xt, '0':Xt0, 'c':XtC, 'u':XtU, '0c':Xt0C, '0u':Xt0U}

        # Additional processing check
        # Cued and uncued stim
        #Cstim = np.multiply(TRstim[:,1], np.sign(TRstim[:,0])) # Cued stim
        #Ustim = np.multiply(TRstim[:,3], np.sign(TRstim[:,2]))  # Uncued stim
        #f_far = np.zeros([Nstim,2])
        #for nn in range(Nstim):
        #    tr1 = np.where(Cstim == stimlist[nn])[0]
        #    tr2 = np.where(Ustim == stimlist[nn])[0]
        #    f_far[nn,0] = np.sum(TRchoice[tr1] > 0)/len(tr1)
        #    f_far[nn,1] = np.sum(TRchoice[tr2] > 0)/len(tr2)

        # Prepare stimulus using input argument 'which_stim'
        self.prepare_stim( which_stim=which_stim )

        # Make drift-design matrix using anchor points at each cycle
        cued_transitions = np.where(abs(np.diff(self.TRcued)) > 0)[0]
        anchors = [0] + list(cued_transitions[range(1,len(cued_transitions)+1, 2)])
        self.construct_drift_design_matrix(block_anchors = anchors) 
    # END HNdata.__init__()

    def prepare_stim( self, which_stim='left', num_lags=None ):
        if which_stim in ['left', 'L', 'Left']:
            stim = torch.tensor( self.stimL, dtype=torch.float32 )
        else:
            stim = torch.tensor( self.stimR, dtype=torch.float32 )

        # Zero out invalid time points (disp=0) before time embedding
        df_generic = torch.zeros( stim.shape, dtype=torch.float32 )
        df_generic[self.used_inds, :] = 1.0
        stim = stim * df_generic
        
        self.stim_dims = [1, stim.shape[1], 1, 1]  # Put one-hot on first spatial dimension

        if num_lags is None:
            # then read from dataset (already set):
            num_lags = self.num_lags

        self.stim = self.time_embedding( stim=stim, nlags=num_lags )
        # This will return a torch-tensor
    # END HNdata.prepare_stim()

    @staticmethod
    def train_test_assign( trial_ns, fold=4, use_random=True ):  # this should be a static function
        num_tr = len(trial_ns)
        if use_random:
            permu = np.random.permutation(num_tr)
            xtr = np.sort( trial_ns[ permu[range(np.floor(num_tr/fold).astype(int))] ]) 
            utr = np.sort( trial_ns[ permu[range(np.floor(num_tr/fold).astype(int), num_tr)] ])
        else:
            xlist = np.arange(fold//2, num_tr, fold, dtype='int32')
            ulist = np.setdiff1d(np.arange(num_tr), xlist)
            xtr = trial_ns[xlist]
            utr = trial_ns[ulist]
        return utr, xtr
    # END HNdata.train_test_assign()

    @staticmethod
    def channel_list_scrub( fnames, subset=None, display_names=True ):  # This should also be a static function
        chnames = []
        if subset is None:
            subset = list(range(len(fnames)))
        for nn in subset:
            fn = fnames[nn]
            a = fn.find('c')   # finds the 'ch'
            b = fn.find('s')-1  # finds the 'sort'
            chn = deepcopy(fn[a:b])
            chnames.append(chn)
        if display_names:
            print(chnames)
        return chnames

    def __getitem__(self, idx):

        if len(self.cells_out) == 0:
            out = {'stim': self.stim[idx, :],
                'robs': self.robs[idx, :],
                'dfs': self.dfs[idx, :]}
        else:
            assert isinstance(self.cells_out, list), 'cells_out must be a list'
            robs_tmp =  self.robs[:, self.cells_out]
            dfs_tmp =  self.dfs[:, self.cells_out]
            out = {'stim': self.stim[idx, :],
                'robs': robs_tmp[idx, :],
                'dfs': dfs_tmp[idx, :]}
            
        if self.Xdrift is not None:
            out['Xdrift'] = self.Xdrift[idx, :]

        return out
    # END HNdata.__getitem()
    