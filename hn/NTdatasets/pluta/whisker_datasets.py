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

class WhiskerData(SensoryBase):

    def __init__(self, expt_name=None, hemi=0, num_lags=30, **kwargs):

        assert expt_name is not None, "Must specify expt_name, which is the directory name within the datadir."
        # call parent constructor
        super().__init__(filenames=expt_name, num_lags=num_lags, **kwargs)
        #self.expt_name = expt_name

        ExptMat = sio.loadmat( self.datadir+expt_name+'/ExpInfo.mat')['ExpInfo_dat']
        NeurMat = sio.loadmat( self.datadir+expt_name+'/NeuralData.mat')['NeuralData'].astype(np.float32)
        CellLocs = sio.loadmat( self.datadir+expt_name+'/Location.mat')['Location']

        NT = ExptMat.shape[0]
        self.NT = NT

        # Read basic properties
        pistons = ExptMat[:, range(1,5)]
        self.touchfull = ExptMat[:, range(5, 9)]
        self.angles = ExptMat[:, range(9, 13)]
        self.curves = ExptMat[:, range(13, 17)]
        self.phases = ExptMat[:, range(17, 21)]
        behavior = ExptMat[:, range(22,26)]
        self.run_speed = ExptMat[:, 21]
        self.licks = ExptMat[:, 26]

        # Make trial-blocks and throw in used_inds
        self.used_inds = np.ones(NT)
        TRinds = self.trial_parse(ExptMat[:,0])
        Ntr = TRinds.shape[0]
        for bb in range(Ntr):
            self.block_inds.append(np.arange(TRinds[bb,0], TRinds[bb,1], dtype=np.int64))
            self.used_inds[TRinds[bb,0]+np.arange(num_lags)] = 0  # Zero out beginning of every trial

        # Process locations
        self.num_cells, self.electrode_info = self.process_locations( CellLocs )
        self.NC = np.sum(self.num_cells)
        self.Rparse = [list(np.arange(self.num_cells[0])), list(np.arange(self.num_cells[0], self.NC))]

        # Process neurons 
        assert NeurMat.shape[1]-1 == self.NC, "Cell count problem"

        self.robs = torch.tensor( NeurMat[:, 1:], dtype=torch.float32, device=self.device )
        self.dfs = torch.ones([NT, self.NC], dtype=torch.float32, device=self.device) * self.used_inds[:, None]

        self.cells_in = []
        self.set_hemispheres(out_config=hemi, in_config=2)   # default settings

        # Assign XV indices
        Xtr = np.arange(2, Ntr, 5, dtype='int64')
        Utr = np.array(list(set(np.arange(Ntr, dtype='int64'))-set(Xtr)))
        Ui, Xi, used_inds = np.zeros(0, dtype='int64'), np.zeros(0, dtype='int64'), np.zeros(0, dtype='int64')
        for tr in Utr:
            Ui = np.concatenate( (Ui, self.block_inds[tr]), axis=0)
        for tr in Xtr:
            Xi = np.concatenate( (Xi, self.block_inds[tr]), axis=0)

        ##### Additional Stim processing #####
        self.touches = np.zeros([NT, 4])
        for ww in range(4):
            self.touches[np.where(np.diff(self.touchfull[:, ww]) > 0)[0]+1, ww] = 1

        self.TRpistons, self.TRoutcomes = self.trial_classify(TRinds, pistons, behavior)
        self.TRhit = np.where(self.TRoutcomes == 1)[0]
        self.TRmiss = np.where(self.TRoutcomes == 2)[0]
        self.TRfpos = np.where(self.TRoutcomes == 3)[0]
        self.TRcrej = np.where(self.TRoutcomes == 4)[0]
        self.TRuni = np.where(self.TRoutcomes == 5)[0]
        print("Hits: %d\tMisses: %d\nFalse Pos: %d\tCorrect rej %d\nUnilateral stim: %d"%(len(self.TRhit), 
                                                                                        len(self.TRmiss), 
                                                                                        len(self.TRfpos), 
                                                                                        len(self.TRcrej),
                                                                                        len(self.TRuni)))
        # Make drift matrix
        self.construct_drift_design_matrix() 

        # Configure stimulus  # default is just touches (onset)
        self.prepare_stim()
    # END WhiskerData.__init__()

    def prepare_stim( self, stim_config=0, num_lags=None ):

        #self.stim = torch.tensor( self.touches, dtype=torch.float32, device=device )
        if num_lags is None:
            num_lags = self.num_lags

        self.stim_dims = [2, 1, 1, 1]
        if stim_config == 0:
            self.stim = self.time_embedding( stim=self.touches[:, :2], nlags=num_lags )
            self.stimA = self.time_embedding( stim=self.touches[:, 2:], nlags=num_lags )
        elif stim_config == 1:
            self.stimA = self.time_embedding( stim=self.touches[:, :2], nlags=num_lags )
            self.stim = self.time_embedding( stim=self.touches[:, 2:], nlags=num_lags )
        else:
            self.stim_dims = [4, 1, 1, 1]
            self.stim = self.time_embedding( stim=self.touches, nlags=num_lags )
            self.stimA = None
    # END WhiskerData.prepare_stim()

    def set_hemispheres( self, out_config=0, in_config=0 ):
        """This sets cells_out and cells_in based on hemisphere, making things easy. 
        Can also set cells_out and cells_in by hand.
        
        out_config, in_config: 0=left outputs, 1=right outputs, 2=both"""
        if out_config < 2:
            self.cells_out = self.Rparse[out_config]
        else:
            self.cells_out = []

        if in_config < 2:
            self.cells_in = self.Rparse[in_config]
        else:
            self.cells_in = []
    # END WhiskerData.set_hemispheres

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
            
        if self.stimA is not None:
            out['stimA'] = stimA[idx, :]

        if self.Xdrift is not None:
            out['Xdrift'] = self.Xdrift[idx, :]

        return out
    # END WhiskerData.__getitem()

    def WTAs( self, r0=5, r1=30):
        """
        Inputs: 
            Ton: list of touch onsets for all 4 whiskers
            Rs: Robs 
            r0, r1: how many lags before and after touch onset to include (and block out)
        Output:
            wtas: whisker-triggered averages of firing rate
            nontouchFRs: average firing rate (spike prob) away from all four whisker touches
            """
        L = r1+r0
        wtas = np.zeros([L,4, self.NC])
        wcounts = deepcopy(wtas)
        #valws = np.zeros([NT,4])
        nontouchFRs = np.zeros([5, self.NC])
        for ww in range(4):
            #valws[val_inds.astype(int),ww] = 1.0
            wts = np.where(self.touches[:, ww] > 0)[0]
            print( "w%d: %d touches"%(ww+1, len(wts)))
            for tt in wts:
                t0 = np.maximum(tt-r0,0)
                t1 = np.minimum(tt+r1, self.NT)
                if t1-t0 == L: # then valid event
                    footprint = np.expand_dims(valws[range(t0,t1), ww], 1) # All touches probably valid, but just in case
                    wcounts[:,ww,:] += footprint
                    wtas[:,ww,:] += Rs[range(t0,t1),:]*footprint
                #valws[range(t0,t1),ww] = 0
            wtas[:,ww,:] = wtas[:,ww,:] / wcounts[:,ww,:] 
            #nontouchFRs[ww,:] = np.sum(valws[:,[ww]]*Rs,axis=0) /np.sum(valws[:,ww])

        # Stats where there are no touches from any whisker
        #valtot = np.expand_dims(np.prod(valws, axis=1), 1)
        #nontouchFRs[4,:] = np.sum(valtot*Rs,axis=0)/np.sum(valtot)
        
        return wtas, nontouchFRs

    @staticmethod
    def create_NLmap_design_matrix( x, num_bins, val_inds=None, thresh=5, 
                                borderL=None, borderR=None, anchorL=True, rightskip=False):
        """Make design matrix of certain number of bins that maps variable of interest
        anchorL is so there is not an overall bias fit implicitly"""
        NT = x.shape[0]
        if val_inds is None:
            val_inds = range(NT)    
        #m = np.mean(x[val_inds])
        # Determine 5% and 95% intervals (related to thresh)
        h, be = np.histogram(x[val_inds], bins=100)
        h = h/np.sum(h)*100
        cumu = 0
        if borderL is None:
            borderL = np.nan
        if borderR is None:
            borderR = np.nan
        for nn in range(len(h)):
            cumu += h[nn]
            if np.isnan(borderL) and (cumu >=  thresh):
                borderL = be[nn]
            if np.isnan(borderR) and (cumu >= 100-thresh):
                borderR = be[nn]
        # equal divisions between 95-max
        if rightskip:
            bins = np.arange(num_bins)*(borderR-borderL)/num_bins + borderL
        else:
            bins = np.arange(num_bins+1)*(borderR-borderL)/num_bins + borderL
        print(bins)
        XNL = NDNutils.design_matrix_tent_basis( x, bins, zero_left=anchorL )
        return XNL

    @staticmethod
    def find_first_locmin(trace, buf=0, sm=0):
        der = np.diff(trace)
        loc = np.where(np.diff(trace[buf:]) >= 0)[0][0]+buf
        return loc

    @staticmethod
    def prop_distrib(events, prop_name):
        assert prop_name in events[0], 'Invalid property name.'
        distrib = np.zeros(len(events))
        for tt in range(len(events)):
            distrib[tt] = events[tt][prop_name]
        return distrib

    @staticmethod
    def trial_parse( frames ):
        trial_starts = np.where(frames == 1)[0]
        num_trials = len(trial_starts)
        blks = np.zeros([num_trials, 2], dtype='int64')
        for nn in range(num_trials-1):
            blks[nn, :] = [trial_starts[nn], trial_starts[nn+1]]
        blks[-1, :] = [trial_starts[-1], len(frames)]
        return blks

    @staticmethod
    def trial_classify( blks, pistons, outcomes=None ):
        """outcomes: 1=hit, 2=miss, 3=false alarm, 4=correct reject"""
        #assert pistons is not None, "pistons cannot be empty"
        Ntr = blks.shape[0]
        #= np.mean(blk_inds, axis=1).astype('int64')
        TRpistons = np.zeros(Ntr, dtype='int64')
        TRoutcomes = np.zeros(Ntr, dtype='int64')
        for nn in range(Ntr):
            ps = np.median(pistons[range(blks[nn,0], blks[nn,1]),:], axis=0)
            TRpistons[nn] = (ps[0] + 2*ps[1] + 4*ps[2] + 8*ps[3]).astype('int64')
            if outcomes is not None:
                os = np.where(np.median(outcomes[range(blks[nn,0], blks[nn,1]),:], axis=0) > 0)[0]
                if len(os) != 1:
                    print("Warning: trial %d had unclear outcome."%nn)
                else:
                    TRoutcomes[nn] = os[0]+1
        # Reclassify unilateral (or no) touch trials as Rew = 5
        unilateral = np.where((TRpistons <=2) | (TRpistons == 4) | (TRpistons==8))[0]
        TRoutcomes[unilateral] = 5
        
        return(TRpistons, TRoutcomes)

    @staticmethod
    def process_locations( clocs ):
        """electrode_info: first column is shank membership, second column is electrode depth"""
        hemis = np.where(clocs[:,0] == 1)[0]
        NC = clocs.shape[0]
        if len(hemis) > 1:
            num_cells = [hemis[1], NC-hemis[1]]
        else:
            num_cells = [NC, 0]
        electrode_info = [None]*2
        for hh in range(len(hemis)):
            if hh == 0:
                crange = range(num_cells[0])
            else:
                crange = range(num_cells[0], NC)
            ei = clocs[crange, :]
            electrode_info[hh] = ei[:, 1:]
        return num_cells, electrode_info
        