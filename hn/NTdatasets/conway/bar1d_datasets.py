
import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import NDNT.utils as utils
#from NDNT.utils import download_file, ensure_dir
from copy import deepcopy
import h5py


class Bar1D(Dataset):
    """
    -- can load batches from multiple datasets
    -- hdf5 files must have the following information:
        Robs
        RobsMU
        stim: 4-d stimulus: time x nx x ny x color
        block_inds: start and stop of 'trials' (perhaps fixations for now)
        other things: saccades? or should that be in trials? 

    Constructor will take eye position, which for now is an input from data
    generated in the session (not on disk). It should have the length size 
    of the total number of fixations x1.
    """

    def __init__(self,
        sess_list,
        datadir, 
        # Stim setup
        num_lags=10, 
        #which_stim = 'stimET',
        stim_crop = None,
        Hstim_shift = 0,  # to shift-and-wrap stimH stimulus
        Hdiscardzero=True,
        time_embed = 2,  # 0 is no time embedding, 1 is time_embedding with get_item, 2 is pre-time_embedded
        folded_lags=False,
        # Stim configuation
        combine_stim = False, # if False, horizontal stim will be 'stim'
        stim_gap = 8,
        drift_interval = 16,  # how many trials to anchor each drift term
        # eye configuration
        eye_config = 2,  # 0 = all, 1, -1, and 2 are options (2 = binocular)
        # other
        ignore_saccades = True,
        include_MUs = False,
        preload = True,
        eyepos = None, 
        device=torch.device('cpu')):
        """Constructor options"""

        self.datadir = datadir
        self.sess_list = sess_list
        self.device = device
        
        self.num_lags = num_lags
        if time_embed == 2:
            assert preload, "Cannot pre-time-embed without preloading."
        self.time_embed = time_embed
        self.preload = preload
        self.stim_crop = stim_crop
        self.folded_lags = folded_lags
        self.combine_stim = combine_stim
        self.eye_config = eye_config
        self.stim_gap = stim_gap
        self.stimHshift = Hstim_shift
        self.Hdiscardzero = Hdiscardzero
        self.drift_interval = drift_interval

        # get hdf5 file handles
        self.fhandles = [h5py.File(os.path.join(datadir, sess + '.mat'), 'r') for sess in self.sess_list]
        self.dfs_orig = None

        # Data to just read in and store
        self.ETstim_location = np.array(self.fhandles[0]['ETstim_location'], dtype=int)
        self.fix_location = np.array(self.fhandles[0]['fix_location'], dtype=int)[:, 0]
        self.probeIDs = np.array(self.fhandles[0]['Robs_probe_ID'], dtype=int)[0,:]-1
        self.probeIDsMU = np.array(self.fhandles[0]['RobsMU_probe_ID'], dtype=int)[0,:]-1
        self.dt = np.array(self.fhandles[0]['dt'], dtype=np.float32)[0, 0]
        self.pixel_size = np.array(self.fhandles[0]['pixel_size'], dtype=int)[0, 0]
        self.exptdate = [] # need to learn matlab strings in HDF5 format (not saved as HDF5 string)

        # build index map
        self.data_threshold = 6  # how many valid time points required to include saccade?
        self.file_index = [] # which file the block corresponds to
        self.sacc_inds = []
        #self.unit_ids = []
        self.num_units, self.num_sus, self.num_mus = [], [], []
        self.sus = []
        self.NC = 0
        #self.stim_dims = None
        self.eyepos = eyepos
        self.generate_Xfix = False
        self.num_blks = np.zeros(len(sess_list), dtype=int)
        self.block_inds = []
        self.block_filemapping = []
        self.include_MUs = include_MUs
        self.SUinds = []
        self.MUinds = []
        self.cells_out = []  # can be list to output specific cells in get_item
        self.avRs = None
        self.with_shifter = False
        self.shifts = None

        # Set up to store default train_, val_, test_inds
        self.test_inds = None
        self.val_inds = None
        self.train_inds = None
        self.train_blks, self.val_blks, self.test_blks = None, None, None

        # Data to construct and store in memory
        self.fix_n = []
        #self.sacc_on = []
        #self.sacc_off = []
        self.used_inds = []

        tcount, fix_count = 0, 0

        #if which_stim is None:
        #    stimname = 'stimET'
        #else:
        #    stimname = which_stim
        self.stimname = 'stimET'
        self.ETtraceHR = []

        for fnum, fhandle in enumerate(self.fhandles):

            NT, NSUfile = fhandle['Robs'].shape
            NX = fhandle['stimET'].shape[1]
            NMUfile = fhandle['RobsMU'].shape[1]
            self.num_sus.append(NSUfile)
            self.num_mus.append(NMUfile)
            self.sus = self.sus + list(range(self.NC, self.NC+NSUfile))
            blk_inds = np.array(fhandle['block_inds'], dtype=np.int64).T
            stim_eyes = np.array(fhandle['useLeye'], dtype=int) +  \
                            2*np.array(fhandle['useReye'], dtype=int)
            
            Reye = fhandle['useReye']
            NCfile = NSUfile
            if self.include_MUs:
                NCfile += NMUfile
            
            # This will associate each block with each file
            self.block_filemapping += list(np.ones(blk_inds.shape[0], dtype=int)*fnum)
            self.num_blks[fnum]= blk_inds.shape[0]
            self.num_blks[fnum] = 0
            #if self.stim_dims is None:
            #if folded_lags:
            self.dims = [1] + list([fhandle[self.stimname].shape[1]]) + [1, 1] #list(fhandle[stimname].shape[1:4]) + [1]
            self.NX = self.dims[1]
          
            if self.time_embed > 0:
                self.dims[3] = self.num_lags

            print('Stim check:', self.stimname, folded_lags, self.dims)

            #self.unit_ids.append(self.NC + np.asarray(range(NCfile)))
            self.num_units.append(NCfile)
            self.NC += NCfile

            self.sacc_inds = np.array(fhandle['sacc_inds'], dtype=np.int64)
            fix_n = np.zeros(NT, dtype=np.int64)  # each time point labels fixation number
            #sacc_on = np.zeros(NT, dtype=np.float32)  # each time point labels fixation number
            #sacc_on[sacc_inds[:,0]-1] = 1.0
            #sacc_off = np.zeros(NT, dtype=np.float32)  # each time point labels fixation number
            #sacc_off[sacc_inds[:,1]-1] = 1.0

            valid_inds = np.array(fhandle['valid_data'], dtype=np.int64)[0,:]-1  #range(self.NT)  # default -- to be changed at end of init

            ### Process blocks and fixations/saccades
            for ii in range(blk_inds.shape[0]):
                # note this will be the inds in each file -- file offset must be added for mult files
                include_block = True
                ts = np.arange( blk_inds[ii,0]-1, blk_inds[ii,1], dtype=int)
                if self.eye_config > 0:
                    if np.sum(stim_eyes[ts] == self.eye_config) == 0:
                        include_block = False
                if include_block:
                    self.block_inds.append(deepcopy(ts))
                    #self.block_inds.append( np.arange( blk_inds[ii,0], blk_inds[ii,1], dtype=np.int64) )

            # Got through each block to segment into fixations
            #for nn in range(blk_inds.shape[0]):
            #    # note this will be the inds in each file -- file offset must be added for mult files
            #    include_block = True
            #    ts = np.arange( blk_inds[nn,0]-1, blk_inds[nn,1], dtype=int)
            #    if self.eye_config > 0:
            #        if np.sum(stim_eyes[ts] == self.eye_config) == 0:
            #            include_block = False
            #    if include_block:
            #        self.block_inds.append(deepcopy(ts))
            #        t0 = blk_inds[nn, 0]-1
                    #valid_inds[range(t0, t0+num_lags)] = 0  # this already adjusted for?

                    # Parse fixation numbers within block
            #        if not ignore_saccades:
            #            rel_saccs = np.where((sacc_inds[:,0] > t0) & (sacc_inds[:,0] < blk_inds[nn,1]))[0]
            #            for mm in range(len(rel_saccs)):
            #                fix_count += 1
            #                fix_n[ range(t0, sacc_inds[rel_saccs[mm], 0]) ] = fix_count
            #                t0 = sacc_inds[rel_saccs[mm], 1]-1
                    # Put in last (or only) fixation number
            #        if t0 < blk_inds[nn, 1]:
            #            fix_count += 1
            #            fix_n[ range(t0, blk_inds[nn, 1]) ] = fix_count
            
            tcount += NT
            # make larger fix_n, valid_inds, sacc_inds, block_inds as self

        self.NT  = tcount

        # For now let's just debug with one file
        if len(sess_list) > 1:
            print('Warning: currently ignoring multiple files')
        self.used_inds = deepcopy(valid_inds)
        #self.fix_n = deepcopy(fix_n)
        #self.sac_on = deepcopy(sacc_on)
        #self.sac_off = deepcopy(sacc_off)
        #self.sacc_inds = deepcopy(sacc_inds)
        
        # PULL OTHER INFO
        self.ETtraceHR = np.array(fhandle['ETtrace_raw'], dtype=np.float32)
        self.sort_rating = np.array(fhandle['Robs_rating'], dtype=np.int64)[0, :]
        self.sort_ratingMU = np.array(fhandle['RobsMU_rating'], dtype=np.int64)[0,:]

        # Go through saccades to establish val_indices and produce saccade timing vector 
        # Note that sacc_ts will be generated even without preload -- small enough that doesnt matter
    #    self.sacc_ts = np.zeros([self.NT, 1], dtype=np.float32)
    #    self.fix_n = np.zeros(self.NT, dtype=np.int64)  # label of which fixation is in each range
    #    for nn in range(self.num_fixations):
    #        print(nn, self.sacc_inds[nn][0], self.sacc_inds[nn][1] )
    #        self.sacc_ts[self.sacc_inds[nn][0]] = 1 
    #        self.fix_n[range(self.sacc_inds[nn][0], self.sacc_inds[nn][1])] = nn
        #self.fix_n = list(self.fix_n)  # better list than numpy

        #self.dims = np.unique(np.asarray(self.dims)) # assumes they're all the same    
        #if self.eyepos is not None:
            #assert len(self.eyepos) == self.num_fixations, \
            #    "eyepos input should have %d fixations."%self.num_fixations

        if preload:
            print("Loading data into memory...")
            self.preload_numpy()

            if self.eyepos is not None:
                assert (self.eyepos.shape[0] == self.NT) and (self.eyepos.shape[1] == 2), "eye pos wrong shape"
                # Apply horizontal shifts to stim (stil in numpy)
                Hshifts = np.unique(eyepos[:,0])

                newstim = deepcopy(self.stimH)
                for ss in range(len(Hshifts)):
                    sh = Hshifts[ss]
                    Lc = NX-abs(sh)
                    ts = np.where(eyepos[:,0] == sh)[0]
                    tmp = np.zeros([len(ts), self.NX])
                    if sh < 0:
                        tmp[:, range(Lc)] = deepcopy(self.stimH[ts, :][:, range(abs(sh), self.NX)])
                        tmp[:, range(Lc, self.NX)] = deepcopy(self.stimH[ts,:][:, range(abs(sh))])
                    elif sh > 0:
                        tmp[:, range(sh, self.NX)] = deepcopy(self.stimH[ts,:][:, range(Lc)])
                        tmp[:, range(sh)] = deepcopy(self.stimH[ts,:][:, range(Lc, self.NX)])
                    if sh != 0:
                        newstim[ts, :] = deepcopy(tmp)
                self.stimH = deepcopy(newstim)

                # Apply horizontal shifts to stim (stil in numpy)
                Vshifts = np.unique(eyepos[:,1])
                newstim = deepcopy(self.stimV)
                for ss in range(len(Vshifts)):
                    sh = Vshifts[ss]
                    Lc = NX-abs(sh)
                    ts = np.where(eyepos[:,1] == sh)[0]
                    tmp = np.zeros([len(ts), self.NX])
                    if sh < 0:
                        tmp[:, range(Lc)] = deepcopy(self.stimV[ts, :][:, range(abs(sh), self.NX)])
                        tmp[:, range(Lc, self.NX)] = deepcopy(self.stimV[ts,:][:, range(abs(sh))])
                    elif sh > 0:
                        tmp[:, range(sh, self.NX)] = deepcopy(self.stimV[ts,:][:, range(Lc)])
                        tmp[:, range(sh)] = deepcopy(self.stimV[ts,:][:, range(Lc, self.NX)])
                    if sh != 0:
                        newstim[ts,:] = tmp
                self.stimV = deepcopy(newstim)
    
            if self.stim_crop is not None:
                print('stimulus cropping not implemented yet')

            if time_embed == 2:
                print("Time embedding...")
                idx = np.arange(self.NT)
                tmp_stimH = self.stimH[np.arange(self.NT)[:,None]-np.arange(num_lags), ...]
                tmp_stimV = self.stimV[np.arange(self.NT)[:,None]-np.arange(num_lags), ...]
                if self.combine_stim:
                    tmp_stimC = self.stimC[np.arange(self.NT)[:,None]-np.arange(num_lags), ...]

                if self.folded_lags:
                    self.stimH = tmp_stimH
                    self.stimV = tmp_stimV
                    if self.combine_stim:
                        self.stimC = tmp_stimC 
                    print("Folded lags: stim-dim = ", self.stim.shape)
                else:
                    self.stimH = np.transpose( tmp_stimH, axes=[0,2,1] )
                    self.stimV = np.transpose( tmp_stimV, axes=[0,2,1] )
                    if self.combine_stim:
                        self.stimC = np.transpose( tmp_stimC, axes=[0,2,1] )

            # Flatten stim 
            self.stimV = np.reshape(self.stimV, [self.NT, -1])
            self.stimH = np.reshape(self.stimH, [self.NT, -1])
            if self.combine_stim:
                self.stimC = np.reshape(self.stimC, [self.NT, -1])

            # Have data_filters represend used_inds (in case it gets through)
            unified_df = np.zeros([self.NT, 1], dtype=np.float32)
            unified_df[self.used_inds] = 1.0
            self.dfs *= unified_df

            # Prepare design matrix for drift (on trial-by-trial basis)
            #self.Xdrift = np.zeros([self.NT, len(self.block_inds)//], dtype=np.float32)
            #for nn in range(len(self.block_inds)):
            #    self.Xdrift[self.block_inds[nn], nn] = 1.0
            NBL = len(self.block_inds)
            Nanchors = NBL//self.drift_interval
            anchors = np.zeros(Nanchors, dtype=np.int64)
            for bb in range(Nanchors):
                anchors[bb] = self.block_inds[self.drift_interval*bb][0]
            self.Xdrift = utils.design_matrix_drift( self.NT, anchors, zero_left=False, const_right=True)

            self.process_fixations()

            # Convert data to tensors
            #if self.device is not None:
            self.to_tensor(self.device)
            print("Done")

        # Create valid indices and first pass of cross-validation indices
        #self.create_valid_indices()
        # Develop default train, validation, and test datasets 
        self.crossval_setup()

        # Fix Xdrift to remove test and validation sets
        #self.adjust_Xdrift_xval()

        if self.combine_stim:
            self.dims[1] = self.NX*2 + stim_gap

        # Reflects block structure
        #vblks, trblks = self.fold_sample(len(self.block_inds), 5, random_gen=True)
        #self.train_inds = []
        #for nn in trblks:
        #    self.train_inds += list(deepcopy(self.block_inds[nn]))
        #self.val_inds = []
        #for nn in vblks:
        #    self.val_inds += list(deepcopy(self.block_inds[nn]))
        #self.train_inds = np.array(self.train_inds, dtype=np.int64)
        #self.val_inds = np.array(self.val_inds, dtype=np.int64)
    # END ColorClouds.__init__

    def preload_numpy(self):
        """Note this loads stimulus but does not time-embed"""

        NT = self.NT
        NX = self.dims[1]
        NX2 = NX*2 + self.stim_gap

        ''' 
        Pre-allocate memory for data
        '''
        self.stimH = np.zeros( [NT] + [NX], dtype=np.float32)
        self.stimV = np.zeros( [NT] + [NX], dtype=np.float32)
        if self.combine_stim:
            self.stimC = np.zeros( [NT] + [NX2], dtype=np.float32)
        else:
            self.stimC = None

        self.robs = np.zeros( [NT, self.NC], dtype=np.float32)
        self.dfs = np.ones( [NT, self.NC], dtype=np.float32)
        #self.eyepos = np.zeros([NT, 2], dtype=np.float32)
        #self.frame_times = np.zeros([NT,1], dtype=np.float32)

        t_counter = 0
        unit_counter = 0
        for ee in range(len(self.fhandles)):
            
            fhandle = self.fhandles[ee]
            sz = fhandle[self.stimname].shape
            inds = np.arange(t_counter, t_counter+sz[0], dtype=np.int64)
            #inds = self.stim_indices[expt][stim]['inds']
            #self.stim[inds, ...] = np.transpose( np.array(fhandle[self.stimname], dtype=np.float32), axes=[0,3,1,2])
            tmp_stim = np.array(fhandle[self.stimname], dtype=np.float32)
            self.stimH[inds, :] = deepcopy(tmp_stim)
            self.stimV[inds, :] = tmp_stim
            stim_ori = np.array(fhandle['stimETori'], dtype=np.int32)[:, 0]
            Hinds = np.where(stim_ori > 45)[0]
            Vinds = np.where(stim_ori < 45)[0]

            self.stimH[inds[Vinds], ...] = 0.0
            self.stimV[inds[Hinds], ...] = 0.0

            """ Robs and DATAFILTERS"""
            robs_tmp = np.zeros( [len(inds), self.NC], dtype=np.float32 )
            dfs_tmp = np.zeros( [len(inds), self.NC], dtype=np.float32 )
            num_sus = fhandle['Robs'].shape[1]
            units = range(unit_counter, unit_counter+num_sus)
            robs_tmp[:, units] = np.array(fhandle['Robs'], dtype=np.float32)
            dfs_tmp[:, units] = np.array(fhandle['datafilts'], dtype=np.float32)
            if self.include_MUs:
                num_mus = fhandle['RobsMU'].shape[1]
                units = range(unit_counter+num_sus, unit_counter+num_sus+num_mus)
                robs_tmp[:, units] = np.array(fhandle['RobsMU'], dtype=np.float32)
                dfs_tmp[:, units] = np.array(fhandle['datafiltsMU'], dtype=np.float32)
            
            self.robs[inds,:] = deepcopy(robs_tmp)
            self.dfs[inds,:] = deepcopy(dfs_tmp)

            t_counter += sz[0]
            unit_counter += self.num_units[ee]

        self.stimname = 'stim'  # not sure what this does

        if self.combine_stim:
            self.stimC[:, range(NX)] = deepcopy(self.stimH)
            self.stimC[:, range(NX+self.stim_gap, NX2)] = deepcopy(self.stimV)

        sh = abs(self.stimHshift)
        if sh > 0:
            Lc = self.NX - sh
            tmp_stim = np.zeros(self.stimH.shape, dtype=np.float32)
            if self.Hdiscardzero:
                print('  Discarding stim at 0 horizontal position')
                if self.stimHshift < 0:
                    tmp_stim[:, range(Lc)] = deepcopy(self.stimH[:, range(sh, self.NX)])
                    tmp_stim[:, range(Lc, self.NX-1)] = deepcopy(self.stimH[:, range(1,sh)])
                elif self.stimHshift > 0:
                    tmp_stim[:, range(sh, self.NX-1)] = deepcopy(self.stimH[:, range(1,Lc)])
                    tmp_stim[:, range(sh)] = deepcopy(self.stimH[:, range(Lc, self.NX)])
            else:
                if self.stimHshift < 0:
                    tmp_stim[:, range(Lc)] = deepcopy(self.stimH[:, range(sh, self.NX)])
                    tmp_stim[:, range(Lc, self.NX)] = deepcopy(self.stimH[:, range(sh)])
                elif self.stimHshift > 0:
                    tmp_stim[:, range(sh, self.NX)] = deepcopy(self.stimH[:, range(Lc)])
                    tmp_stim[:, range(sh)] = deepcopy(self.stimH[:, range(Lc, self.NX)])

            self.stimH = deepcopy(tmp_stim)
    # END Bar1D.preload_numpy()
        
    def to_tensor(self, device):
        if isinstance(self.stimH, torch.Tensor):
            # then already converted: just moving device
            self.stimH = self.stimH.to(device)
            self.stimV = self.stimV.to(device)
            self.robs = self.robs.to(device)
            self.dfs = self.dfs.to(device)
            self.Xdrift = self.Xdrift.to(device)
            self.fix_n = self.fix_n.to(device)
            if self.combine_stim:
                self.stimC = self.stimC.to(device)

        else:
            self.stimH = torch.tensor(self.stimH, dtype=torch.float32, device=device)
            self.stimV = torch.tensor(self.stimV, dtype=torch.float32, device=device)
            self.robs = torch.tensor(self.robs, dtype=torch.float32, device=device)
            self.dfs = torch.tensor(self.dfs, dtype=torch.float32, device=device)
            self.fix_n = torch.tensor(self.fix_n, dtype=torch.int64, device=device)
            self.Xdrift = torch.tensor(self.Xdrift, dtype=torch.float32, device=device)
            if self.combine_stim:
                self.stimC = torch.tensor(self.stimC, dtype=torch.float32, device=device)
    # END Bar1D.to_tensor

    def process_fixations( self, sacc_in=None ):
        """Processes fixation informatiom from dataset, but also allows new saccade detection
        to be input and put in the right format within the dataset (main use)"""
        if sacc_in is None:
            sacc_in = self.sacc_inds[:, 0]

        fix_n = np.zeros(self.NT, dtype=np.int64) 
        fix_count = 0
        for ii in range(len(self.block_inds)):
            # note this will be the inds in each file -- file offset must be added for mult files
            #self.block_inds.append(np.arange( blk_inds[ii,0], blk_inds[ii,1], dtype=np.int64))

            # Parse fixation numbers within block
            rel_saccs = np.where((sacc_in > self.block_inds[ii][0]+6) & (sacc_in < self.block_inds[ii][-1]))[0]

            tfix = self.block_inds[ii][0]  # Beginning of fixation by definition
            for mm in range(len(rel_saccs)):
                fix_count += 1
                # Range goes to beginning of next fixation (note no gap)
                fix_n[ range(tfix, sacc_in[rel_saccs[mm]]) ] = fix_count
                tfix = sacc_in[rel_saccs[mm]] + 6
            # Put in last (or only) fixation number
            if tfix < self.block_inds[ii][-1]:
                fix_count += 1
                fix_n[ range(tfix, self.block_inds[ii][-1]) ] = fix_count

        # Determine whether to be numpy or tensor
        if isinstance(self.robs, torch.Tensor):
            self.fix_n = torch.tensor(fix_n, dtype=torch.int64, device=self.robs.device)
        else:
            self.fix_n = fix_n
    # END: Bar1D.process_fixations()

    def add_shifter( self, shifts ):
        assert len(shifts == self.NT), "Entered shifts is wrong size."
        if not self.with_shifter:
            self.with_shifter = True
        # else: Delete old shifts?
        
        device = self.stim.device

        if isinstance( shifts, torch.tensor):
            self.shifts = shifts.to(device)
        else:
            self.shifts = torch.tensor( shifts, dtype=torch.float32, device=device)

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
                # then precalculated and do not need to do
                return self.avRs

        # Otherwise calculate across all data
        if self.preload:
            Reff = (self.dfs * self.robs).sum(dim=0).cpu()
            Teff = self.dfs.sum(dim=0).clamp(min=1e-6).cpu()
            return (Reff/Teff).detach().numpy()
        else:
            print('Still need to implement avRs without preloading')
            return None
    # END .avrates()

    def apply_data_mask( self, dmask, crange=None ):
        """For when data_filters (or a section) is modified based on models, and want to apply
        to the dataset. Ideally would save the original"""
        assert self.preload, "data must be preloaded to apply mask"
        if not isinstance(dmask, torch.Tensor):
            dmask = torch.tensor(dmask, dtype=torch.float32)
        if len(dmask.shape) == 1:
            dmask = dmask[:, None]
        NT, mask_size = dmask.shape
        assert NT == self.NT, "Data mask is wrong length"
        if len(self.cells_out) == 0:
            cells_out = np.arange(self.NC, dtype=int)
        else:
            cells_out = np.array(self.cells_out, dtype=int)
        if crange is None:
            crange = np.arange(len(cells_out), dtype=int)
        assert mask_size == len(crange), "mask does not cover the right range"

        if self.dfs_orig is None:
            self.dfs_orig = deepcopy(self.dfs)
        # Given all assertions worked, then make copy

        self.dfs.data[:, cells_out[crange]] = deepcopy(dmask)
    # END .apply_data_mask()

    def data_mask_revert( self, crange=None ):
        assert self.dfs_orig is not None, "Nothing to revert"
        self.dfs = deepcopy(self.dfs_orig)
        self.dfs_orig = None

    def shift_stim_fixation( self, stim, shift):
        """Simple shift by integer (rounded shift) and zero padded. Note that this is not in 
        is in units of number of bars, rather than -1 to +1. It assumes the stim
        has a batch dimension (over a fixation), and shifts the whole stim by the same amount."""
        print('Currently needs to be fixed to work with 2D')
        sh = round(shift)
        shstim = stim.new_zeros(*stim.shape)
        if sh < 0:
            shstim[:, -sh:] = stim[:, :sh]
        elif sh > 0:
            shstim[:, :-sh] = stim[:, sh:]
        else:
            shstim = deepcopy(stim)

        return shstim
    # END .shift_stim_fixation

    def create_valid_indices(self, post_sacc_gap=None):
        """
        This creates self.valid_inds vector that is used for __get_item__ 
        -- Will default to num_lags following each saccade beginning"""

        if post_sacc_gap is None:
            post_sacc_gap = self.num_lags

        # first, throw out all data where all data_filters are zero
        is_valid = np.zeros(self.NT, dtype=np.int64)
        is_valid[torch.sum(self.dfs, axis=1) > 0] = 1
        
        # Now invalid post_sacc_gap following saccades
        for nn in range(self.num_fixations):
            print(self.sacc_inds[nn, :])
            sts = self.sacc_inds[nn, :]
            is_valid[range(sts[0], np.minimum(sts[0]+post_sacc_gap, self.NT))] = 0
        
        #self.valid_inds = list(np.where(is_valid > 0)[0])
        self.valid_inds = np.where(is_valid > 0)[0]
    # END .create_valid_indices

    def crossval_setup(self, folds=5, random_gen=False, test_set=False, verbose=False):
        """This sets the cross-validation indices up We can add featuers here. Many ways to do this
        but will stick to some standard for now. It sets the internal indices, which can be read out
        directly or with helper functions. Perhaps helper_functions is the best way....
        
        Inputs: 
            random_gen: whether to pick random fixations for validation or uniformly distributed
            test_set: whether to set aside first an n-fold test set, and then within the rest n-fold train/val sets
        Outputs:
            None: sets internal variables test_inds, train_inds, val_inds
        """
        assert self.used_inds is not None, "Must first specify valid_indices before setting up cross-validation."

        # Reflect block structure
        Nblks = len(self.block_inds)
        val_blk1, tr_blk1 = self.fold_sample(Nblks, folds, random_gen=random_gen)

        if test_set:
            self.test_blks = val_blk1
            val_blk2, tr_blk2 = self.fold_sample(len(tr_blk1), folds, random_gen=random_gen)
            self.val_blks = tr_blk1[val_blk2]
            self.train_blks = tr_blk1[tr_blk2]
        else:
            self.val_blks = val_blk1
            self.train_blks = tr_blk1
            self.test_blks = []

        if verbose:
            print("Partitioned %d fixations total: tr %d, val %d, te %d"
                %(len(self.test_blks)+len(self.train_blks)+len(self.val_blks),len(self.train_blks), len(self.val_blks), len(self.test_blks)))  

        # Now pull indices from each saccade 
        tr_inds, te_inds, val_inds = [], [], []
        for nn in self.train_blks:
            tr_inds += list(deepcopy(self.block_inds[nn]))
        for nn in self.val_blks:
            val_inds += list(deepcopy(self.block_inds[nn]))
        for nn in self.test_blks:
            te_inds += list(deepcopy(self.block_inds[nn]))

        if verbose:
            print( "Pre-valid data indices: tr %d, val %d, te %d" %(len(tr_inds), len(val_inds), len(te_inds)) )

        # Finally intersect with used_inds
        self.train_inds = np.array(list(set(tr_inds) & set(self.used_inds)))
        self.val_inds = np.array(list(set(val_inds) & set(self.used_inds)))
        self.test_inds = np.array(list(set(te_inds) & set(self.used_inds)))

        if verbose:
            print( "Valid data indices: tr %d, val %d, te %d" %(len(self.train_inds), len(self.val_inds), len(self.test_inds)) )

    # END .crossval_setup()

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

    def adjust_Xdrift_xval( self ):
    # Adjust drift matrix to not fit xval parameterz
        Xnew = self.Xdrift[:, self.train_blks]
        for ii in range(len(self.val_blks)):
            newpos = self.val_blks[ii] - ii # this will be where lower range is
            if newpos == 0:
                Xnew[self.block_inds[0], 0] = 1.0
            elif newpos == Xnew.shape[1]:
                Xnew[self.block_inds[-1], -1] = 1.0
            else:  # In between
                Xnew[self.block_inds[self.val_blks[ii]], newpos-1] = 0.5
                Xnew[self.block_inds[self.val_blks[ii]], newpos] = 0.5
        self.Xdrift = Xnew
    # END .adjust_drift_xval

    def get_max_samples(self, gpu_n=0, history_size=1, nquad=0, num_cells=None, buffer=1.2):
        """
        get the maximum number of samples that fit in memory -- for GLM/GQM x LBFGS

        Inputs:
            dataset: the dataset to get the samples from
            device: the device to put the samples on
        """
        if gpu_n == 0:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cuda:1')

        if num_cells is None:
            num_cells = self.NC
        
        t = torch.cuda.get_device_properties(device).total_memory
        r = torch.cuda.memory_reserved(device)
        a = torch.cuda.memory_allocated(device)
        free = t - (a+r)

        data = self[0]
        mempersample = data[self.stimname].element_size() * data[self.stimname].nelement() + 2*data['robs'].element_size() * data['robs'].nelement()
    
        mempercell = mempersample * (nquad+1) * (history_size + 1)
        buffer_bytes = buffer*1024**3

        maxsamples = int(free - mempercell*num_cells - buffer_bytes) // mempersample
        print("# samples that can fit on device: {}".format(maxsamples))
        return maxsamples
    # END .get_max_samples

    def __getitem__(self, idx):
        
        if self.preload:

            if self.time_embed == 1:
                print("get_item time embedding not implemented yet")
                # if self.folded_lags:
                #    stim = np.transpose( tmp_stim, axes=[0,2,1,3,4] ) 
                #else:
                #    stim = np.transpose( tmp_stim, axes=[0,2,3,4,1] )
    
            else:
                if self.combine_stim:
                    out = {
                        'stim': self.stimC[idx, :], 
                        'stimH': self.stimH[idx, :], 
                        'stimV': self.stimV[idx, :]}
                else:
                    out = {
                        'stim': self.stimH[idx, :], 
                        'stimV': self.stimV[idx, :]}
                out['fix_n'] = self.fix_n[idx]
                out['Xdrift'] = self.Xdrift[idx]

                if len(self.cells_out) == 0:
                    out['robs'] = self.robs[idx, :]
                    out['dfs'] = self.dfs[idx, :]
                else:
                    assert isinstance(self.cells_out, list), 'cells_out must be a list'
                    robs_tmp = self.robs[:, self.cells_out]
                    dfs_tmp = self.dfs[:, self.cells_out]
                    out['robs'] = robs_tmp[idx, :]
                    out['dfs'] = dfs_tmp[idx, :]
            
        else:
            inds = self.valid_inds[idx]
            stim, stimV = [], []
            robs = []
            dfs = []
            num_dims = self.dims[0]*self.dims[1]*self.dims[2]

            """ Stim """
            # need file handle
            f = 0
            #f = self.file_index[inds]  # problem is this could span across several files
            print("NOT DONE YET")
            stim = torch.tensor(self.fhandles[f][stimname][inds,:], dtype=torch.float32)
            # reshape and flatten stim: currently its NT x NX x NY x Nclrs
            stim = stim.permute([0,3,1,2]).reshape([-1, num_dims])
                
            """ Spikes: needs padding so all are B x NC """ 
            robs = torch.tensor(self.fhandles[f]['Robs'][inds,:], dtype=torch.float32)
            if self.include_MUs:
                robs = torch.cat(
                    (robs, torch.tensor(self.fhandles[f]['RobsMU'][inds,:], dtype=torch.float32)), 
                    dim=1)

            """ Datafilters: needs padding like robs """
            dfs = torch.tensor(self.fhandles[f]['DFs'][inds,:], dtype=torch.float32)
            if self.include_MUs:
                dfs = torch.cat(
                    (dfs, torch.tensor(self.fhandles[f]['DFsMU'][inds,:], dtype=torch.float32)),
                    dim=1)

            out = {'stim': stim, 'stimV': stimV, 'robs': robs, 'dfs': dfs, 'fix_n': self.fix_n[inds]}

        ### THIS IS NOT NEEDED WITH TIME-EMBEDDING: needs to be on fixation-process side...
        #         # Augment to eliminate first X time steps in data_filters
        #if (self.num_lags > 0) &  ~utils.is_int(idx):
        #    if out['dfs'].shape[0] <= self.num_lags:
        #        print( "Warning: fixation shorter than num_lags: %d <= %d"%(out['dfs'].shape[0], self.num_lags))
        #    else:
        #        out['dfs'][:self.num_lags, :] = 0.0

        if self.with_shifter:
            assert self.shifts is not None, "DATASET: shifts missing but 'with_shifter' is true" 
            out['shifts'] = self.shifts[idx]
            
        return out                

    def __len__(self):
        return self.NT
