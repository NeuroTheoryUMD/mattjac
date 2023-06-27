# Dataset preprocessor to prepare the data for training
import numpy as np
import torch
from time import time
import scipy.io as sio

from NTdatasets.generic import GenericDataset
from NTdatasets.cumming.monocular import MultiDataset
import NTdatasets.conway.cloud_datasets as datasets

import NDNT.utils as utils # some other utilities
import ColorDataUtils.ConwayUtils as CU
import ColorDataUtils.EyeTrackingUtils as ETutils


class MonocularData:
    def __init__(self, dataset_expt, datadir, trainer_params, cells=None):
        # load the dataset
        print('Loading dataset for', dataset_expt)
        self.datadir = datadir
        self.dataset_expt = dataset_expt
        self.trainer_params = trainer_params
        
    def load(self):
        dataset_params = {
            'time_embed': True,
            'datadir': self.datadir,
            'filenames': self.dataset_expt,
            'include_MUs': self.trainer_params.include_MUs,
            'num_lags': self.trainer_params.num_lags
        }

        data = MultiDataset(**dataset_params)
        data.set_cells()  # specify which cells to use (use all if no params provided)
    
        assert data.train_inds is not None, 'dataset is missing train_inds'
        assert data.val_inds is not None, 'dataset is missing val_inds'
        
        return data
        

class ColorData:
    def __init__(self, datadir, dirname, device, filename=None, top_corner_lam=None, num_lags=None):
        # TODO: store the params in the class
        self.datadir = datadir
        self.dirname = dirname
        self.device = device
    
    def load(self):
        # Load data (all stim)
        # TODO: these should be passed in as parameters

        fn = 'Jocamo_220715_full_CC_ETCC_nofix_v08'
        num_lags=12
        
        t0 = time()
        data = datasets.ColorClouds(
            datadir=self.datadir, filenames=[fn], eye_config=3, drift_interval=16,
            luminance_only=True, binocular=False, include_MUs=False, num_lags=num_lags,
            trial_sample=True)
        t1 = time()
        print(t1-t0, 'sec elapsed')
        
        NT = data.robs.shape[0]
        NA = data.Xdrift.shape[1]
        print("%d (%d valid) time points"%(NT, len(data)))
        #data.valid_inds = np.arange(NT, dtype=np.int64)
        
        lam_units = np.where(data.channel_map < 32)[0]
        ETunits = np.where(data.channel_map >= 32)[0]
        UTunits = np.where(data.channel_map >= 32+127)[0]
        
        print( "%d laminar units, %d ET units"%(len(lam_units), len(ETunits)))
        
        # Replace DFs
        matdat = sio.loadmat(self.datadir+'Jocamo_220715_full_CC_ETCC_nofix_v08_DFextra.mat')
        data.dfs = torch.tensor( matdat['XDF'][:NT, :], dtype=torch.float32 )
        
        # Pull correct saccades
        matdat = sio.loadmat( self.datadir+'Jocamo_220715_full_CC_ETCC_v08_ETupdate.mat')
        sac_ts_all = matdat['ALLsac_bins'][0, :]
        
        data.process_fixations( sac_ts_all )
        sac_tsB = matdat['sac_binsB'][0, :]
        sac_tsL = matdat['sac_binsL'][0, :]
        sac_tsR = matdat['sac_binsR'][0, :]
        
        NFIX = torch.max(data.fix_n).detach().numpy()
        print(NFIX, 'fixations')
        et1kHzB = matdat['et1kHzB']
        et60B = matdat['et60HzB']
        et60all = matdat['et60Hz_all']
        
    
        # Set cells-to-analyze and pull best model configuration and mus
        Reff = torch.mul(data.robs[:, lam_units], data.dfs[:, lam_units]).numpy()
        nspks = np.sum(Reff, axis=0)
        a = np.where(nspks > 10)[0]
        vallam = lam_units[a]
        NCv = len(vallam)
        print("%d out of %d units used"%(len(vallam), len(lam_units)))
        
        ## CONVERT LLsNULL, which is based on 
        
        # Read in previous data
        dirname2 = self.dirname+'0715/et/'
        matdat = sio.loadmat(dirname2+'LLsGLM.mat')
        Dreg = matdat['Dreg']
        top_cornerUT = matdat['top_corner'][:, 0]
        
        data.set_cells(vallam)
        
        # Load shifts and previous models
        dirname2 = self.dirname+'0715/et/'
        SHfile = sio.loadmat( dirname2 + 'BDshifts1.mat' )
        fix_n = SHfile['fix_n']
        shifts = SHfile['shifts']
        metricsLL = SHfile['metricsLL']
        metricsTH = SHfile['metricsTH']
        ETshifts = SHfile['ETshifts']
        ETmetrics = SHfile['ETmetrics']
        Ukeeps = SHfile['Ctrain']
        XVkeeps = SHfile['Cval']
    
        # TODO: this should be passed in as a parameter
        top_corner_lam = [938, 515]
    
        # Make 60x60 STAs (and GLMs)
        Xshift = 0 #8+4 
        Yshift = 0 #-10+4
        NX = 60
        
        new_tc = np.array([top_corner_lam[0]-Xshift, top_corner_lam[1]-Yshift], dtype=np.int64)
        data.draw_stim_locations(top_corner = new_tc, L=NX)
        
        data.assemble_stimulus(top_corner=[new_tc[0], new_tc[1]], L=NX, fixdot=0, shifts=-shifts)
    
    
        goodfix = np.where(ETmetrics[:,1] < 0.80)[0]
        valfix = torch.zeros([ETmetrics.shape[0], 1], dtype=torch.float32)
        valfix[goodfix] = 1.0
        # Test base-level performance (full DFs and then modify DFs)
        #DFsave = deepcopy(data2.dfs)  # this is also in data.dfs
        data.dfs_out *= valfix
        
        return data
