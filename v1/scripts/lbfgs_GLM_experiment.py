import sys
sys.path.insert(0, '../lib')
sys.path.insert(0, '../')

import torch

# NDN tools
import NDNT.utils as utils # some other utilities
from NTdatasets.cumming.monocular import MultiDataset
from NDNT.modules.layers import *
from NDNT.networks import *

import experiment as exp
import model as m

device = torch.device("cuda:1")
dtype = torch.float32

# load sample dataset to construct the model appropriately
datadir = '../Mdata/'
num_lags = 10
dataset = MultiDataset(
    datadir=datadir,
    filenames=['expt04'],
    include_MUs=False,
    time_embed=True,
    num_lags=num_lags)

fit_pars = utils.create_optimizer_params(
    optimizer_type='lbfgs',
    tolerance_change=1e-10,
    tolerance_grad=1e-10,
    history_size=10,
    batch_size=4000,
    max_epochs=25,
    max_iter = 2000,
    device = device)
fit_pars['device'] = device


def glm(num_filters, reg_vals):
    # create the model
    inp = m.Input(covariate='stim', input_dims=[1,36,1,10])
    layer = m.Layer(
        norm_type=m.Norm.none,
        num_filters=num_filters,
        NLtype=m.NL.softplus,
        bias=True,
        initialize_center=True,
        reg_vals=reg_vals
    )
    network = m.Network(layers=[layer],
                         network_type=m.NetworkType.normal,
                         name='GLM')
    output = m.Output(num_neurons=11)
    
    inp.to(network)
    network.to(output)
    
    return m.Model(output, verbose=True)


# ------------------------------
# parameters to iterate over
experiment_name = 'glm_experiment_1'
experiment_desc = 'Testing GLM training on all data'
modelstr = "GLM"
expt = ['expt04']
include_MUs = False
batch_size = 2000
reg_vals = {'d2xt': 0.01, 'l1': 0.0001, 'center': 0.01, 'bcs': {'d2xt': 1}}


# why are the LLs infinite for some neurons when include_MUs?
# the infinite ones are the ones that are not included in the list of SUs.
from torch.utils.data.dataset import Subset
def eval_function(model, dataset, device):
    # get just the single units from the dataset
    # make a dataset from these and use that as val_ds
    #val_ds = GenericDataset(dataset[dataset.val_inds], device=device)
    val_ds = Subset(dataset, dataset.val_inds)
    return model.NDN.eval_models(val_ds, null_adjusted=True)


def generate_trial(prev_trials):
    trial_idx = 0

    #fit_pars['batch_size'] = batch_size

    print('Loading dataset for', expt)
    dataset_params = {
        'datadir': datadir,
        'filenames': expt,
        'include_MUs': include_MUs,
        'time_embed': True,
        'num_lags': num_lags
    }
    expt_dataset = MultiDataset(**dataset_params)
    expt_dataset.set_cells() # specify which cells to use (use all if no params provided)

    # make the model
    model = glm(expt_dataset.NC, reg_vals)

    # track the specific parameters going into this trial
    trial_params = {
        'null_adjusted_LL': True,
        'expt': '+'.join(expt),
        'include_MUs': include_MUs,
        'batch_size': batch_size,
        'modelstr': modelstr
    }
    # add individual reg_vals to the trial_params
    for k,v in reg_vals.items():
        trial_params[k] = v

    trial_info = exp.TrialInfo(name=modelstr+str(trial_idx),
                               description=modelstr+' with specified parameters',
                               trial_params=trial_params,
                               dataset_params=dataset_params,
                               dataset_class=MultiDataset,
                               fit_params=fit_pars)

    trial = exp.Trial(trial_info=trial_info,
                      model=model,
                      dataset=expt_dataset,
                      eval_function = eval_function)
    trial_idx += 1
    yield trial


# run the experiment
experiment = exp.Experiment(name=experiment_name,
                            description=experiment_desc,
                            generate_trial=generate_trial,
                            experiment_location='../experiments',
                            overwrite=exp.Overwrite.overwrite)
experiment.run(device)
