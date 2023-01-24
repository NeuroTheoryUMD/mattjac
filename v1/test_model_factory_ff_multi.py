import model_factory as mf
import model as m
import torch
import NDNT.utils as utils # some other utilities
import NDNT.modules.layers as l
import NDNT.modules.activations as acts # for some activations
from torch import nn # for other activations

# prevent pytest from truncating long lines
from _pytest.assertion import truncate
truncate.DEFAULT_MAX_LINES = 9999
truncate.DEFAULT_MAX_CHARS = 9999

device = torch.device('cuda:0')

def create_adam_params():
    # create ADAM params
    adam_pars = utils.create_optimizer_params(
        optimizer_type='AdamW', batch_size=2000, num_workers=0,
        learning_rate=0.01, early_stopping_patience=4,
        optimize_graph=False, weight_decay = 0.1)
    adam_pars['device'] = device
    return adam_pars


def create_two_ff_multi_networks(verbose=False):
    netAlayer0 = m.Layer(
        norm_type=m.Norm.none,
        NLtype=m.NL.relu,
        bias=False,
        initialize_center=True,
        reg_vals={'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}},
        num_filters=[2,4]
    )
    netBlayer0 = m.Layer(
        norm_type=m.Norm.none,
        NLtype=m.NL.relu,
        bias=False,
        initialize_center=True,
        reg_vals={'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}},
        num_filters=[6,8]
    )
    netBlayer1 = m.Layer(
        norm_type=[m.Norm.none, m.Norm.max],
        NLtype=m.NL.relu,
        bias=False,
        initialize_center=True,
        reg_vals={'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    )
    
    inp_stim = m.Input(covariate='stim', input_dims=[1,36,1,10])
    netA = m.Network(layers=[netAlayer0], name='A')
    netB = m.Network(layers=[netBlayer0, netBlayer1], name='B')
    output_11 = m.Output(num_neurons=11)

    inp_stim.to(netA)
    netA.to(netB)
    netB.to(output_11)
    model = m.Model(output_11)

    created_models = mf.create_models(model, verbose)
    return created_models



##### NETWORK TESTS ####
def test_two_ff_multi_creation():
    created_models = create_two_ff_multi_networks(verbose=False)
    # test_correct_num_models_created
    assert len(created_models) == 8

    netAlayer0_num_filters = [2,2,2,2,4,4,4,4]
    netBlayer0_num_filters = [6,6,8,8,6,6,8,8]
    netBlayer1_norm_type   = [0,2,0,2,0,2,0,2]
    
    for i in range(len(created_models)):
        model = created_models[i]
        print('NFA', model.networks[0].layers[0].params['num_filters'])
        print('NFB', model.networks[1].layers[0].params['num_filters'])
        print('NORM_TYPE', model.networks[1].layers[1].params['norm_type'])
        # test_correct_model_created
        assert len(model.networks) == 2
        assert len(model.networks[0].layers) == 1
        assert len(model.networks[1].layers) == 2
        
        input_stim = model.inputs[0]
        netA = model.networks[0]
        netB = model.networks[1]
        output_11 = model.output
        
        # test_correct_inputs_created
        assert input_stim.name == 'stim'
        assert input_stim.index == -1
        assert input_stim.covariate == 'stim'
        assert input_stim.input_dims == [1,36,1,10]
        assert len(input_stim.inputs) == 0
        assert input_stim.output == netA
        assert len(input_stim.layers) == 1
        assert len(input_stim.layers[0].params) == 10
        # TODO: maybe we should not use a 'virtual' layer?
        assert input_stim.layers[0].params['internal_layer_type'] is l.NDNLayer
        assert input_stim.output is netA
        
        # test_correct_output_created
        assert output_11.num_neurons == 11
        assert len(output_11.inputs) == 1
        assert output_11.inputs[0] is netB
    
        # test_correct_networks_created
        # netA
        assert netA.model is model
        assert netA.name == 'A'
        assert netA.index == 0
        assert len(netA.inputs) == 1
        assert netA.inputs[0] is input_stim
        assert netA.output is netB    
        assert netA.input_covariate == 'stim'
        assert netA.ffnet_type == 'normal'
        # netB
        assert netB.model is model
        assert netB.name == 'B'
        assert netB.index == 1
        assert len(netB.inputs) == 1
        assert netB.inputs[0] is netA
        assert netB.output is output_11
        assert netB.input_covariate is None
        assert netB.ffnet_type == 'normal'
    
        # test_correct_layers_created
        # netA.layer0
        assert netA.layers[0].network is netA
        assert netA.layers[0].index == 0
        assert netA.layers[0].params['internal_layer_type'] is l.NDNLayer
        assert netA.layers[0].params['norm_type'] == 0
        assert netA.layers[0].params['NLtype'] == 'relu'
        assert netA.layers[0].params['bias'] == False
        assert netA.layers[0].params['initialize_center'] == True
        assert netA.layers[0].params['num_filters'] == netAlayer0_num_filters[i]
        assert netA.layers[0].params['reg_vals'] == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
        assert netA.layers[0].params['input_dims'] == [1,36,1,10]
        # netB.layer0
        assert netB.layers[0].network is netB
        assert netB.layers[0].index == 0
        assert netB.layers[0].params['internal_layer_type'] is l.NDNLayer
        assert netB.layers[0].params['norm_type'] == 0
        assert netB.layers[0].params['NLtype'] == 'relu'
        assert netB.layers[0].params['bias'] == False
        assert netB.layers[0].params['initialize_center'] == True
        assert netB.layers[0].params['num_filters'] == netBlayer0_num_filters[i]
        assert netB.layers[0].params['reg_vals'] == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
        assert 'input_dims' not in netB.layers[0].params # these should be empty
        # netB.layer1
        assert netB.layers[1].network is netB
        assert netB.layers[1].index == 1
        assert netB.layers[1].params['internal_layer_type'] is l.NDNLayer
        assert netB.layers[1].params['norm_type'] == netBlayer1_norm_type[i]
        assert netB.layers[1].params['NLtype'] == 'relu'
        assert netB.layers[1].params['bias'] == False
        assert netB.layers[1].params['initialize_center'] == True
        assert netB.layers[1].params['num_filters'] == 11 # this should be set to num_neurons
        assert netB.layers[1].params['reg_vals'] == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
        assert 'input_dims' not in netB.layers[1].params # these should be empty    
        
        
        # TODO: this is actually testing the NDN construction, not just my params
        #       this is more complicated than it is worth, I believe...
        # ##### NDN TESTS ####
        NDN0 = model.NDN
        # test_correct_NDN_created
        assert len(NDN0.networks) == 2
    
        ffnetA = NDN0.networks[0]
        ffnetB = NDN0.networks[1]
        
        # test_correct_ffnets_created
        # ffnetA
        assert ffnetA.ffnets_in is None
        assert ffnetA.xstim_n == 'stim'
        assert ffnetA.network_type == 'normal'
        assert len(ffnetA.layer_list) == 1
        # ffnetB
        assert len(ffnetB.ffnets_in) == 1
        assert ffnetB.ffnets_in[0] == 0 # assert netB has ffnetA as network input
        assert ffnetB.xstim_n is None # assert netB has no data inputs
        assert ffnetB.network_type == 'normal'
        assert len(ffnetB.layer_list) == 2 # assert layers length
        
        # test_correct_NDNLayers_created
        # ffnetA.ndnlayer0
        assert isinstance(ffnetA.layers[0], l.NDNLayer) # check that we made the right class type
        assert ffnetA.layers[0].norm_type == 0
        assert isinstance(ffnetA.layers[0].NL, nn.ReLU)
        #TODO: assert isinstance(ffnetA.layers[0].bias, nn.Parameter) # just check it is the right kind of thing
        #TODO: assert ffnetA.layers[0].initialize_center == True
        assert ffnetA.layers[0].num_filters == netAlayer0_num_filters[i]
        #TODO: assert ffnetA.layers[0].reg_vals == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
        assert ffnetA.layers[0].input_dims == [1,36,1,10]
    
        # ffnetB.ndnlayer0
        assert isinstance(ffnetB.layers[0], l.NDNLayer) # check that we made the right class type
        assert ffnetB.layers[0].norm_type == 0
        assert isinstance(ffnetB.layers[0].NL, nn.ReLU)
        #TODO: assert isinstance(ffnetB.layers[0].bias, nn.Parameter) # just check it is the right kind of thing
        #TODO: assert ffnetB.layers[0].initialize_center == True
        assert ffnetB.layers[0].num_filters == netBlayer0_num_filters[i]
        #TODO: assert ffnetB.layers[0].reg_vals == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
        assert ffnetB.layers[0].input_dims == [netAlayer0_num_filters[i],1,1,1] # this depends on the previous layer's num_filters
    
        # ffnetB.ndnlayer1
        assert isinstance(ffnetB.layers[1], l.NDNLayer) # check that we made the right class type
        assert ffnetB.layers[1].norm_type == netBlayer1_norm_type[i]
        assert isinstance(ffnetB.layers[1].NL, nn.ReLU)
        #TODO: assert isinstance(ffnetB.layers[1].bias, nn.Parameter) # just check it is the right kind of thing
        #TODO: assert ffnetB.layers[1].initialize_center == True
        assert ffnetB.layers[1].num_filters == 11 # equal to the num_neurons
        #TODO: assert ffnetB.layers[1].reg_vals == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
        assert ffnetB.layers[1].input_dims == [netBlayer0_num_filters[i],1,1,1]
