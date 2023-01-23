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


def create_two_simple_networks(verbose=False):
    netAlayer0 = m.Layer(
        norm_type=m.Norm.none,
        NLtype=m.NL.relu,
        bias=False,
        initialize_center=True,
        reg_vals={'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}},
        num_filters=2
    )
    netBlayer0 = m.Layer(
        norm_type=m.Norm.none,
        NLtype=m.NL.relu,
        bias=False,
        initialize_center=True,
        reg_vals={'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}},
        num_filters=4
    )
    netBlayer1 = m.Layer(
        norm_type=m.Norm.none,
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


def create_three_simple_networks(verbose=False):
    netAlayer0 = m.Layer(
        norm_type=m.Norm.none,
        NLtype=m.NL.relu,
        bias=False,
        initialize_center=True,
        reg_vals={'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}},
        num_filters=3
    )
    netBlayer0 = m.Layer(
        norm_type=m.Norm.none,
        NLtype=m.NL.relu,
        bias=False,
        initialize_center=True,
        reg_vals={'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}},
        num_filters=4
    )
    netBlayer1 = m.Layer(
        norm_type=m.Norm.none,
        NLtype=m.NL.relu,
        bias=False,
        initialize_center=True,
        reg_vals={'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}},
        num_filters=5
    )
    netClayer0 = m.Layer(
        norm_type=m.Norm.none,
        NLtype=m.NL.relu,
        bias=False,
        initialize_center=True,
        reg_vals={'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    )

    inp_stim = m.Input(covariate='stim', input_dims=[1,36,1,10])
    netA = m.Network(layers=[netAlayer0], name='A')
    netB = m.Network(layers=[netBlayer0, netBlayer1], name='B')
    netC = m.Network(layers=[netClayer0], name='C')
    output_11 = m.Output(num_neurons=11)

    inp_stim.to(netA)
    netA.to(netB)
    netB.to(netC)
    netC.to(output_11)
    model = m.Model(output_11)

    created_models = mf.create_models(model, verbose)
    return created_models



##### NETWORK TESTS ####
def test_two_ff_single_creation():
    created_models = create_two_simple_networks(verbose=False)
    # test_correct_num_models_created
    assert len(created_models) == 1, 'there should be 4 models created'

    model0 = created_models[0]
    # test_correct_model_created
    assert len(model0.networks) == 2
    assert len(model0.networks[0].layers) == 1
    assert len(model0.networks[1].layers) == 2
    
    input_stim = model0.inputs[0]
    netA = model0.networks[0]
    netB = model0.networks[1]
    output_11 = model0.output
    
    # test_correct_inputs_created
    assert input_stim.name == 'stim'
    assert input_stim.index == -1
    assert input_stim.covariate == 'stim'
    assert input_stim.input_dims == [1,36,1,10]
    assert len(input_stim.inputs) == 0
    assert input_stim.output == netA
    assert len(input_stim.layers) == 1
    assert len(input_stim.layers[0].params) == 9 # has the default number of params
    # TODO: maybe we should not use a 'virtual' layer?
    assert input_stim.layers[0].params['internal_layer_type'] is l.NDNLayer
    assert input_stim.output is netA
    
    # test_correct_output_created
    assert output_11.num_neurons == 11
    assert len(output_11.inputs) == 1
    assert output_11.inputs[0] is netB

    # test_correct_networks_created
    # netA
    assert netA.model is model0
    assert netA.name == 'A'
    assert netA.index == 0
    assert len(netA.inputs) == 1
    assert netA.inputs[0] is input_stim
    assert netA.output is netB    
    assert netA.input_covariate == 'stim'
    assert netA.ffnet_type == 'normal'
    # netB
    assert netB.model is model0
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
    assert netA.layers[0].params['internal_layer_type'] is l.NDNLayer
    assert netA.layers[0].params['norm_type'] == 0
    assert netA.layers[0].params['NLtype'] == 'relu'
    assert netA.layers[0].params['bias'] == False
    assert netA.layers[0].params['initialize_center'] == True
    assert netA.layers[0].params['num_filters'] == 2
    assert netA.layers[0].params['reg_vals'] == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert netA.layers[0].params['input_dims'] == [1,36,1,10]
    # netB.layer0
    assert netB.layers[0].network is netB
    assert netB.layers[0].params['internal_layer_type'] is l.NDNLayer
    assert netB.layers[0].params['norm_type'] == 0
    assert netB.layers[0].params['NLtype'] == 'relu'
    assert netB.layers[0].params['bias'] == False
    assert netB.layers[0].params['initialize_center'] == True
    assert netB.layers[0].params['num_filters'] == 4
    assert netB.layers[0].params['reg_vals'] == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert 'input_dims' not in netB.layers[0].params # these should be empty
    # netB.layer1
    assert netB.layers[1].network is netB
    assert netB.layers[1].params['internal_layer_type'] is l.NDNLayer
    assert netB.layers[1].params['norm_type'] == 0
    assert netB.layers[1].params['NLtype'] == 'relu'
    assert netB.layers[1].params['bias'] == False
    assert netB.layers[1].params['initialize_center'] == True
    assert netB.layers[1].params['num_filters'] == 11 # this should be set to num_neurons
    assert netB.layers[1].params['reg_vals'] == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert 'input_dims' not in netB.layers[1].params # these should be empty    
    
    # TODO: this is actually testing the NDN construction, not just my params
    #       this is more complicated than it is worth, I believe...
    # ##### NDN TESTS ####
    NDN0 = model0.NDN
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
    # netA.layer0
    assert isinstance(ffnetA.layers[0], l.NDNLayer) # check that we made the right class type
    assert ffnetA.layers[0].norm_type == 0
    assert isinstance(ffnetA.layers[0].NL, nn.ReLU)
    #TODO: assert isinstance(ffnetA.layers[0].bias, nn.Parameter) # just check it is the right kind of thing
    #TODO: assert ffnetA.layers[0].initialize_center == True
    assert ffnetA.layers[0].num_filters == 2
    #TODO: assert ffnetA.layers[0].reg_vals == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert ffnetA.layers[0].input_dims == [1,36,1,10]

    # netB.layer0
    assert isinstance(ffnetB.layers[0], l.NDNLayer) # check that we made the right class type
    assert ffnetB.layers[0].norm_type == 0
    assert isinstance(ffnetB.layers[0].NL, nn.ReLU)
    #TODO: assert isinstance(ffnetB.layers[0].bias, nn.Parameter) # just check it is the right kind of thing
    #TODO: assert ffnetB.layers[0].initialize_center == True
    assert ffnetB.layers[0].num_filters == 4
    #TODO: assert ffnetB.layers[0].reg_vals == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert ffnetB.layers[0].input_dims == [2,1,1,1]

    # netB.layer1
    assert isinstance(ffnetB.layers[1], l.NDNLayer) # check that we made the right class type
    assert ffnetB.layers[1].norm_type == 0
    assert isinstance(ffnetB.layers[1].NL, nn.ReLU)
    #TODO: assert isinstance(ffnetB.layers[1].bias, nn.Parameter) # just check it is the right kind of thing
    #TODO: assert ffnetB.layers[1].initialize_center == True
    assert ffnetB.layers[1].num_filters == 11 # equal to the num_neurons
    #TODO: assert ffnetB.layers[1].reg_vals == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert ffnetB.layers[1].input_dims == [4,1,1,1]


def test_three_ff_single_creation():
    created_models = create_three_simple_networks(verbose=False)
    # test_correct_num_models_created
    assert len(created_models) == 1, 'there should be 4 models created'

    model0 = created_models[0]
    # test_correct_model_created
    assert len(model0.networks) == 3
    assert len(model0.networks[0].layers) == 1
    assert len(model0.networks[1].layers) == 2
    assert len(model0.networks[2].layers) == 1

    input_stim = model0.inputs[0]
    netA = model0.networks[0]
    netB = model0.networks[1]
    netC = model0.networks[2]
    output_11 = model0.output

    # test_correct_inputs_created
    assert input_stim.name == 'stim'
    assert input_stim.index == -1
    assert input_stim.covariate == 'stim'
    assert input_stim.input_dims == [1,36,1,10]
    assert len(input_stim.inputs) == 0
    assert input_stim.output == netA
    assert len(input_stim.layers) == 1
    assert len(input_stim.layers[0].params) == 9 # has the default number of params
    # TODO: maybe we should not use a 'virtual' layer?
    assert input_stim.layers[0].params['internal_layer_type'] is l.NDNLayer
    assert input_stim.output is netA

    # test_correct_output_created
    assert output_11.num_neurons == 11
    assert len(output_11.inputs) == 1
    assert output_11.inputs[0] is netC

    # test_correct_networks_created
    # netA
    assert netA.model is model0
    assert netA.name == 'A'
    assert netA.index == 0
    assert len(netA.inputs) == 1
    assert netA.inputs[0] is input_stim
    assert netA.output is netB
    assert netA.input_covariate == 'stim'
    assert netA.ffnet_type == 'normal'
    # netB
    assert netB.model is model0
    assert netB.name == 'B'
    assert netB.index == 1
    assert len(netB.inputs) == 1
    assert netB.inputs[0] is netA
    assert netB.output is netC
    assert netB.input_covariate is None
    assert netB.ffnet_type == 'normal'
    # netC
    assert netC.model is model0
    assert netC.name == 'C'
    assert netC.index == 2
    assert len(netC.inputs) == 1
    assert netC.inputs[0] is netB
    assert netC.output is output_11
    assert netC.input_covariate is None
    assert netC.ffnet_type == 'normal'

    # test_correct_layers_created
    # netA.layer0
    assert netA.layers[0].network is netA
    assert netA.layers[0].params['internal_layer_type'] is l.NDNLayer
    assert netA.layers[0].params['norm_type'] == 0
    assert netA.layers[0].params['NLtype'] == 'relu'
    assert netA.layers[0].params['bias'] == False
    assert netA.layers[0].params['initialize_center'] == True
    assert netA.layers[0].params['num_filters'] == 3
    assert netA.layers[0].params['reg_vals'] == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert netA.layers[0].params['input_dims'] == [1,36,1,10]
    # netB.layer0
    assert netB.layers[0].network is netB
    assert netB.layers[0].params['internal_layer_type'] is l.NDNLayer
    assert netB.layers[0].params['norm_type'] == 0
    assert netB.layers[0].params['NLtype'] == 'relu'
    assert netB.layers[0].params['bias'] == False
    assert netB.layers[0].params['initialize_center'] == True
    assert netB.layers[0].params['num_filters'] == 4
    assert netB.layers[0].params['reg_vals'] == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert 'input_dims' not in netB.layers[0].params # these should be empty
    # netB.layer1
    assert netB.layers[1].network is netB
    assert netB.layers[1].params['internal_layer_type'] is l.NDNLayer
    assert netB.layers[1].params['norm_type'] == 0
    assert netB.layers[1].params['NLtype'] == 'relu'
    assert netB.layers[1].params['bias'] == False
    assert netB.layers[1].params['initialize_center'] == True
    assert netB.layers[1].params['num_filters'] == 5
    assert netB.layers[1].params['reg_vals'] == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert 'input_dims' not in netB.layers[1].params # these should be empty    
    # netC.layer0
    assert netC.layers[0].network is netC
    assert netC.layers[0].params['internal_layer_type'] is l.NDNLayer
    assert netC.layers[0].params['norm_type'] == 0
    assert netC.layers[0].params['NLtype'] == 'relu'
    assert netC.layers[0].params['bias'] == False
    assert netC.layers[0].params['initialize_center'] == True
    assert netC.layers[0].params['num_filters'] == 11 # this should be set to num_neurons
    assert netC.layers[0].params['reg_vals'] == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert 'input_dims' not in netC.layers[0].params # these should be empty  

    # TODO: this is actually testing the NDN construction, not just my params
    #       this is more complicated than it is worth, I believe...
    # ##### NDN TESTS ####
    NDN0 = model0.NDN
    # test_correct_NDN_created
    assert len(NDN0.networks) == 3

    ffnetA = NDN0.networks[0]
    ffnetB = NDN0.networks[1]
    ffnetC = NDN0.networks[2]

    # test_correct_ffnets_created
    # ffnetA
    assert ffnetA.ffnets_in is None
    assert ffnetA.xstim_n == 'stim'
    assert ffnetA.network_type == 'normal'
    assert len(ffnetA.layer_list) == 1
    # ffnetB
    assert len(ffnetB.ffnets_in) == 1
    assert ffnetB.ffnets_in[0] == 0 # assert ffnetB has ffnetA as network input
    assert ffnetB.xstim_n is None # assert netB has no data inputs
    assert ffnetB.network_type == 'normal'
    assert len(ffnetB.layer_list) == 2 # assert layers length
    # ffnetC
    assert len(ffnetC.ffnets_in) == 1
    assert ffnetC.ffnets_in[0] == 1 # assert ffnetC has ffnetB as network input
    assert ffnetC.xstim_n is None # assert netC has no data inputs
    assert ffnetC.network_type == 'normal'
    assert len(ffnetC.layer_list) == 1 # assert layers length

    # test_correct_NDNLayers_created
    # netA.layer0
    assert isinstance(ffnetA.layers[0], l.NDNLayer) # check that we made the right class type
    assert ffnetA.layers[0].norm_type == 0
    assert isinstance(ffnetA.layers[0].NL, nn.ReLU)
    #TODO: assert isinstance(ffnetA.layers[0].bias, nn.Parameter) # just check it is the right kind of thing
    #TODO: assert ffnetA.layers[0].initialize_center == True
    assert ffnetA.layers[0].num_filters == 3
    #TODO: assert ffnetA.layers[0].reg_vals == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert ffnetA.layers[0].input_dims == [1,36,1,10]

    # netB.layer0
    assert isinstance(ffnetB.layers[0], l.NDNLayer) # check that we made the right class type
    assert ffnetB.layers[0].norm_type == 0
    assert isinstance(ffnetB.layers[0].NL, nn.ReLU)
    #TODO: assert isinstance(ffnetB.layers[0].bias, nn.Parameter) # just check it is the right kind of thing
    #TODO: assert ffnetB.layers[0].initialize_center == True
    assert ffnetB.layers[0].num_filters == 4
    #TODO: assert ffnetB.layers[0].reg_vals == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert ffnetB.layers[0].input_dims == [3,1,1,1]

    # netB.layer1
    assert isinstance(ffnetB.layers[1], l.NDNLayer) # check that we made the right class type
    assert ffnetB.layers[1].norm_type == 0
    assert isinstance(ffnetB.layers[1].NL, nn.ReLU)
    #TODO: assert isinstance(ffnetB.layers[1].bias, nn.Parameter) # just check it is the right kind of thing
    #TODO: assert ffnetB.layers[1].initialize_center == True
    assert ffnetB.layers[1].num_filters == 5
    #TODO: assert ffnetB.layers[1].reg_vals == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert ffnetB.layers[1].input_dims == [4,1,1,1]

    # netC.layer0
    assert isinstance(ffnetC.layers[0], l.NDNLayer) # check that we made the right class type
    assert ffnetC.layers[0].norm_type == 0
    assert isinstance(ffnetC.layers[0].NL, nn.ReLU)
    #TODO: assert isinstance(ffnetC.layers[0].bias, nn.Parameter) # just check it is the right kind of thing
    #TODO: assert ffnetC.layers[0].initialize_center == True
    assert ffnetC.layers[0].num_filters == 11 # equal to the num_neurons
    #TODO: assert ffnetC.layers[0].reg_vals == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert ffnetC.layers[0].input_dims == [5,1,1,1]

