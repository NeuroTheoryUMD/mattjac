import model_generator as mgen
import model as m
import NDNT.modules.layers as l
from torch import nn # for other activations

# prevent pytest from truncating long lines
from _pytest.assertion import truncate
truncate.DEFAULT_MAX_LINES = 9999
truncate.DEFAULT_MAX_CHARS = 9999


def create_one_conv_network(verbose=False):
    netAlayer0 = m.ConvolutionalLayer(
        norm_type=m.Norm.none,
        NLtype=m.NL.relu,
        bias=False,
        initialize_center=True,
        reg_vals={'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}},
        num_filters=8,
        filter_dims=21,
        window='hamming'
    )
    netAlayer1 = m.Layer(
        norm_type=m.Norm.none,
        NLtype=m.NL.relu,
        bias=False,
        initialize_center=True,
        reg_vals={'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    )
    
    inp_stim = m.Input(covariate='stim', input_dims=[1,36,1,10])
    netA = m.Network(layers=[netAlayer0, netAlayer1], name='A')
    output_11 = m.Output(num_neurons=11)
    
    inp_stim.to(netA)
    netA.to(output_11)
    model = m.Model(output_11)

    created_models = mgen.generate(model, explorer=mgen.Explorer.grid, verbose=verbose)
    return created_models



##### NETWORK TESTS ####
def test_one_conv_network_creation():
    created_models = create_one_conv_network(verbose=False)
    # test_correct_num_models_created
    assert len(created_models) == 1, 'there should be 4 models created'

    model0 = created_models[0]
    # test_correct_model_created
    assert len(model0.networks) == 1
    assert len(model0.networks[0].layers) == 2
    
    input_stim = model0.inputs[0]
    netA = model0.networks[0]
    output_11 = model0.output
    
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
    assert output_11.inputs[0] is netA

    # test_correct_networks_created
    # netA
    assert netA.model is model0
    assert netA.name == 'A'
    assert netA.index == 0
    assert len(netA.inputs) == 1
    assert netA.inputs[0] is input_stim
    assert netA.output is output_11
    assert netA.input_covariate == 'stim'
    assert netA.ffnet_type == 'normal'

    # test_correct_layers_created
    # netA.layer0
    assert netA.layers[0].network is netA
    assert netA.layers[0].index == 0
    assert netA.layers[0].params['internal_layer_type'] is l.ConvLayer
    assert netA.layers[0].params['norm_type'] == 0
    assert netA.layers[0].params['NLtype'] == 'relu'
    assert netA.layers[0].params['bias'] == False
    assert netA.layers[0].params['initialize_center'] == True
    assert netA.layers[0].params['num_filters'] == 8
    assert netA.layers[0].params['reg_vals'] == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert netA.layers[0].params['input_dims'] == [1,36,1,10]
    assert netA.layers[0].params['padding'] == 'same'
    assert netA.layers[0].params['filter_dims'] == 21
    assert netA.layers[0].params['window'] == 'hamming'
    # netA.layer1
    assert netA.layers[1].network is netA
    assert netA.layers[1].index == 1
    assert netA.layers[1].params['internal_layer_type'] is l.NDNLayer
    assert netA.layers[1].params['norm_type'] == 0
    assert netA.layers[1].params['NLtype'] == 'relu'
    assert netA.layers[1].params['bias'] == False
    assert netA.layers[1].params['initialize_center'] == True
    assert netA.layers[1].params['num_filters'] == 11 # this should be set to num_neurons
    assert netA.layers[1].params['reg_vals'] == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert 'input_dims' not in netA.layers[1].params # these should be empty
    
    # TODO: this is actually testing the NDN construction, not just my params
    #       this is more complicated than it is worth, I believe...
    # ##### NDN TESTS ####
    NDN0 = model0.NDN
    # test_correct_NDN_created
    assert len(NDN0.networks) == 1

    ffnetA = NDN0.networks[0]
    
    # test_correct_ffnets_created
    # ffnetA
    assert ffnetA.ffnets_in is None
    assert ffnetA.xstim_n == 'stim'
    assert ffnetA.network_type == 'normal'
    assert len(ffnetA.layer_list) == 2
    
    # test_correct_NDNLayers_created
    # netA.layer0
    assert isinstance(ffnetA.layers[0], l.ConvLayer) # check that we made the right class type
    assert ffnetA.layers[0].norm_type == 0
    assert isinstance(ffnetA.layers[0].NL, nn.ReLU)
    #TODO: assert isinstance(ffnetA.layers[0].bias, nn.Parameter) # just check it is the right kind of thing
    #TODO: assert ffnetA.layers[0].initialize_center == True
    assert ffnetA.layers[0].num_filters == 8
    #TODO: assert ffnetA.layers[0].reg_vals == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert ffnetA.layers[0].input_dims == [1,36,1,10]

    # netA.layer1
    assert isinstance(ffnetA.layers[1], l.NDNLayer) # check that we made the right class type
    assert ffnetA.layers[1].norm_type == 0
    assert isinstance(ffnetA.layers[1].NL, nn.ReLU)
    #TODO: assert isinstance(ffnetA.layers[1].bias, nn.Parameter) # just check it is the right kind of thing
    #TODO: assert ffnetA.layers[1].initialize_center == True
    assert ffnetA.layers[1].num_filters == 11 # equal to the num_neurons
    #TODO: assert ffnetA.layers[1].reg_vals == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert ffnetA.layers[1].input_dims == [8,36,1,1]
