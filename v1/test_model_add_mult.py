import model as m
import NDNT.modules.layers as l
from torch import nn # for other activations

# prevent pytest from truncating long lines
from _pytest.assertion import truncate
truncate.DEFAULT_MAX_LINES = 9999
truncate.DEFAULT_MAX_CHARS = 9999


def create_two_add_network(verbose=False):
    l1 = m.Layer(num_filters=8, NLtype=m.NL.relu, bias=False)
    l2 = m.Layer(num_filters=8, NLtype=m.NL.relu)
    l3 = m.Layer(num_filters=4, NLtype=m.NL.softplus)
    l4 = m.Layer(num_filters=4, NLtype=m.NL.softplus)
    l5 = m.Layer(num_filters=2, NLtype=m.NL.relu)
    l6 = m.Layer(NLtype=m.NL.relu, bias=True)
    
    i1 = m.Input(covariate='i1', input_dims=[1,36,1,10])
    i2 = m.Input(covariate='i2', input_dims=[1,36,1,10])
    i3 = m.Input(covariate='i3', input_dims=[1,36,1,10])
    n1 = m.Network(layers=[l1], name='n1')
    n2 = m.Network(layers=[l2], name='n2')
    n3 = m.Network(layers=[l3], name='n3')
    n4 = m.Network(layers=[l4], name='n4')
    n5 = m.Network(layers=[l5], name='n5')
    n6 = m.Network(layers=[l6], name='n6')
    o  = m.Output(11)
    
    i1.to(n1)
    i2.to(n2)
    i3.to(n4)
    m.Add(networks=[n1,n2]).to(n3)
    m.Mult(networks=[n3,n4]).to(n5)
    n5.to(n6)
    n6.to(o)

    model = m.Model(o, verbose)
    return model


##### NETWORK TESTS ####
def test_two_add_network_creation():
    model = create_two_add_network(verbose=False)

    # test_correct_model_created
    assert len(model.networks) == 1
    assert len(model.networks[0].layers) == 2

    input_stim = model.inputs[0]
    netA = model.networks[0]
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
    assert output_11.inputs[0] is netA

    # test_correct_networks_created
    # netA
    assert netA.model is model
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
    NDN0 = model.NDN
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
