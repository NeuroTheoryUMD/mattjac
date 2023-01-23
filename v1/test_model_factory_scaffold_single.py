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

def create_scaffold_single_network(verbose=False):
    conv_layer0 = m.ConvolutionalLayer(
        norm_type=m.Norm.none,
        NLtype=m.NL.relu,
        bias=False,
        initialize_center=True,
        reg_vals={'d2xt': 0.0001, 'center': None, 'bcs':{'d2xt':1} },
        num_filters=8,
        filter_dims=21,
        window='hamming',
        output_norm='batch',
        num_inh=4)
    conv_layer1 = m.ConvolutionalLayer().like(conv_layer0)
    conv_layer1.params['num_filters'] = [8]
    conv_layer1.params['num_inh'] = [4]
    conv_layer1.params['filter_dims'] = [9]
    conv_layer2 = m.ConvolutionalLayer().like(conv_layer0)
    conv_layer2.params['num_filters'] = [4]
    conv_layer2.params['num_inh'] = [2]
    conv_layer2.params['filter_dims'] = [9]
    
    readout_layer0 = m.Layer(
        norm_type=m.Norm.none,
        NLtype=m.NL.softplus,
        bias=True,
        reg_vals={'glocalx': 0.001},
        pos_constraint=True
    )
    
    inp_stim = m.Input(covariate='stim', input_dims=[1,36,1,10])
    
    scaffold_net = m.Network(layers=[conv_layer0, conv_layer1, conv_layer2],
                             network_type=m.NetworkType.scaffold,
                             name='scaffold')
    readout_net = m.Network(layers=[readout_layer0],
                            name='readout')
    output_11 = m.Output(num_neurons=11)
    
    inp_stim.to(scaffold_net)
    scaffold_net.to(readout_net)
    readout_net.to(output_11)
    model = m.Model(output_11)
    
    created_models = mf.create_models(model, verbose)
    return created_models


def test_scaffold_network_creation():
    created_models = create_scaffold_single_network(verbose=False)
    # test_correct_num_models_created
    assert len(created_models) == 1, 'there should be 4 models created'

    model0 = created_models[0]
    # test_correct_model_created
    assert len(model0.networks) == 2
    assert len(model0.networks[0].layers) == 3
    assert len(model0.networks[1].layers) == 1

    input_stim = model0.inputs[0]
    net_scaffold = model0.networks[0]
    net_readout = model0.networks[1]
    output_11 = model0.output

    # test_correct_inputs_created
    assert input_stim.name == 'stim'
    assert input_stim.index == -1
    assert input_stim.covariate == 'stim'
    assert input_stim.input_dims == [1,36,1,10]
    assert len(input_stim.inputs) == 0
    assert input_stim.output == net_scaffold
    assert len(input_stim.layers) == 1
    assert len(input_stim.layers[0].params) == 10
    # TODO: maybe we should not use a 'virtual' layer?
    assert input_stim.layers[0].params['internal_layer_type'] is l.NDNLayer
    assert input_stim.output is net_scaffold

    # test_correct_output_created
    assert output_11.num_neurons == 11
    assert len(output_11.inputs) == 1
    assert output_11.inputs[0] is net_readout
    

    # TODO: this is actually testing the NDN construction, not just my params
    #       this is more complicated than it is worth, I believe...
    # ##### NDN TESTS ####
    NDN0 = model0.NDN
    # test_correct_NDN_created
    assert len(NDN0.networks) == 2

    ffnet_scaffold = NDN0.networks[0]
    ffnet_readout = NDN0.networks[1]

    # test_correct_ffnets_created
    # ffnet_scaffold
    assert ffnet_scaffold.ffnets_in is None
    assert ffnet_scaffold.xstim_n == 'stim'
    assert ffnet_scaffold.network_type == 'scaffold'
    assert len(ffnet_scaffold.layer_list) == 3

    # test_correct_NDNLayers_created
    # ffnet_scaffold.layer0
    assert isinstance(ffnet_scaffold.layers[0], l.ConvLayer) # check that we made the right class type
    assert ffnet_scaffold.layers[0].norm_type == 0
    assert isinstance(ffnet_scaffold.layers[0].NL, nn.ReLU)
    assert ffnet_scaffold.layers[0].num_filters == 8
    assert ffnet_scaffold.layers[0].num_inh == 4
    assert ffnet_scaffold.layers[0].filter_dims == [1,21,1,10]
    assert ffnet_scaffold.layers[0].input_dims == [1,36,1,10]
    
    # ffnet_scaffold.layer[-1]
    assert isinstance(ffnet_scaffold.layers[-1], l.ConvLayer) # check that we made the right class type
    assert ffnet_scaffold.layers[-1].norm_type == 0
    assert isinstance(ffnet_scaffold.layers[-1].NL, nn.ReLU)
    assert ffnet_scaffold.layers[-1].num_filters == 4
    assert ffnet_scaffold.layers[-1].num_inh == 2
    assert ffnet_scaffold.layers[-1].filter_dims == [8,9,1,1]
    assert ffnet_scaffold.layers[-1].input_dims == [8,36,1,1]

    # ffnet_readout.layer0
    assert isinstance(ffnet_readout.layers[0], l.NDNLayer) # check that we made the right class type
    assert ffnet_readout.layers[0].norm_type == 0
    assert isinstance(ffnet_readout.layers[0].NL, nn.Softplus)
    #TODO: assert isinstance(ffnet_readout.layers[0].bias, nn.Parameter) # just check it is the right kind of thing
    #TODO: assert ffnet_readout.layers[0].initialize_center == True
    assert ffnet_readout.layers[0].num_filters == 11 # equal to the num_neurons
    #TODO: assert ffnet_readout.layers[0].reg_vals == {'l1':0.1, 'localx':0.001, 'bcs':{'d2xt':1}}
    assert ffnet_readout.layers[0].input_dims == [20,36,1,1]
