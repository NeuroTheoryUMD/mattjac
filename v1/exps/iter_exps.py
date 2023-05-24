import sys
sys.path.append('../lib')
import runner as r
import model as m


def conv_scaffold(num_filters, num_inh_percent, kernel_widths):
    conv_layer0 = m.ConvolutionalLayer(
        num_filters=num_filters[0],
        num_inh=int(num_filters[0]*num_inh_percent[0]),
        filter_dims=kernel_widths[0],
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        reg_vals={'d2xt': r.Sample(0.001, 0.1, num_samples=2), 'center': 0.01, 'bcs': {'d2xt': 1}})
    conv_layer1 = m.ConvolutionalLayer().like(conv_layer0)
    conv_layer1.params['num_filters'] = num_filters[1]
    conv_layer1.params['num_inh'] = int(num_filters[1]*num_inh_percent[1])
    conv_layer1.params['filter_dims'] = kernel_widths[1]
    conv_layer1.params['reg_vals'] = {'d2xt': 0.01, 'center': 0.01, 'bcs': {'d2xt': 1}}
    conv_layer2 = m.ConvolutionalLayer().like(conv_layer0)
    conv_layer2.params['num_filters'] = num_filters[2]
    conv_layer2.params['num_inh'] = int(num_filters[2]*num_inh_percent[2])
    conv_layer2.params['filter_dims'] = kernel_widths[2]
    conv_layer2.params['reg_vals'] = {'d2xt': 0.01, 'center': 0.01, 'bcs': {'d2xt': 1}}

    readout_layer0 = m.Layer(
        pos_constraint=True, # because we have inhibitory subunits on the first layer
        norm_type=m.Norm.none,
        NLtype=m.NL.softplus,
        bias=True,
        initialize_center=True,
        reg_vals={'glocalx': r.Sample(0.01, 0.1, num_samples=2)}
    )

    inp_stim = m.Input(covariate='stim', input_dims=[1,36,1,10])

    core_net = m.Network(layers=[conv_layer0, conv_layer1, conv_layer2],
                         network_type=m.NetworkType.scaffold,
                         name='core')
    readout_net = m.Network(layers=[readout_layer0],
                            name='readout')
    # this is set as a starting point, but updated on each iteration
    output_11 = m.Output(num_neurons=11)

    inp_stim.to(core_net)
    core_net.to(readout_net)
    readout_net.to(output_11)
    return m.Model(output_11, name='cnim_scaffold', create_NDN=False, verbose=True)



def iter_scaffold(num_filters, num_inh_percent, num_iter, reg_vals, kernel_widths, kernel_heights, num_lags):
    conv_layer = m.ConvolutionalLayer(
        num_filters=num_filters[0],
        num_inh=int(num_filters[0]*num_inh_percent),
        filter_dims=kernel_widths[0],
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        reg_vals=reg_vals)

    iter_layer = m.IterativeConvolutionalLayer(
        num_filters=num_lags,
        num_inh=int(num_lags*num_inh_percent),
        filter_dims=kernel_widths[0],
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        num_iter=num_iter,
        output_config='full')
    if 'activity' in reg_vals.keys():
        iter_layer.params['reg_vals'] = {'activity': reg_vals['activity']}
    
    readout_layer = m.Layer(
        pos_constraint=True, # because we have inhibitory subunits on the first layer
        norm_type=m.Norm.none,
        NLtype=m.NL.softplus,
        bias=True,
        initialize_center=True,
        reg_vals={'glocalx': 0.01}
    )

    inp_stim = m.Input(covariate='stim', input_dims=[1,36,1,num_lags])

    core_net = m.Network(layers=[conv_layer, iter_layer],
                         network_type=m.NetworkType.scaffold,
                         name='core')
    readout_net = m.Network(layers=[readout_layer],
                            name='readout')
    # this is set as a starting point, but updated on each iteration
    output_11 = m.Output(num_neurons=11)

    inp_stim.to(core_net)
    core_net.to(readout_net)
    readout_net.to(output_11)
    return m.Model(output_11, name='iter_scaffold', create_NDN=False, verbose=True)



def tconv_scaffold(num_filters, num_inh_percent, num_iter, reg_vals, kernel_widths, kernel_heights, num_lags):
    tconv_layer0 = m.TemporalConvolutionalLayer(
        num_filters=num_filters[0],
        num_inh=int(num_filters[0]*num_inh_percent),
        filter_dims=[1, kernel_widths[0], 1, kernel_heights[0]],
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        reg_vals=reg_vals)

    tconv_layer1 = m.ConvolutionalLayer().like(tconv_layer0)
    tconv_layer1.params['num_filters'] = num_filters[1]
    tconv_layer1.params['num_inh'] = int(num_filters[1]*num_inh_percent)
    tconv_layer1.params['filter_dims'] = [1, kernel_widths[1], 1, kernel_heights[1]]
    if 'activity' in reg_vals.keys():
        tconv_layer1.params['reg_vals'] = {'activity': reg_vals['activity']}
    # tconv_layer2 = m.ConvolutionalLayer().like(tconv_layer0)
    # tconv_layer2.params['num_filters'] = num_filters[2]
    # tconv_layer2.params['num_inh'] = int(num_filters[2]*num_inh_percent)
    # tconv_layer2.params['filter_dims'] = [1, kernel_widths[2], 1, kernel_heights[2]]
    # if 'activity' in reg_vals.keys():
    #     tconv_layer2.params['reg_vals'] = {'activity': reg_vals['activity']}

    readout_layer = m.Layer(
        pos_constraint=True, # because we have inhibitory subunits on the first layer
        norm_type=m.Norm.none,
        NLtype=m.NL.softplus,
        bias=True,
        initialize_center=True,
        reg_vals={'glocalx': 0.01}
    )

    inp_stim = m.Input(covariate='stim', input_dims=[1,36,1,num_lags])

    core_net = m.Network(layers=[tconv_layer0],
                         network_type=m.NetworkType.scaffold,
                         name='core')
    readout_net = m.Network(layers=[readout_layer],
                            name='readout')
    # this is set as a starting point, but updated on each iteration
    output_11 = m.Output(num_neurons=11)

    inp_stim.to(core_net)
    core_net.to(readout_net)
    readout_net.to(output_11)
    return m.Model(output_11, name='tconv_scaffold', create_NDN=False, verbose=True)



def tconv_scaffold_iter():
    # Temporal Convolutional Scaffold with Iterative Layer
    tconv_layer = m.TemporalConvolutionalLayer(
        num_filters=8,
        num_inh=4,
        filter_dims=[1, 21, 1, 10-3],
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        reg_vals={'activity':r.Sample(0.00001, 0.1), 'd2xt': r.Sample(0.001, 0.1), 'center': r.Sample(0.001, 0.1), 'bcs': {'d2xt': 1}},
        padding='spatial')
    
    itert_layer = m.IterativeTemporalConvolutionalLayer(
        num_filters=8,
        num_inh=4,
        filter_dims=21,
        num_lags=3,
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        num_iter=3,
        output_config='full',
        reg_vals={'activity':r.Sample(0.00001, 0.1), 'd2xt': r.Sample(0.001, 0.1), 'center': r.Sample(0.001, 0.1), 'bcs': {'d2xt': 1}})
    
    readout_layer = m.Layer(
        pos_constraint=True, # because we have inhibitory subunits on the first layer
        norm_type=m.Norm.none,
        NLtype=m.NL.softplus,
        bias=True,
        initialize_center=True,
        reg_vals={'glocalx': r.Sample(0.01, 0.1)}
    )
    
    inp_stim = m.Input(covariate='stim', input_dims=[1,36,1,10])
    
    core_net = m.Network(layers=[tconv_layer, itert_layer],
                         network_type=m.NetworkType.scaffold,
                         name='core')
    readout_net = m.Network(layers=[readout_layer],
                            name='readout')
    # this is set as a starting point, but updated on each iteration
    output_11 = m.Output(num_neurons=11)
    
    inp_stim.to(core_net)
    core_net.to(readout_net)
    readout_net.to(output_11)
    itert_model = m.Model(output_11, name='tconv_scaffold_iter', create_NDN=False, verbose=True)
    return itert_model


trainer_params = r.TrainerParams(max_epochs=1)

runner = r.Runner(experiment_name='iter_exps2',
                  dataset_expts=[['expt04']],
                  model_templates=[conv_scaffold([8, 4, 4], [0.5, 0.5, 0.5], [21, 7, 3])],
                  trainer_params=trainer_params)

runner.run()
