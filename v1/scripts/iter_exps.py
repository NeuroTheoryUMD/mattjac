import sys
sys.path.append('../lib')
import runner2 as r
import model as m


def tconv_scaffold_iter_stacked(num_lags, num_iter):
    # Temporal Convolutional Scaffold with Iterative Layer
    lgn_layer = m.TemporalConvolutionalLayer(
        num_filters=4,
        num_inh_percent=0.5,
        filter_width=r.Sample(default=17, typ=r.RandomType.odd, values=[17], start=15, end=36, link_id=5),
        num_lags=num_lags-num_iter*2,
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        padding='spatial',
        reg_vals={'d2t': r.Sample(default=0.0001, typ=r.RandomType.float, values=[0.0001], start=0.0, end=0.1),
                  'center': r.Sample(default=0.0, typ=r.RandomType.float, values=[0.0], start=0.0, end=0.1),
                  'bcs': {'d2xt': 1}})

    proj_layer0 = m.TemporalConvolutionalLayer(
        pos_constraint=False,
        num_filters=r.Sample(default=16, typ=r.RandomType.even, values=[16], start=8, end=24, link_id=0),
        num_inh_percent=r.Sample(default=0.5, typ=r.RandomType.float, values=[0.5], start=0, end=1, link_id=1),
        filter_width=r.Sample(default=17, typ=r.RandomType.odd, link_id=5),
        num_lags=1,
        NLtype=m.NL.linear,
        bias=False,
        initialize_center=True,
        reg_vals={'center': r.Sample(default=0.0001, typ=r.RandomType.float, values=[0.0001], start=0.0, end=0.1)})

    itert_layer0 = m.IterativeTemporalConvolutionalLayer(
        pos_constraint=False,
        num_filters=r.Sample(default=16, typ=r.RandomType.even, link_id=0),
        num_inh_percent=r.Sample(default=0.5, typ=r.RandomType.float, link_id=1),
        filter_width=r.Sample(default=7, typ=r.RandomType.odd, values=[7], start=7, end=36, link_id=4),
        num_lags=2,
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        num_iter=num_iter,
        output_config='full',
        res_layer=False,
        reg_vals={#'edge_t0': r.Sample(typ=float, values=[0.0, 1.0], start=0.0, end=0.1),
                  #'activity': r.Sample(typ=float, values=[0.0], start=0.0, end=1.0),
                  'd2xt': r.Sample(default=0.0001, typ=r.RandomType.float, values=[0.0001], start=0.0, end=0.1),
                  'center': r.Sample(default=0.0, typ=r.RandomType.float, values=[0.0], start=0.0, end=0.1),
                  'bcs': {'d2xt': 1}})

    proj_layer1 = m.TemporalConvolutionalLayer(
        pos_constraint=False,
        num_filters=r.Sample(default=32, typ=r.RandomType.even, values=[32], start=8, end=24, link_id=2),
        num_inh_percent=r.Sample(default=0.5, typ=r.RandomType.float, values=[0.5], start=0, end=1, link_id=3),
        filter_width=r.Sample(default=7, typ=r.RandomType.odd, link_id=4),
        num_lags=1,
        NLtype=m.NL.linear,
        bias=False,
        initialize_center=True,
        reg_vals={'center': r.Sample(default=0.0001, typ=r.RandomType.float, values=[0.0001], start=0.0, end=0.1)})

    itert_layer1 = m.IterativeTemporalConvolutionalLayer(
        pos_constraint=False,
        num_filters=r.Sample(default=32, typ=r.RandomType.even, link_id=2),
        num_inh_percent=r.Sample(default=0.5, typ=r.RandomType.float, link_id=3),
        filter_width=r.Sample(default=3, typ=r.RandomType.odd, values=[3], start=3, end=36),
        num_lags=2,
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        num_iter=num_iter,
        output_config='full',
        res_layer=False,
        reg_vals={#'edge_t0': r.Sample(typ=float, values=[0.0, 1.0], start=0.0, end=0.1),
            #'activity': r.Sample(typ=float, values=[0.0], start=0.0, end=1.0),
            'd2xt': r.Sample(default=0.0001, typ=r.RandomType.float, values=[0.0001], start=0.0, end=0.1),
            'center': r.Sample(default=0.0001, typ=r.RandomType.float, values=[0.0], start=0.0, end=0.1),
            'bcs': {'d2xt': 1}})

    readout_layer = m.Layer(
        # because we have inhibitory subunits on the first layer
        pos_constraint=True,
        norm_type=m.Norm.none,
        NLtype=m.NL.softplus,
        bias=True,
        initialize_center=True,
        reg_vals={'glocal': r.Sample(default=0.0, typ=r.RandomType.float, values=[0.0], start=0.0, end=0.1)})

    inp_stim = m.Input(covariate='stim', input_dims=[1,36,1,num_lags])

    core_net = m.Network(layers=[lgn_layer, proj_layer0, itert_layer0, proj_layer1, itert_layer1],
                         network_type=m.NetworkType.scaffold,
                         name='core')
    readout_net = m.Network(layers=[readout_layer],
                            name='readout')
    # this is set as a starting point, but updated on each iteration
    output_11 = m.Output(num_neurons=11)

    inp_stim.to(core_net)
    core_net.to(readout_net)
    readout_net.to(output_11)
    itert_model = m.Model(output_11,
                          name='TconvScaffoldIter',
                          create_NDN=False, verbose=True)
    return itert_model


def tconv_scaffold_iter_expanded(num_lags, num_iter):
    # Temporal Convolutional Scaffold with Iterative Layer
    lgn_layer = m.TemporalConvolutionalLayer(
        num_filters=4,
        num_inh=2,
        filter_dims=[1, 17, 1, num_lags-num_iter],
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        padding='spatial',
        reg_vals={'d2t': r.Sample(typ=float, values=[0.0001, 0.0001], start=0.0, end=0.1),
                  'center': r.Sample(typ=float, values=[0.0, 0.0], start=0.0, end=0.1),
                  'bcs': {'d2xt': 1}})

    proj_layer = m.TemporalConvolutionalLayer(
        pos_constraint=False,
        num_filters=16,
        num_inh=8,
        filter_dims=[4, 17, 1, 1],
        NLtype=m.NL.linear,
        bias=False,
        initialize_center=True,
        window='hamming',
        reg_vals={'center': r.Sample(typ=float, values=[0.0001, 0.0001], start=0.0, end=0.1)})

    itert_layer = m.IterativeTemporalConvolutionalLayer(
        pos_constraint=False,
        num_filters=16,
        num_inh=8,
        filter_dims=7,
        num_lags=2,
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        num_iter=num_iter,
        output_config='full',
        res_layer=False,
        reg_vals={'edge_t0': r.Sample(typ=float, values=[0.0, 1.0], start=0.0, end=0.1),
                  'activity': r.Sample(typ=float, values=[0.0, 0.0], start=0.0, end=1.0),
                  'd2xt': r.Sample(typ=float, values=[0.0001, 0.0001], start=0.0, end=0.1),
                  'center': r.Sample(typ=float, values=[0.0, 0.0], start=0.0, end=0.1),
                  'bcs': {'d2xt': 1}})

    readout_layer = m.Layer(
        # because we have inhibitory subunits on the first layer
        pos_constraint=True,
        norm_type=m.Norm.none,
        NLtype=m.NL.softplus,
        bias=True,
        initialize_center=True,
        reg_vals={'glocal': r.Sample(typ=float, values=[0.0, 0.0], start=0.0, end=0.1)})

    inp_stim = m.Input(covariate='stim', input_dims=[1,36,1,num_lags])

    core_net = m.Network(layers=[lgn_layer, proj_layer, itert_layer],
                         network_type=m.NetworkType.scaffold,
                         name='core')
    readout_net = m.Network(layers=[readout_layer],
                            name='readout')
    # this is set as a starting point, but updated on each iteration
    output_11 = m.Output(num_neurons=11)

    inp_stim.to(core_net)
    core_net.to(readout_net)
    readout_net.to(output_11)
    itert_model = m.Model(output_11,
                          name='TconvScaffoldIter',
                          create_NDN=False, verbose=True)
    return itert_model


expt = ['expt04']
#expt = ['expt04', 'expt06', 'expt09', 'expt11']
#expt = ['expt01', 'expt02', 'expt03', 'expt04', 'expt06', 'expt09', 'expt11', 'expt17', 'expt19', 'expt19']
num_lags = 14
num_iter = 2

trainer_params = r.TrainerParams(num_lags=num_lags,
                                 device="cuda:1", # use the second GPU
                                 max_epochs=1, # just for testing
                                 batch_size=6000,
                                 include_MUs=False,
                                 init_num_samples=0,
                                 bayes_num_steps=0,
                                 num_initializations=1)

runner = r.Runner(experiment_name='iter_exps23',
                  experiment_desc='Trying stacked network.',
                  dataset_expt=expt,
                  dataset_on_gpu=True,
                  model_template=tconv_scaffold_iter_stacked(num_lags, num_iter),
                  trainer_params=trainer_params,
                  trial_params={'num_iter': num_iter,
                                'num_lags': num_lags},
                  overwrite=True)
runner.run()

    
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################


def tconv_scaffold_iter(num_lags, num_iter, layer1_num_lags, num_filters):
    def getL0num_lags(num_iter, iter_filter_height):
        if num_lags-(num_iter*(iter_filter_height-1)) <= 0:
            print('num_lags', num_lags, '<= num_iter', num_iter,
                  '* iter_filter_height-1', iter_filter_height-1,
                  '=', num_iter*(iter_filter_height-1))
        assert num_lags-(num_iter*(iter_filter_height-1)) > 0, \
            'num_lags must be greater than num_iter*(iter_filter_height-1)'
        return num_lags-(num_iter*(iter_filter_height-1))

    # TODO: we need to enforce odd constraints on the filter_width
    #       and enforce the coupling betweeen the num_filters for both layers

    # Temporal Convolutional Scaffold with Iterative Layer
    tconv_layer = m.TemporalConvolutionalLayer(
        num_filters=num_filters,
        num_inh=int(0.5*num_filters),
        filter_width=17,
        num_lags=11,
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        padding='spatial',
        reg_vals={'d2xt': r.Sample(typ=float, values=[0.0001, 0.0001], start=0.00001, end=0.1),
                  'center': r.Sample(typ=float, values=[0, 0], start=0, end=0.1),
                  'bcs': {'d2xt': 1}})

    itert_layer = m.IterativeTemporalConvolutionalLayer(
        num_filters=num_filters,
        num_inh=int(0.5*num_filters),
        filter_dims=7,
        num_lags=layer1_num_lags,
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        num_iter=num_iter,
        output_config='full',
        reg_vals={'activity': r.Sample(typ=float, values=[0, 0.005], start=0.00001, end=0.1),
                  'd2xt': r.Sample(typ=float, values=[0.0001, 0.0001], start=0.00001, end=0.1),
                  'center': r.Sample(typ=float, values=[0, 0], start=0, end=0.1),
                  'bcs': {'d2xt': 1}})

    readout_layer = m.Layer(
        # because we have inhibitory subunits on the first layer
        pos_constraint=True,
        norm_type=m.Norm.none,
        NLtype=m.NL.softplus,
        bias=True,
        initialize_center=True,
        reg_vals={'glocal': r.Sample(typ=float, values=[0, 0], start=0, end=0.1)})

    inp_stim = m.Input(covariate='stim', input_dims=[1,36,1,num_lags])

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
    itert_model = m.Model(output_11,
                          name='TconvScaffoldIter',
                          create_NDN=False, verbose=True)
    return itert_model


# def conv_scaffold(num_filters, num_inh_percent, kernel_widths):
#     conv_layer0 = m.ConvolutionalLayer(
#         num_filters=num_filters[0],
#         num_inh=int(num_filters[0]*num_inh_percent[0]),
#         filter_dims=kernel_widths[0],
#         window='hamming',
#         NLtype=m.NL.relu,
#         norm_type=m.Norm.unit,
#         bias=False,
#         initialize_center=True,
#         output_norm='batch',
#         reg_vals={'d2xt': r.Sample(0.001, 0.1, num_samples=2), 'center': 0.01, 'bcs': {'d2xt': 1}})
#     conv_layer1 = m.ConvolutionalLayer().like(conv_layer0)
#     conv_layer1.params['num_filters'] = num_filters[1]
#     conv_layer1.params['num_inh'] = int(num_filters[1]*num_inh_percent[1])
#     conv_layer1.params['filter_dims'] = kernel_widths[1]
#     conv_layer1.params['reg_vals'] = {'d2xt': 0.01, 'center': 0.01, 'bcs': {'d2xt': 1}}
#     conv_layer2 = m.ConvolutionalLayer().like(conv_layer0)
#     conv_layer2.params['num_filters'] = num_filters[2]
#     conv_layer2.params['num_inh'] = int(num_filters[2]*num_inh_percent[2])
#     conv_layer2.params['filter_dims'] = kernel_widths[2]
#     conv_layer2.params['reg_vals'] = {'d2xt': 0.01, 'center': 0.01, 'bcs': {'d2xt': 1}}
# 
#     readout_layer0 = m.Layer(
#         pos_constraint=True, # because we have inhibitory subunits on the first layer
#         norm_type=m.Norm.none,
#         NLtype=m.NL.softplus,
#         bias=True,
#         initialize_center=True,
#         reg_vals={'glocalx': r.Sample(0.01, 0.1, num_samples=2)}
#     )
# 
#     inp_stim = m.Input(covariate='stim', input_dims=[1,36,1,10])
# 
#     core_net = m.Network(layers=[conv_layer0, conv_layer1, conv_layer2],
#                          network_type=m.NetworkType.scaffold,
#                          name='core')
#     readout_net = m.Network(layers=[readout_layer0],
#                             name='readout')
#     # this is set as a starting point, but updated on each iteration
#     output_11 = m.Output(num_neurons=11)
# 
#     inp_stim.to(core_net)
#     core_net.to(readout_net)
#     readout_net.to(output_11)
#     return m.Model(output_11, name='cnim_scaffold', create_NDN=False, verbose=True)
# 
# 
# 
# def iter_scaffold(num_filters, num_inh_percent, num_iter, reg_vals, kernel_widths, kernel_heights, num_lags):
#     conv_layer = m.ConvolutionalLayer(
#         num_filters=num_filters[0],
#         num_inh=int(num_filters[0]*num_inh_percent),
#         filter_dims=kernel_widths[0],
#         window='hamming',
#         NLtype=m.NL.relu,
#         norm_type=m.Norm.unit,
#         bias=False,
#         initialize_center=True,
#         output_norm='batch',
#         reg_vals=reg_vals)
# 
#     iter_layer = m.IterativeConvolutionalLayer(
#         num_filters=num_lags,
#         num_inh=int(num_lags*num_inh_percent),
#         filter_dims=kernel_widths[0],
#         window='hamming',
#         NLtype=m.NL.relu,
#         norm_type=m.Norm.unit,
#         bias=False,
#         initialize_center=True,
#         output_norm='batch',
#         num_iter=num_iter,
#         output_config='full')
#     if 'activity' in reg_vals.keys():
#         iter_layer.params['reg_vals'] = {'activity': reg_vals['activity']}
#     
#     readout_layer = m.Layer(
#         pos_constraint=True, # because we have inhibitory subunits on the first layer
#         norm_type=m.Norm.none,
#         NLtype=m.NL.softplus,
#         bias=True,
#         initialize_center=True,
#         reg_vals={'glocalx': 0.01}
#     )
# 
#     inp_stim = m.Input(covariate='stim', input_dims=[1,36,1,num_lags])
# 
#     core_net = m.Network(layers=[conv_layer, iter_layer],
#                          network_type=m.NetworkType.scaffold,
#                          name='core')
#     readout_net = m.Network(layers=[readout_layer],
#                             name='readout')
#     # this is set as a starting point, but updated on each iteration
#     output_11 = m.Output(num_neurons=11)
# 
#     inp_stim.to(core_net)
#     core_net.to(readout_net)
#     readout_net.to(output_11)
#     return m.Model(output_11, name='iter_scaffold', create_NDN=False, verbose=True)
# 
# 
# 
# def tconv_scaffold(num_filters, num_inh_percent, num_iter, reg_vals, kernel_widths, kernel_heights, num_lags):
#     tconv_layer0 = m.TemporalConvolutionalLayer(
#         num_filters=num_filters[0],
#         num_inh=int(num_filters[0]*num_inh_percent),
#         filter_dims=[1, kernel_widths[0], 1, kernel_heights[0]],
#         window='hamming',
#         NLtype=m.NL.relu,
#         norm_type=m.Norm.unit,
#         bias=False,
#         initialize_center=True,
#         output_norm='batch',
#         reg_vals=reg_vals)
# 
#     tconv_layer1 = m.ConvolutionalLayer().like(tconv_layer0)
#     tconv_layer1.params['num_filters'] = num_filters[1]
#     tconv_layer1.params['num_inh'] = int(num_filters[1]*num_inh_percent)
#     tconv_layer1.params['filter_dims'] = [1, kernel_widths[1], 1, kernel_heights[1]]
#     if 'activity' in reg_vals.keys():
#         tconv_layer1.params['reg_vals'] = {'activity': reg_vals['activity']}
#     # tconv_layer2 = m.ConvolutionalLayer().like(tconv_layer0)
#     # tconv_layer2.params['num_filters'] = num_filters[2]
#     # tconv_layer2.params['num_inh'] = int(num_filters[2]*num_inh_percent)
#     # tconv_layer2.params['filter_dims'] = [1, kernel_widths[2], 1, kernel_heights[2]]
#     # if 'activity' in reg_vals.keys():
#     #     tconv_layer2.params['reg_vals'] = {'activity': reg_vals['activity']}
# 
#     readout_layer = m.Layer(
#         pos_constraint=True, # because we have inhibitory subunits on the first layer
#         norm_type=m.Norm.none,
#         NLtype=m.NL.softplus,
#         bias=True,
#         initialize_center=True,
#         reg_vals={'glocalx': 0.01}
#     )
# 
#     inp_stim = m.Input(covariate='stim', input_dims=[1,36,1,num_lags])
# 
#     core_net = m.Network(layers=[tconv_layer0],
#                          network_type=m.NetworkType.scaffold,
#                          name='core')
#     readout_net = m.Network(layers=[readout_layer],
#                             name='readout')
#     # this is set as a starting point, but updated on each iteration
#     output_11 = m.Output(num_neurons=11)
# 
#     inp_stim.to(core_net)
#     core_net.to(readout_net)
#     readout_net.to(output_11)
#     return m.Model(output_11, name='tconv_scaffold', create_NDN=False, verbose=True)
