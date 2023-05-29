import sys
sys.path.append('../lib')
import runner2 as r
import model as m


def tconv_scaffold_iter(num_lags):
    default_num_iter = 3
    default_layer1_num_lags = 2
    
    def getL0num_lags(num_iter, iter_filter_height):
        if num_lags-(num_iter*(iter_filter_height-1)) <= 0:
            print('num_lags', num_lags, '<= num_iter', num_iter,
                  '* iter_filter_height-1', iter_filter_height-1, 
                  '=', num_iter*(iter_filter_height-1))
        assert num_lags-(num_iter*(iter_filter_height-1)) > 0,\
            'num_lags must be greater than num_iter*(iter_filter_height-1)'
        return num_lags-(num_iter*(iter_filter_height-1))
    
    
    # Temporal Convolutional Scaffold with Iterative Layer
    tconv_layer = m.TemporalConvolutionalLayer(
        num_filters=[24, None, r.ValueList([24, 24, 24, 24])],
        num_inh=[12, None, r.ValueList([12, 12, 12, 12])],
        filter_width=[17, None, r.ValueList([17, 17, 17, 17])],
        num_lags=[getL0num_lags(default_num_iter, default_layer1_num_lags), None, 
                  r.ValueList([getL0num_lags(3, default_layer1_num_lags),
                               getL0num_lags(5, default_layer1_num_lags),
                               getL0num_lags(7, default_layer1_num_lags),
                               getL0num_lags(9, default_layer1_num_lags)])],
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        padding='spatial',
        reg_vals={'d2xt': [0.0001, r.Sample(0.00001, 0.1), None], 
                  'center': [0, r.Sample(0.0001, 0.1), None], 
                  'bcs': {'d2xt': 1}})

    itert_layer = m.IterativeTemporalConvolutionalLayer(
        num_filters=[24, None, r.ValueList([24, 24, 24, 24])],
        num_inh=[12, None, r.ValueList([12, 12, 12, 12])],
        filter_dims=[7, None, r.ValueList([7, 7, 7, 7])],
        num_lags=default_layer1_num_lags,
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        num_iter=[default_num_iter, None, r.ValueList([3, 5, 7, 9])],
        output_config='full',
        reg_vals={'activity':[0, r.Sample(0.00001, 0.1), None],
                  'd2xt': [0.0001, r.Sample(0.00001, 0.1), None],
                  'center': [0, r.Sample(0.0001, 0.1), None],
                  'bcs': {'d2xt': 1}})

    readout_layer = m.Layer(
        # because we have inhibitory subunits on the first layer
        pos_constraint=True,
        norm_type=m.Norm.none,
        NLtype=m.NL.softplus,
        bias=True,
        initialize_center=True)

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

expt = ['expt04', 'expt06', 'expt07', 'expt09', 'expt11']
#expt = ['expt04']
num_lags = 20

model_templates = []
trainer_params = r.TrainerParams(num_lags=num_lags,
                                 device="cuda:1", # use the second GPU
                                 #max_epochs=1, # just for testing
                                 include_MUs=True,
                                 bayes_init_num_samples=5,
                                 bayes_max_num_samples=10)

runner = r.Runner(experiment_name='iter_exps03',
                  dataset_expt=expt,
                  model_template=tconv_scaffold_iter(num_lags),
                  trainer_params=trainer_params)
runner.run()



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
