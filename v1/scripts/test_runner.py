import runner as r
import model as m

def cnim_scaffold(num_filters, num_inh_percent, reg_vals, kernel_widths, kernel_heights):
    conv_layer0 = m.ConvolutionalLayer(
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
    conv_layer1 = m.ConvolutionalLayer().like(conv_layer0)
    conv_layer1.params['num_filters'] = num_filters[1]
    conv_layer1.params['num_inh'] = int(num_filters[1]*num_inh_percent)
    conv_layer1.params['filter_dims'] = kernel_widths[1]
    if 'activity' in reg_vals.keys():
        conv_layer1.params['reg_vals'] = {'activity': reg_vals['activity']}
    conv_layer2 = m.ConvolutionalLayer().like(conv_layer0)
    conv_layer2.params['num_filters'] = num_filters[2]
    conv_layer2.params['num_inh'] = int(num_filters[2]*num_inh_percent)
    conv_layer2.params['filter_dims'] = kernel_widths[2]
    if 'activity' in reg_vals.keys():
        conv_layer2.params['reg_vals'] = {'activity': reg_vals['activity']}

    readout_layer0 = m.Layer(
        pos_constraint=True, # because we have inhibitory subunits on the first layer
        norm_type=m.Norm.none,
        NLtype=m.NL.softplus,
        bias=True,
        initialize_center=True,
        reg_vals={'glocalx': 0.01}
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
    return m.Model(output_11, name='cnim_scaffold', verbose=True)


experiment_name = 'test_runner'
expts =  [['expt04']]
num_lags = 10
copy_weightses = [False]
freeze_weightses = [False]
include_MUses = [False]
is_multiexps = [False]
batch_sizes = [3000]
num_filterses = [[16, 8, 8]]
num_inh_percents = [0.5]
kernel_widthses = [[21, 11, 5]]
kernel_heightses = [[3, 3, 3]]
num_runs = 1 # the number of times to run each trial
reg_valses = [{'activity':0.00001, 'd2xt': 0.01, 'center': 0.01, 'bcs': {'d2xt': 1}}]

hyperparameter_walker = r.HyperparameterWalker(num_filterses=num_filterses,
                                               num_inh_percents=num_inh_percents,
                                               kernel_widthses=kernel_widthses,
                                               kernel_heightses=kernel_heightses,
                                               reg_valses=reg_valses,
                                               include_MUses=include_MUses,
                                               is_multiexps=is_multiexps,
                                               batch_sizes=batch_sizes)

runner = r.Runner(experiment_name=experiment_name,
                  dataset_expts=expts,
                  num_lags=num_lags,
                  model_templates=[cnim_scaffold],
                  hyperparameter_walker=hyperparameter_walker,
                  max_epochs=1)

runner.run()
