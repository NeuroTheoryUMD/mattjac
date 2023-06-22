import sys
sys.path.insert(0, './')
import runner2 as r
import model as m


def create_iter_network_with_sampled_reg_vals(num_lags, num_iter):
    # Temporal Convolutional Scaffold with Iterative Layer
    lgn_layer = m.TemporalConvolutionalLayer(
        num_filters=4,
        num_inh_percent=0.5,
        filter_width=17,
        num_lags=num_lags-num_iter,
        window='hamming',
        NLtype=m.NL.relu,
        norm_type=m.Norm.unit,
        bias=False,
        initialize_center=True,
        output_norm='batch',
        padding='spatial',
        reg_vals={'d2t': r.Sample(default=0.0001, typ=r.RandomType.float, values=[0.0001, 0.0001], start=0.0, end=0.1),
                  'center': r.Sample(default=0.0, typ=r.RandomType.float, values=[0.0, 0.0], start=0.0, end=0.1),
                  'bcs': {'d2xt': 1}})

    proj_layer = m.TemporalConvolutionalLayer(
        pos_constraint=False,
        num_filters=16,
        num_inh_percent=0.5,
        filter_width=17,
        num_lags=1,
        NLtype=m.NL.linear,
        bias=False,
        initialize_center=True,
        window='hamming',
        reg_vals={'center': r.Sample(default=0.0001, typ=r.RandomType.float, values=[0.0001, 0.0001], start=0.0, end=0.1)})

    itert_layer = m.IterativeTemporalConvolutionalLayer(
        pos_constraint=False,
        num_filters=16,
        num_inh_percent=0.5,
        filter_width=7,
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
        reg_vals={'edge_t0': r.Sample(default=0.0, typ=r.RandomType.float, values=[0.0, 1.0], start=0.0, end=0.1),
                  'activity': r.Sample(default=0.0, typ=r.RandomType.float, values=[0.0, 0.0], start=0.0, end=1.0),
                  'd2xt': r.Sample(default=0.0001, typ=r.RandomType.float, values=[0.0001, 0.0001], start=0.0, end=0.1),
                  'center': r.Sample(default=0.0, typ=r.RandomType.float, values=[0.0, 0.0], start=0.0, end=0.1),
                  'bcs': {'d2xt': 1}})

    readout_layer = m.Layer(
        # because we have inhibitory subunits on the first layer
        pos_constraint=True,
        norm_type=m.Norm.none,
        NLtype=m.NL.softplus,
        bias=True,
        initialize_center=True,
        reg_vals={'glocal': r.Sample(default=0.0, typ=r.RandomType.float, values=[0.0, 0.0], start=0.0, end=0.1)})

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


def create_iter_network_with_sampled_all_params(num_lags, num_iter):
    # Temporal Convolutional Scaffold with Iterative Layer
    lgn_layer = m.TemporalConvolutionalLayer(
        num_filters=4,
        num_inh_percent=0.5,
        filter_width=17,
        num_lags=num_lags-num_iter,
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

    proj_layer = m.TemporalConvolutionalLayer(
        pos_constraint=False,
        num_filters=r.Sample(default=16, typ=r.RandomType.even, values=[16], start=8, end=24, link_id=0),
        num_inh_percent=r.Sample(default=0.5, typ=r.RandomType.float, values=[0.5], start=0, end=1, link_id=1),
        filter_width=17,
        num_lags=1,
        NLtype=m.NL.linear,
        bias=False,
        initialize_center=True,
        window='hamming',
        reg_vals={'center': r.Sample(default=0.0001, typ=r.RandomType.float, values=[0.0001], start=0.0, end=0.1)})

    itert_layer = m.IterativeTemporalConvolutionalLayer(
        pos_constraint=False,
        num_filters=r.Sample(default=16, typ=r.RandomType.even, link_id=0),
        num_inh_percent=r.Sample(default=0.5, typ=r.RandomType.odd, link_id=1),
        filter_width=r.Sample(default=7, typ=r.RandomType.odd, values=[7], start=1, end=36),
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
        reg_vals={'edge_t0': r.Sample(default=0.0, typ=r.RandomType.float, values=[0.0], start=0.0, end=0.1),
                  'activity': r.Sample(default=0.0, typ=r.RandomType.float, values=[0.0], start=0.0, end=1.0),
                  'd2xt': r.Sample(default=0.0001, typ=r.RandomType.float, values=[0.0001], start=0.0, end=0.1),
                  'center': r.Sample(default=0.0, typ=r.RandomType.float, values=[0.0], start=0.0, end=0.1),
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


def test_only_two_values():
    expt = ['expt04']
    num_lags = 14
    num_iter = 2
    
    trainer_params = r.TrainerParams(num_lags=num_lags,
                                     device="cuda:1", # use the second GPU
                                     #max_epochs=1, # just for testing
                                     batch_size=6000,
                                     include_MUs=True,
                                     init_num_samples=0,
                                     bayes_num_steps=0,
                                     num_initializations=1)
    
    runner = r.Runner(experiment_name='test_runner2',
                      experiment_desc='Trying stacked network.',
                      dataset_expt=expt,
                      dataset_on_gpu=False,
                      model_template=create_iter_network_with_sampled_reg_vals(num_lags, num_iter),
                      trainer_params=trainer_params,
                      trial_params={'num_iter': num_iter,
                                    'num_lags': num_lags},
                      overwrite=True)
    
    # test the initial models created by the HyperparameterWalker in the Runner
    assert len(runner.hyperparameter_walker.models) == 2


def test_num_initializations():
    expt = ['expt04']
    num_lags = 14
    num_iter = 2

    trainer_params = r.TrainerParams(num_lags=num_lags,
                                     device="cuda:1", # use the second GPU
                                     #max_epochs=1, # just for testing
                                     batch_size=6000,
                                     include_MUs=True,
                                     init_num_samples=0,
                                     bayes_num_steps=0,
                                     num_initializations=3)

    runner = r.Runner(experiment_name='test_runner2',
                      experiment_desc='Trying stacked network.',
                      dataset_expt=expt,
                      dataset_on_gpu=False,
                      model_template=create_iter_network_with_sampled_reg_vals(num_lags, num_iter),
                      trainer_params=trainer_params,
                      trial_params={'num_iter': num_iter,
                                    'num_lags': num_lags},
                      overwrite=True)

    # test the initial models created by the HyperparameterWalker in the Runner
    assert len(runner.hyperparameter_walker.models) == 6


def test_init_num_samples():
    expt = ['expt04']
    num_lags = 14
    num_iter = 2

    trainer_params = r.TrainerParams(num_lags=num_lags,
                                     device="cuda:1", # use the second GPU
                                     #max_epochs=1, # just for testing
                                     batch_size=6000,
                                     include_MUs=True,
                                     init_num_samples=5,
                                     bayes_num_steps=0,
                                     num_initializations=1)

    runner = r.Runner(experiment_name='test_runner2',
                      experiment_desc='Trying stacked network.',
                      dataset_expt=expt,
                      dataset_on_gpu=False,
                      model_template=create_iter_network_with_sampled_reg_vals(num_lags, num_iter),
                      trainer_params=trainer_params,
                      trial_params={'num_iter': num_iter,
                                    'num_lags': num_lags},
                      overwrite=True)

    # test the initial models created by the HyperparameterWalker in the Runner
    assert len(runner.hyperparameter_walker.models) == 7


def test_sample_all_params():
    expt = ['expt04']
    num_lags = 14
    num_iter = 2

    trainer_params = r.TrainerParams(num_lags=num_lags,
                                     device="cuda:1", # use the second GPU
                                     #max_epochs=1, # just for testing
                                     batch_size=6000,
                                     include_MUs=False,
                                     init_num_samples=5,
                                     bayes_num_steps=0,
                                     num_initializations=1)

    runner = r.Runner(experiment_name='test_runner2',
                      experiment_desc='Trying stacked network.',
                      dataset_expt=expt,
                      dataset_on_gpu=False,
                      model_template=create_iter_network_with_sampled_all_params(num_lags, num_iter),
                      trainer_params=trainer_params,
                      trial_params={'num_iter': num_iter,
                                    'num_lags': num_lags},
                      overwrite=True)

    # test the initial models created by the HyperparameterWalker in the Runner
    assert len(runner.hyperparameter_walker.models) == 6
    
    # test the linked parameters
    model3 = runner.hyperparameter_walker.models[3]
    assert model3.networks[0].layers[1].params['num_filters'] == model3.networks[0].layers[2].params['num_filters']
    
# def test_model_update():
#     while runner.hyperparameter_walker.has_next():
#        model, sample_param_keys, sample_param_vals = runner.hyperparameter_walker.get_next([])

