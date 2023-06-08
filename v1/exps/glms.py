import sys
sys.path.append('../lib')
import runner2 as r
import model as m


def glm(num_lags):
    glm_layer = m.Layer(
        NLtype=m.NL.softplus,
        bias=True,
        initialize_center=True,
        reg_vals={'d2xt': r.Sample(typ=float, values=[0.01], start=0.00001, end=0.1),
                  'l1': r.Sample(typ=float, values=[0.0001], start=0.00001, end=0.001),
                  'bcs': {'d2xt': 1}})

    inp_stim = m.Input(covariate='stim', input_dims=[1,36,1,num_lags])
    
    glm_net = m.Network(layers=[glm_layer],
                        name='glm')
    # this is set as a starting point, but updated on each iteration
    output_11 = m.Output(num_neurons=11)

    inp_stim.to(glm_net)
    glm_net.to(output_11)
    glm_model = m.Model(output_11,
                          name='GLM',
                          create_NDN=False, verbose=True)
    return glm_model


expts = [['expt01','expt02','expt03'],
         ['expt04','expt05','expt06'],
         ['expt07','expt08','expt09'],
         ['expt10','expt11','expt12'],
         ['expt13','expt14','expt15'],
         ['expt16','expt17','expt18'],
         ['expt19','expt20','expt21']]
num_lags = 14
for expt in expts:
    trainer_params = r.TrainerParams(num_lags=num_lags,
                                     device="cuda:1", # use the second GPU
                                     max_epochs=3, # just for testing
                                     batch_size=4000,
                                     include_MUs=True,
                                     init_num_samples=0,
                                     bayes_num_steps=0,
                                     num_initializations=1,
                                     trainer_type=r.TrainerType.lbfgs)
    
    runner = r.Runner(experiment_name='glm02_'+'.'.join(expt),
                      dataset_expt=expt,
                      dataset_on_gpu=False,
                      model_template=glm(num_lags),
                      trainer_params=trainer_params,
                      trial_params={'num_lags': num_lags})
    runner.run()