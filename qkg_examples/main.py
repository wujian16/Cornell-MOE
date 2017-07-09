import numpy as np
import os, sys
import time

from moe.optimal_learning.python.cpp_wrappers.domain import TensorProductDomain as cppTensorProductDomain
from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient_mcmc import PosteriorMeanMCMC
from moe.optimal_learning.python.cpp_wrappers.log_likelihood_mcmc import GaussianProcessLogLikelihoodMCMC as cppGaussianProcessLogLikelihoodMCMC
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentParameters as cppGradientDescentParameters
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentOptimizer as cppGradientDescentOptimizer
from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient import posterior_mean_optimization, PosteriorMean

from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.repeated_domain import RepeatedDomain
from moe.optimal_learning.python.default_priors import DefaultPrior

from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.optimization import GradientDescentParameters as pyGradientDescentParameters
from moe.optimal_learning.python.python_version.optimization import GradientDescentOptimizer as pyGradientDescentOptimizer
from moe.optimal_learning.python.python_version.optimization import multistart_optimize as multistart_optimize

import bgo_methods
import obj_functions

# arguments for calling this script:
# python main.py [obj_func_name] [num_to_sample] [num_lhc] [job_id]
# example: python main.py Hartmann3 4 1000 1
# you can define your own obj_function and then just change the objective_func object below, and run this script.

argv = sys.argv[1:]
obj_func_name = str(argv[0])
num_to_sample = int(argv[1])
lhc_search_itr = int(argv[2])
job_id = int(argv[3])

# constants
num_func_eval = 100
num_iteration = int(num_func_eval / num_to_sample) + 1

obj_func_dict = {'BraninNoNoise': obj_functions.BraninNoNoise(), 'RosenbrockNoNoise': obj_functions.RosenbrockNoNoise(),
                 'Hartmann3': obj_functions.Hartmann3(), 'HartmannNoNoise': obj_functions.Hartmann6()}
objective_func = obj_func_dict[obj_func_name]
dim = int(objective_func._dim)
num_initial_points = int(objective_func._num_init_pts)

python_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in objective_func._search_domain])
cpp_search_domain = cppTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in objective_func._search_domain])

# get the initial data
init_pts = python_search_domain.generate_uniform_random_points_in_domain(3)
# observe
derivatives = []
observations = [0] + [i+1 for i in derivatives]
init_pts_value = np.array([objective_func.evaluate(pt) for pt in init_pts])[:, observations]

true_value_init = np.array([objective_func.evaluate_true(pt) for pt in init_pts])[:, observations]

init_data = HistoricalData(dim = objective_func._dim, num_derivatives = len(derivatives))
init_data.append_sample_points([SamplePoint(pt, [init_pts_value[num, i] for i in observations],
                                            objective_func._sample_var) for num, pt in enumerate(init_pts)])

num_fidelity = 2

# initialize the model
prior = DefaultPrior(1+dim+len(observations), len(observations))
# noisy = False means the underlying function being optimized is noise-free
cpp_gp_loglikelihood = cppGaussianProcessLogLikelihoodMCMC(historical_data = init_data, derivatives = derivatives, prior = prior,
                                                           chain_length = 2000, burnin_steps = 1000, n_hypers = 8, noisy = False)
cpp_gp_loglikelihood.train()

py_sgd_params_ps = pyGradientDescentParameters(max_num_steps=100, max_num_restarts=2,
                                               num_steps_averaged=15, gamma=0.7, pre_mult=0.01,
                                               max_relative_change=0.5, tolerance=1.0e-5)

cpp_sgd_params_ps = cppGradientDescentParameters(num_multistarts=1, max_num_steps=20, max_num_restarts=1,
                                                 num_steps_averaged=3, gamma=0.7, pre_mult=0.03,
                                                 max_relative_change=0.2, tolerance=1.0e-5)

cpp_sgd_params_kg = cppGradientDescentParameters(num_multistarts=120, max_num_steps=20, max_num_restarts=1,
                                                 num_steps_averaged=4, gamma=0.7, pre_mult=0.3,
                                                 max_relative_change=0.6, tolerance=1.0e-5)

# minimum of the mean surface
eval_pts = python_search_domain.generate_uniform_random_points_in_domain(int(1e3))
eval_pts = np.reshape(np.append(eval_pts, (cpp_gp_loglikelihood.get_historical_data_copy()).points_sampled),
                      (eval_pts.shape[0] + cpp_gp_loglikelihood._num_sampled, cpp_gp_loglikelihood.dim))
test = np.zeros(eval_pts.shape[0])
for cpp_gp in cpp_gp_loglikelihood.models:
    test += cpp_gp.compute_mean_of_points(eval_pts)
test /= len(cpp_gp_loglikelihood.models)
report_point = eval_pts[np.argmin(test)].reshape((1, cpp_gp_loglikelihood.dim))

ps = PosteriorMeanMCMC(cpp_gp_loglikelihood.models, num_fidelity)
py_repeated_search_domain = RepeatedDomain(num_repeats = 1, domain = python_search_domain)
ps_mean_opt = pyGradientDescentOptimizer(py_repeated_search_domain, ps, py_sgd_params_ps)
report_point = multistart_optimize(ps_mean_opt, report_point, num_multistarts = 1)[0]
report_point = report_point.ravel()

print "best so far in the initial data {0}".format(true_value_init[np.argmin(true_value_init[:,0])][0])
for n in xrange(num_iteration):
    print "KG, {0}th job, {1}th iteration, func={2}, q={3}".format(
            job_id, n, obj_func_name, num_to_sample
    )
    time1 = time.time()
    discrete_pts_list = []
    discrete = python_search_domain.generate_uniform_random_points_in_domain(200)
    for i, cpp_gp in enumerate(cpp_gp_loglikelihood.models):
        discrete_pts_optima = np.array(discrete)

        eval_pts = python_search_domain.generate_uniform_random_points_in_domain(int(1e3))
        eval_pts = np.reshape(np.append(eval_pts, (cpp_gp.get_historical_data_copy()).points_sampled), (eval_pts.shape[0] + cpp_gp.num_sampled, cpp_gp.dim))
        test = cpp_gp.compute_mean_of_points(eval_pts)
        initial_point = eval_pts[np.argmin(test)]

        ps_evaluator = PosteriorMean(cpp_gp, num_fidelity)
        ps_sgd_optimizer = cppGradientDescentOptimizer(cpp_search_domain, ps_evaluator, cpp_sgd_params_ps)
        report_point = posterior_mean_optimization(ps_sgd_optimizer, initial_guess = initial_point, max_num_threads = 4)
        if cpp_gp.compute_mean_of_points(report_point.reshape(1, dim)) > cpp_gp.compute_mean_of_points(initial_point.reshape(1, dim)):
            report_point = initial_point

        discrete_pts_optima = np.reshape(np.append(discrete_pts_optima, report_point),
                                         (discrete_pts_optima.shape[0] + 1, cpp_gp.dim))

        discrete_pts_list.append(discrete_pts_optima)

    ps_evaluator = PosteriorMean(cpp_gp_loglikelihood.models[0], num_fidelity)
    ps_sgd_optimizer = cppGradientDescentOptimizer(cpp_search_domain, ps_evaluator, cpp_sgd_params_ps)
    next_points, voi = bgo_methods.gen_sample_from_qkg_mcmc(cpp_gp_loglikelihood._gaussian_process_mcmc, cpp_gp_loglikelihood.models,
                                                            ps_sgd_optimizer, cpp_search_domain, num_fidelity, discrete_pts_list,
                                                            cpp_sgd_params_kg, num_to_sample, num_mc=100, lhc_itr=lhc_search_itr)
    print "KG takes "+str((time.time()-time1)/60)+" mins"
    time1 = time.time()

    sampled_points = [SamplePoint(pt, objective_func.evaluate(pt)[observations], objective_func._sample_var) for pt in next_points]

    print "evaluating takes "+str((time.time()-time1)/60)+" mins"

    # retrain the model
    time1 = time.time()

    cpp_gp_loglikelihood.add_sampled_points(sampled_points)
    cpp_gp_loglikelihood.train()

    print "retraining the model takes "+str((time.time()-time1)/60)+" mins"
    time1 = time.time()

    # report the point
    eval_pts = python_search_domain.generate_uniform_random_points_in_domain(int(1e4))
    eval_pts = np.reshape(np.append(eval_pts, (cpp_gp_loglikelihood.get_historical_data_copy()).points_sampled),
                          (eval_pts.shape[0] + cpp_gp_loglikelihood._num_sampled, cpp_gp_loglikelihood.dim))
    test = np.zeros(eval_pts.shape[0])
    for cpp_gp in cpp_gp_loglikelihood.models:
        test += cpp_gp.compute_mean_of_points(eval_pts)
    test /= len(cpp_gp_loglikelihood.models)
    initial_point = eval_pts[np.argmin(test)].reshape((1, cpp_gp_loglikelihood.dim))

    ps = PosteriorMeanMCMC(cpp_gp_loglikelihood.models, num_fidelity)
    py_repeated_search_domain = RepeatedDomain(num_repeats = 1, domain = python_search_domain)
    ps_mean_opt = pyGradientDescentOptimizer(py_repeated_search_domain, ps, py_sgd_params_ps)
    report_point = multistart_optimize(ps_mean_opt, initial_point, num_multistarts = 1)[0]
    if ps.compute_posterior_mean_mcmc(report_point.reshape(1, dim)) < ps.compute_posterior_mean_mcmc(initial_point.reshape(1, dim)):
        report_point = initial_point
    report_point = report_point.ravel()

    print "recommended points: ",
    print report_point
    print "recommending the point takes "+str((time.time()-time1)/60)+" mins"
    print "KG, VOI {0}, best so far {1}".format(voi, objective_func.evaluate_true(report_point)[0])