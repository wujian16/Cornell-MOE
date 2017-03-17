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
from moe.optimal_learning.python.random_features import sample_from_global_optima

from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.optimization import GradientDescentParameters as pyGradientDescentParameters
from moe.optimal_learning.python.python_version.optimization import GradientDescentOptimizer as pyGradientDescentOptimizer
from moe.optimal_learning.python.python_version.optimization import multistart_optimize as multistart_optimize

import bgo_methods
import obj_functions

# arguments for calling this script:
# python synthetic.test.functions.py [num_to_sample] [num_lhc] [job_id]
# example: python bayesian.test.functions.py 8 1000 1
argv = sys.argv[1:]
num_to_sample = int(argv[0])
lhc_search_itr = int(argv[1])
job_id = int(argv[2])

num_func_eval = 150
num_iteration = int(num_func_eval / num_to_sample) + 1

objective_func = obj_functions.BraninNoNoise()
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

# initialize the model
prior = DefaultPrior(1+dim+len(observations), len(observations), noisy = False)
cpp_gp_loglikelihood = cppGaussianProcessLogLikelihoodMCMC(historical_data = init_data, derivatives = derivatives, prior = prior,
                                                           chain_length = 50, burnin_steps = 50, n_hypers = 40)
cpp_gp_loglikelihood.train()

py_sgd_params_kg = pyGradientDescentParameters(max_num_steps=10, max_num_restarts=2,
                                               num_steps_averaged=15, gamma=0.7, pre_mult=0.1,
                                               max_relative_change=0.1, tolerance=1.0e-5)

py_sgd_params_ps = pyGradientDescentParameters(max_num_steps=20, max_num_restarts=2,
                                               num_steps_averaged=15, gamma=0.7, pre_mult=0.02,
                                               max_relative_change=0.02, tolerance=1.0e-5)

cpp_sgd_params_ps = cppGradientDescentParameters(num_multistarts=1, max_num_steps=20, max_num_restarts=2,
                                                 num_steps_averaged=15, gamma=0.7, pre_mult=0.02,
                                                 max_relative_change=0.02, tolerance=1.0e-5)

# minimum of the mean surface
eval_pts = python_search_domain.generate_uniform_random_points_in_domain(int(1e5))
eval_pts = np.reshape(np.append(eval_pts, (cpp_gp_loglikelihood.get_historical_data_copy()).points_sampled),
                      (eval_pts.shape[0] + cpp_gp_loglikelihood._num_sampled, cpp_gp_loglikelihood.dim))
test = np.zeros(eval_pts.shape[0])
for cpp_gp in cpp_gp_loglikelihood.models:
    test += cpp_gp.compute_mean_of_points(eval_pts)
test /= len(cpp_gp_loglikelihood.models)
report_point = eval_pts[np.argmin(test)].reshape((1, cpp_gp_loglikelihood.dim))

ps = PosteriorMeanMCMC(cpp_gp_loglikelihood.models)
py_repeated_search_domain = RepeatedDomain(num_repeats = 1, domain = python_search_domain)
ps_mean_opt = pyGradientDescentOptimizer(py_repeated_search_domain, ps, py_sgd_params_ps)
report_point = multistart_optimize(ps_mean_opt, report_point, num_multistarts = 1)[0]
report_point = report_point.ravel()

print "best so far in the initial data {0}".format(true_value_init[np.argmin(true_value_init[:,0])][0])
for n in xrange(num_iteration):
    print "KG, {0}th job, {1}th iteration, func=Branin, q={2}".format(
            job_id, n, num_to_sample
    )
    time1 = time.time()
    discrete_pts_list = []
    for i, cpp_gp in enumerate(cpp_gp_loglikelihood.models):
        init_points = python_search_domain.generate_uniform_random_points_in_domain(int(1e3))
        discrete_pts_optima = sample_from_global_optima(cpp_gp, 1000, objective_func._search_domain, init_points, 20)

        eval_pts = python_search_domain.generate_uniform_random_points_in_domain(int(1e3))
        eval_pts = np.reshape(np.append(eval_pts, (cpp_gp.get_historical_data_copy()).points_sampled), (eval_pts.shape[0] + cpp_gp.num_sampled, cpp_gp.dim))
        test = cpp_gp.compute_mean_of_points(eval_pts)
        report_point = eval_pts[np.argmin(test)]
        print cpp_gp.compute_mean_of_points(report_point.reshape(1, dim))
        ps_evaluator = PosteriorMean(cpp_gp)
        ps_sgd_optimizer = cppGradientDescentOptimizer(cpp_search_domain, ps_evaluator, cpp_sgd_params_ps)
        report_point = posterior_mean_optimization(ps_sgd_optimizer, initial_guess = report_point, max_num_threads = 4)
        print cpp_gp.compute_mean_of_points(report_point.reshape(1, dim))

        discrete_pts_optima = np.reshape(np.append(discrete_pts_optima, report_point),
                                  (discrete_pts_optima.shape[0] + 1, cpp_gp.dim))
        discrete_pts_list.append(discrete_pts_optima)

    ps_evaluator = PosteriorMean(cpp_gp_loglikelihood.models[0])
    ps_sgd_optimizer = cppGradientDescentOptimizer(cpp_search_domain, ps_evaluator, cpp_sgd_params_ps)
    next_points, voi = bgo_methods.gen_sample_from_qkg_mcmc(cpp_gp_loglikelihood.models, ps_sgd_optimizer, python_search_domain, discrete_pts_list,
                                                            py_sgd_params_kg, num_to_sample, num_mc=200, lhc_itr=lhc_search_itr)

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
    eval_pts = python_search_domain.generate_uniform_random_points_in_domain(int(1e3))
    eval_pts = np.reshape(np.append(eval_pts, (cpp_gp_loglikelihood.get_historical_data_copy()).points_sampled),
                          (eval_pts.shape[0] + cpp_gp_loglikelihood._num_sampled, cpp_gp_loglikelihood.dim))
    test = np.zeros(eval_pts.shape[0])
    for cpp_gp in cpp_gp_loglikelihood.models:
        test += cpp_gp.compute_mean_of_points(eval_pts)
    test /= len(cpp_gp_loglikelihood.models)
    report_point = eval_pts[np.argmin(test)].reshape((1, cpp_gp_loglikelihood.dim))

    ps = PosteriorMeanMCMC(cpp_gp_loglikelihood.models)
    py_repeated_search_domain = RepeatedDomain(num_repeats = 1, domain = python_search_domain)
    ps_mean_opt = pyGradientDescentOptimizer(py_repeated_search_domain, ps, py_sgd_params_ps)
    report_point = multistart_optimize(ps_mean_opt, report_point, num_multistarts = 1)[0]
    report_point = report_point.ravel()

    print "recommending the point takes "+str((time.time()-time1)/60)+" mins"
    print "KG, best so far {0}".format(objective_func.evaluate_true(report_point)[0])