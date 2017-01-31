import numpy as np
import pandas as pd
import sys
import time

from moe.optimal_learning.python.cpp_wrappers.covariance import SquareExponential as cppSquareExponential
from moe.optimal_learning.python.cpp_wrappers.domain import TensorProductDomain as cppTensorProductDomain
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess as cppGaussianProcess
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentParameters as cppGradientDescentParameters
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentOptimizer as cppGradientDescentOptimizer
from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient import posterior_mean_optimization, PosteriorMean

from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
from moe.optimal_learning.python.geometry_utils import ClosedInterval

from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain

from moe.optimal_learning.python.cpp_wrappers.log_likelihood import GaussianProcessLogLikelihood as cppGaussianProcessLogLikelihood
from moe.optimal_learning.python.cpp_wrappers.log_likelihood import multistart_hyperparameter_optimization

import bgo_methods
import obj_functions
import GP_Approximation_Random_Feature


obj_func_name = "Branin"

num_func_eval_dict = {"Branin": 150, "LG": 60, "Hartmann": 60, "Ackley": 200, "Rosenbrock": 100, "Levy": 60}

# constants
obj_func_dict = {'Branin': obj_functions.Branin(), 'Hartmann': obj_functions.Hartmann(), 'Rosenbrock': obj_functions.Rosenbrock(),
                 'Ackley': obj_functions.Ackley(), 'Levy': obj_functions.Levy()}

# opt_method = {'qKGg': bgo_methods.gen_sample_from_qkg(), 'qEIg': bgo_methods.gen_sample_from_qei()}

cpp_sgd_params_ei = cppGradientDescentParameters(num_multistarts=200, max_num_steps=100, max_num_restarts=2,
                                                 num_steps_averaged=15, gamma=0.7, pre_mult=0.1,
                                                 max_relative_change=0.7, tolerance=1.0e-5)

cpp_sgd_params_hyper = cppGradientDescentParameters(num_multistarts=100, max_num_steps=100, max_num_restarts=2,
                                                    num_steps_averaged=15, gamma=0.7, pre_mult=0.1,
                                                    max_relative_change=0.02, tolerance=1.0e-5)

cpp_sgd_params_kg = cppGradientDescentParameters(num_multistarts=50, max_num_steps=100, max_num_restarts=2,
                                                 num_steps_averaged=15, gamma=0.7, pre_mult=0.1,
                                                 max_relative_change=0.1, tolerance=1.0e-5)

cpp_sgd_params_ps = cppGradientDescentParameters(num_multistarts=1, max_num_steps=100, max_num_restarts=2,
                                                 num_steps_averaged=15, gamma=0.7, pre_mult=0.05,
                                                 max_relative_change=0.02, tolerance=1.0e-5)

objective_func = obj_func_dict[obj_func_name]
dim = int(objective_func._dim)
num_initial_points = int(objective_func._num_init_pts)

python_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in objective_func._search_domain])


# read the initial data
file = "/fs/home/jw926/KG.Gradients/Initial.Points/"+obj_func_name+".initial.points.csv"
data = pd.read_csv(file, sep=",", index_col=0)

job_id = 1
init_pts = data.iloc[num_initial_points*(job_id-1) : num_initial_points*job_id, :dim].as_matrix()
init_pts_value = data.iloc[num_initial_points*(job_id-1) : num_initial_points*job_id, dim:].as_matrix()

derivatives = np.arange(2)
observations = [0] + [derivatives[i]+1 for i in derivatives]


init_data = HistoricalData(dim = objective_func._dim, num_derivatives = len(derivatives))
init_data.append_sample_points([SamplePoint(pt, [init_pts_value[num, i] for i in observations], objective_func._sample_var) for num, pt in enumerate(init_pts)])

cpp_cov = cppSquareExponential(np.ones(objective_func._dim+1))
noise_variance = np.ones(1+len(derivatives))

hyper_domain = np.zeros((2+dim+len(derivatives), 2))
hyper_domain[:(2+dim),:] = objective_func._hyper_domain[:(2+dim),:]
for g in range(len(derivatives)):
    hyper_domain[g+2+dim, :] = objective_func._hyper_domain[derivatives[g]+2+dim,:]
hyper_search_domain = cppTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in hyper_domain])
hyper_params = np.ones(2+dim+len(derivatives))

cpp_gp = cppGaussianProcess(cpp_cov, hyper_params[(1+objective_func._dim):], init_data, derivatives)

cpp_gp_loglikelihood = cppGaussianProcessLogLikelihood(cpp_cov, cpp_gp.get_historical_data_copy(), noise_variance, derivatives)
sgd_optimizer = cppGradientDescentOptimizer(hyper_search_domain, cpp_gp_loglikelihood, cpp_sgd_params_hyper)
hyper_params = multistart_hyperparameter_optimization(log_likelihood_optimizer=sgd_optimizer, num_multistarts=None, max_num_threads=8)

discrete_pts_optima = python_search_domain.generate_uniform_random_points_in_domain(1000)

points = GP_Approximation_Random_Feature.sample_from_global_optima(cpp_gp, 1000, objective_func._search_domain, discrete_pts_optima, 200)
print points
print cpp_gp.compute_mean_of_points(points)