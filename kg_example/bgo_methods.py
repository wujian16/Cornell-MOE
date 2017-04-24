import numpy

from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient_mcmc import KnowledgeGradientMCMC as cppKnowledgeGradientMCMC

from moe.optimal_learning.python.python_version.optimization import GradientDescentOptimizer as pyGradientDescentOptimizer
from moe.optimal_learning.python.python_version.optimization import multistart_optimize as multistart_optimize

from moe.optimal_learning.python.repeated_domain import RepeatedDomain

def gen_sample_from_qkg_mcmc(cpp_gp_list, inner_optimizer, py_search_domain, discrete_pts_list, sgd_params, num_to_sample, num_mc=1e3, lhc_itr=1e3):
    """
    :param cpp_gp: trained cpp version of GaussianProcess model
    :param cpp_search_domain: cpp version of TensorProductDomain
    :param sgd_params: GradientDescentParameters
    :param num_to_sample: number of points to sample for the next iteration
    :param use_gpu: bool, whether to use gpu
    :param which_gpu: gpu device number
    :param num_mc: number of Monte Carlo iterations
    :param lhc_itr: number of points used in latin hypercube search
    :return: (points to sample next, knowledge gradient at this set of points)
    """
    cpp_kg_evaluator = cppKnowledgeGradientMCMC(gaussian_process_list=cpp_gp_list, inner_optimizer = inner_optimizer, discrete_pts_list=discrete_pts_list,
                                                num_to_sample = num_to_sample, num_mc_iterations=int(num_mc))
    py_repeated_search_domain = RepeatedDomain(num_repeats = num_to_sample, domain = py_search_domain)

    points_list = py_repeated_search_domain.generate_uniform_random_points_in_domain(int(1e2))
    max_kg = -float("inf")
    initial_point = None
    for points in points_list:
        cpp_kg_evaluator.set_current_point(points)
        temp = cpp_kg_evaluator.compute_objective_function()
        if temp > max_kg:
            max_kg = temp
            initial_point = points
    #kgs = cpp_kg_evaluator.evaluate_at_point_list(points_list)

    optimizer = pyGradientDescentOptimizer(py_repeated_search_domain, cpp_kg_evaluator, sgd_params, int(lhc_itr))
    points_to_sample_list = []
    kg_list = []

    points_to_sample_list.append(multistart_optimize(optimizer, numpy.array([initial_point]), num_multistarts = 1)[0])
    cpp_kg_evaluator.set_current_point(points_to_sample_list[0])
    kg_list.append(cpp_kg_evaluator.compute_objective_function())
    return points_to_sample_list[numpy.argmax(kg_list)], numpy.amax(kg_list)



