import numpy

from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient_mcmc import KnowledgeGradientMCMC as cppKnowledgeGradientMCMC
from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient_mcmc import multistart_knowledge_gradient_mcmc_optimization

from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentOptimizer as cppGradientDescentOptimizer

def gen_sample_from_qkg_mcmc(cpp_gp_mcmc, cpp_gp_list, inner_optimizer, cpp_search_domain,
                             discrete_pts_list, sgd_params, num_to_sample, num_mc=1e3, lhc_itr=1e3):
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
    cpp_kg_evaluator = cppKnowledgeGradientMCMC(gaussian_process_mcmc = cpp_gp_mcmc, gaussian_process_list=cpp_gp_list,
                                                num_fidelity = 0, inner_optimizer = inner_optimizer, discrete_pts_list=discrete_pts_list,
                                                num_to_sample = num_to_sample, num_mc_iterations=int(num_mc))
    optimizer = cppGradientDescentOptimizer(cpp_search_domain, cpp_kg_evaluator, sgd_params, int(lhc_itr))
    points_to_sample_list = []
    kg_list = []

    points_to_sample_list.append(multistart_knowledge_gradient_mcmc_optimization(optimizer, inner_optimizer, None, discrete_pts_list,
                                                                                 num_to_sample=num_to_sample,
                                                                                 num_pts=discrete_pts_list[0].shape[0],
                                                                                 max_num_threads=8))

    cpp_kg_evaluator.set_current_point(points_to_sample_list[0])
    kg_list.append(cpp_kg_evaluator.compute_objective_function())
    return points_to_sample_list[numpy.argmax(kg_list)], numpy.amax(kg_list)



