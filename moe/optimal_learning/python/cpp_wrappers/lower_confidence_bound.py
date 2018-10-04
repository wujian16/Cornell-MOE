# -*- coding: utf-8 -*-
"""Tools to compute LCB and optimize the next best point(s) to sample using LCB through C++ calls.

This file contains a class to compute  + derivatives and a functions to solve the q,p-KG optimization problem.
The :class:`moe.optimal_learning.python.cpp_wrappers.knowledge_gradient.KnowledgeGradient`
The optimization functions are convenient wrappers around the matching C++ calls.

See gpp_knowledge_gradient_optimization.hpp/cpp for further details on knowledge gradient.

"""
from builtins import range
import numpy

from moe.optimal_learning.python.data_containers import SamplePoint


def lower_confidence_bound_optimization(
        gaussian_process,
        candidate_pts,
        num_to_sample,
):
    """Solve the q,p-LCB problem, returning the optimal set of q points to sample CONCURRENTLY in future experiments.

    .. NOTE:: The following comments are copied from gpp_math.hpp, ComputeOptimalPointsToSample().
      These comments are copied into
      :func:`moe.optimal_learning.python.python_version.expected_improvement.multistart_expected_improvement_optimization`

    This is the primary entry-point for EI optimization in the optimal_learning library. It offers our best shot at
    improving robustness by combining higher accuracy methods like gradient descent with fail-safes like random/grid search.

    Returns the optimal set of q points to sample CONCURRENTLY by solving the q,p-EI problem.  That is, we may want to run 4
    experiments at the same time and maximize the EI across all 4 experiments at once while knowing of 2 ongoing experiments
    (4,2-EI). This function handles this use case. Evaluation of q,p-EI (and its gradient) for q > 1 or p > 1 is expensive
    (requires monte-carlo iteration), so this method is usually very expensive.

    Compared to ComputeHeuristicPointsToSample() (``gpp_heuristic_expected_improvement_optimization.hpp``), this function
    makes no external assumptions about the underlying objective function. Instead, it utilizes a feature of the
    GaussianProcess that allows the GP to account for ongoing/incomplete experiments.

    If ``num_to_sample = 1``, this is the same as ComputeOptimalPointsToSampleWithRandomStarts().

    The option of using GPU to compute general q,p-EI via MC simulation is also available. To enable it, make sure you have
    installed GPU components of MOE, otherwise, it will throw Runtime excpetion.

    :param num_to_sample: how many simultaneous experiments you would like to run (i.e., the q in q,p-EI)
    :type num_to_sample: int >= 1
    :return: point(s) that maximize the knowledge gradient (solving the q,p-KG problem)
    :rtype: array of float64 with shape (num_to_sample, ei_optimizer.objective_function.dim)

    """
    # Create enough randomness sources if none are specified.
    mean_surface = gaussian_process.compute_mean_of_points(candidate_pts)
    standard_deviation = numpy.zeros(candidate_pts.shape[0])
    for pt in range(candidate_pts.shape[0]):
        standard_deviation[pt] = gaussian_process.compute_cholesky_variance_of_points(candidate_pts[[pt],:])[0,0]

    target = mean_surface - standard_deviation
    index = numpy.argmin(target)

    ucb = mean_surface + standard_deviation
    upper_bound = numpy.min(ucb)
    condition = target <= upper_bound
    satisfied_candidate_pts = candidate_pts[condition,:]
    satisfied_standard_deviation = numpy.zeros(satisfied_candidate_pts.shape[0])

    results = numpy.zeros((num_to_sample, gaussian_process.dim))
    results[0] = candidate_pts[index]

    for i in range(1, num_to_sample):
        sample_point = [SamplePoint(results[i-1],
                        numpy.zeros(gaussian_process.num_derivatives+1),
                        0.25)]
        gaussian_process.add_sampled_points(sample_point)
        for pt in range(satisfied_standard_deviation.shape[0]):
            satisfied_standard_deviation[pt] = gaussian_process.compute_cholesky_variance_of_points(satisfied_candidate_pts[[pt],:])[0,0]

        index = numpy.argmax(satisfied_standard_deviation)
        results[i] = satisfied_candidate_pts[index]

    return results, 0.0

