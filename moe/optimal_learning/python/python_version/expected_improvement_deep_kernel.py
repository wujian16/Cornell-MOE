# -*- coding: utf-8 -*-
"""Classes (Python) to compute the Expected Improvement with scalable deep kernels (monte carlo implementations).

"""
import logging

import numpy

import tensorflow as tf

from moe.optimal_learning.python.constant import DEFAULT_MAX_NUM_THREADS
from moe.optimal_learning.python.python_version.optimization import multistart_optimize, NullOptimizer

def multistart_expected_improvement_optimization(
        ei_optimizer,
        num_multistarts,
        num_to_sample,
        randomness=None,
        max_num_threads=DEFAULT_MAX_NUM_THREADS,
        status=None,
):
    """Solve the q,p-EI problem, returning the optimal set of q points to sample CONCURRENTLY in future experiments.

    When ``points_being_sampled.shape[0] == 0 && num_to_sample == 1``, this function will use (fast) analytic EI computations.

    .. NOTE:: The following comments are copied from
      :func:`moe.optimal_learning.python.cpp_wrappers.expected_improvement.multistart_expected_improvement_optimization`.

    This is the primary entry-point for EI optimization in the optimal_learning library. It offers our best shot at
    improving robustness by combining higher accuracy methods like gradient descent with fail-safes like random/grid search.

    Returns the optimal set of q points to sample CONCURRENTLY by solving the q,p-EI problem.  That is, we may want to run 4
    experiments at the same time and maximize the EI across all 4 experiments at once while knowing of 2 ongoing experiments
    (4,2-EI). This function handles this use case. Evaluation of q,p-EI (and its gradient) for q > 1 or p > 1 is expensive
    (requires monte-carlo iteration), so this method is usually very expensive.

    Compared to ComputeHeuristicPointsToSample() (``gpp_heuristic_expected_improvement_optimization.hpp``), this function
    makes no external assumptions about the underlying objective function. Instead, it utilizes the Expected (Parallel)
    Improvement, allowing the GP to account for ongoing/incomplete experiments.

    If ``num_to_sample = 1``, this is the same as ComputeOptimalPointsToSampleWithRandomStarts().

    TODO(GH-56): Allow callers to pass in a source of randomness.

    :param ei_optimizer: object that optimizes (e.g., gradient descent, newton) EI over a domain
    :type ei_optimizer: interfaces.optimization_interfaces.OptimizerInterface subclass
    :param num_multistarts: number of times to multistart ``ei_optimizer``
    :type num_multistarts: int > 0
    :param num_to_sample: how many simultaneous experiments you would like to run (i.e., the q in q,p-EI) (UNUSED, specify through ei_optimizer)
    :type num_to_sample: int >= 1
    :param randomness: random source(s) used to generate multistart points and perform monte-carlo integration (when applicable) (UNUSED)
    :type randomness: (UNUSED)
    :param max_num_threads: maximum number of threads to use, >= 1 (UNUSED)
    :type max_num_threads: int > 0
    :param status: (output) status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict
    :return: point(s) that maximize the expected improvement (solving the q,p-EI problem)
    :rtype: array of float64 with shape (num_to_sample, ei_evaluator.dim)

    """
    random_starts = ei_optimizer.domain.generate_uniform_random_points_in_domain(num_points=num_multistarts)
    best_point, _ = multistart_optimize(ei_optimizer, starting_points=random_starts)

    # TODO(GH-59): Have GD actually indicate whether updates were found.
    found_flag = True
    if status is not None:
        status["gradient_descent_found_update"] = found_flag

    return best_point


class ExpectedImprovement(object):

    r"""Implementation of Expected Improvement computation in Python: EI and its gradient at specified point(s) sampled from a GaussianProcess.

    A class to encapsulate the computation of expected improvement and its spatial gradient using points sampled from an
    associated GaussianProcess. The general EI computation requires monte-carlo integration; it can support q,p-EI optimization.
    It is designed to work with any GaussianProcess.

    When available, fast, analytic formulas replace monte-carlo loops.

    .. Note:: Equivalent methods of ExpectedImprovementInterface and OptimizableInterface are aliased below (e.g.,
      compute_expected_improvement and compute_objective_function, etc).

    See :class:`moe.optimal_learning.python.interfaces.expected_improvement_interface.ExpectedImprovementInterface` for further details.

    """

    def __init__(
            self,
            scalable_gaussian_process,
            points_to_sample=None,
            points_being_sampled=None,
            num_mc_iterations=100
    ):
        """Construct an ExpectedImprovement object that supports q,p-EI.

        TODO(GH-56): Allow callers to pass in a source of randomness.

        :param gaussian_process: GaussianProcess describing
        :type gaussian_process: interfaces.gaussian_process_interface.GaussianProcessInterface subclass
        :param points_to_sample: points at which to evaluate EI and/or its gradient to check their value in future experiments (i.e., "q" in q,p-EI)
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :param points_being_sampled: points being sampled in concurrent experiments (i.e., "p" in q,p-EI)
        :type points_being_sampled: array of float64 with shape (num_being_sampled, dim)
        :param num_mc_iterations: number of monte-carlo iterations to use (when monte-carlo integration is used to compute EI)
        :type num_mc_iterations: int > 0

        """
        self._num_mc_iterations = num_mc_iterations
        self._gaussian_process = scalable_gaussian_process
        if scalable_gaussian_process._points_sampled_value.size > 0:
            self._best_so_far = numpy.amin(scalable_gaussian_process._points_sampled_value)
        else:
            self._best_so_far = numpy.finfo(numpy.float64).max

        if points_being_sampled is None:
            self._points_being_sampled = numpy.array([])
        else:
            self._points_being_sampled = numpy.copy(points_being_sampled)

        if points_to_sample is None:
            self.current_point = numpy.zeros((1, scalable_gaussian_process.dim))
        else:
            self.current_point = points_to_sample

        self.log = logging.getLogger(__name__)

    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return self._gaussian_process.dim

    @property
    def num_to_sample(self):
        """Number of points at which to compute/optimize EI, aka potential points to sample in future experiments; i.e., the ``q`` in ``q,p-EI``."""
        return self._points_to_sample.shape[0]

    @property
    def num_being_sampled(self):
        """Number of points being sampled in concurrent experiments; i.e., the ``p`` in ``q,p-EI``."""
        return self._points_being_sampled.shape[0]

    @property
    def problem_size(self):
        """Return the number of independent parameters to optimize."""
        return self.num_to_sample * self.dim

    def get_current_point(self):
        """Get the current_point (array of float64 with shape (problem_size)) at which this object is evaluating the objective function, ``f(x)``."""
        return numpy.copy(self._points_to_sample)

    def set_current_point(self, points_to_sample):
        """Set current_point to the specified point; ordering must match.

        :param points_to_sample: current_point at which to evaluate the objective function, ``f(x)``
        :type points_to_sample: array of float64 with shape (problem_size)

        """
        self._points_to_sample = numpy.copy(numpy.atleast_2d(points_to_sample))

    current_point = property(get_current_point, set_current_point)

    def _compute_expected_improvement_monte_carlo(self, mu_star, var_star):
        r"""Compute EI using (vectorized) monte-carlo integration; this is a general method that works for any input.

        This function cal support the computation of q,p-EI.
        This function requires access to a random number generator.

        .. Note:: comments here are copied from gpp_math.cpp, ExpectedImprovementEvaluator::ComputeExpectedImprovement().

        Let ``Ls * Ls^T = Vars`` and ``w`` = vector of IID normal(0,1) variables
        Then:
        ``y = mus + Ls * w``  (Equation 4, from file docs)
        simulates drawing from our GP with mean mus and variance Vars.

        Then as given in the file docs, we compute the improvement:
        Then the improvement for this single sample is:
        ``I = { best_known - min(y)   if (best_known - min(y) > 0)      (Equation 5 from file docs)``
        ``    {          0               else``
        This is implemented as ``max_{y} (best_known - y)``.  Notice that improvement takes the value 0 if it would be negative.

        Since we cannot compute ``min(y)`` directly, we do so via monte-carlo (MC) integration.  That is, we draw from the GP
        repeatedly, computing improvement during each iteration, and averaging the result.

        See Scott's PhD thesis, sec 6.2.

        For performance, this function vectorizes the monte-carlo integration loop, using numpy's mask feature to skip
        iterations where the improvement is not positive.

        Lastly, under some situations (e.g., ``points_to_sample`` and ``points_begin_sampled`` are too close
        together or too close to ``points_sampled``), the GP-Variance matrix, ``Vars`` is
        [numerically] singular so that the cholesky factorization ``Ls * Ls^T = Vars`` cannot
        be computed reliably.

        When this happens (as detected by a numpy/scipy ``LinAlgError``), we instead resort to
        a combination of the SVD and the QR factorization to compute the cholesky factorization
        more reliably. SVD and QR (see code) have extremely numerically stable algorithms.

        :param mu_star: self._gaussian_process.compute_mean_of_points(union_of_points)
        :type mu_star: array of float64 with shape (num_points)
        :param var_star: self._gaussian_process.compute_variance_of_points(union_of_points)
        :type var_star: array of float64 with shape (num_points, num_points)
        :return: the expected improvement from sampling ``points_to_sample`` with ``points_being_sampled`` concurrent experiments
        :rtype: float64

        """
        num_points = self.num_to_sample + self.num_being_sampled
        chol_var = -tf.cholesky(var_star)

        normals = tf.random_normal(shape=(num_points, self._num_mc_iterations))

        # TODO(GH-60): Partition num_mc_iterations up into smaller blocks if it helps.
        # so that we don't waste as much mem bandwidth (since each entry of normals is
        # only used once)
        mu_star = self._best_so_far - mu_star
        # Compute Ls * w; note the shape is (self._num_mc_iterations, num_points)
        improvement_each_iter = tf.matmul(chol_var, normals)
        # Now we have improvement = best_so_far - y = best_so_far - (mus + Ls * w)
        improvement_each_iter += mu_star
        # We want the maximum improvement each step; note the shape is (self._num_mc_iterations)
        improvement_each_iter = tf.maximum(improvement_each_iter, 0.0)
        best_improvement_each_iter = tf.reduce_max(improvement_each_iter, axis=0)
        result = tf.reduce_sum(best_improvement_each_iter) / float(self._num_mc_iterations)
        return result

    def compute_expected_improvement(self):
        r"""Compute the expected improvement at ``points_to_sample``, with ``points_being_sampled`` concurrent points being sampled.

        .. Note:: These comments were copied from
          :meth:`moe.optimal_learning.python.interfaces.expected_improvement_interface.ExpectedImprovementInterface.compute_expected_improvement`.

        ``points_to_sample`` is the "q" and ``points_being_sampled`` is the "p" in q,p-EI.

        Computes the expected improvement ``EI(Xs) = E_n[[f^*_n(X) - min(f(Xs_1),...,f(Xs_m))]^+]``, where ``Xs``
        are potential points to sample (union of ``points_to_sample`` and ``points_being_sampled``) and ``X`` are
        already sampled points.  The ``^+`` indicates that the expression in the expectation evaluates to 0 if it
        is negative.  ``f^*(X)`` is the MINIMUM over all known function evaluations (``points_sampled_value``),
        whereas ``f(Xs)`` are *GP-predicted* function evaluations.

        In words, we are computing the expected improvement (over the current ``best_so_far``, best known
        objective function value) that would result from sampling (aka running new experiments) at
        ``points_to_sample`` with ``points_being_sampled`` concurrent/ongoing experiments.

        In general, the EI expression is complex and difficult to evaluate; hence we use Monte-Carlo simulation to approximate it.
        When faster (e.g., analytic) techniques are available, we will prefer them.

        The idea of the MC approach is to repeatedly sample at the union of ``points_to_sample`` and
        ``points_being_sampled``. This is analogous to gaussian_process_interface.sample_point_from_gp,
        but we sample ``num_union`` points at once:
        ``y = \mu + Lw``
        where ``\mu`` is the GP-mean, ``L`` is the ``chol_factor(GP-variance)`` and ``w`` is a vector
        of ``num_union`` draws from N(0, 1). Then:
        ``improvement_per_step = max(max(best_so_far - y), 0.0)``
        Observe that the inner ``max`` means only the smallest component of ``y`` contributes in each iteration.
        We compute the improvement over many random draws and average.

        :param force_monte_carlo: whether to force monte carlo evaluation (vs using fast/accurate analytic eval when possible)
        :type force_monte_carlo: bool
        :param force_1d_ei: whether to force using the 1EI method. Used for testing purposes only. Takes precedence when force_monte_carlo is also True
        :type force_1d_ei: bool
        :return: the expected improvement from sampling ``points_to_sample`` with ``points_being_sampled`` concurrent experiments
        :rtype: float64

        """
        num_points = self.num_to_sample + self.num_being_sampled
        union_of_points = numpy.reshape(numpy.append(self._points_to_sample, self._points_being_sampled), (num_points, self.dim))

        result = 0.0
        for model in self._gaussian_process:
            mu_star = model.compute_mean_of_points(union_of_points)
            var_star = model.compute_variance_of_points(union_of_points)
            result += self._compute_expected_improvement_monte_carlo(mu_star, var_star)
        result /= float(len(self._gaussian_process))
        return result

    compute_objective_function = compute_expected_improvement

    def _compute_grad_expected_improvement_monte_carlo(self, scalable_gp, normals, index):
        r"""Compute the gradient of EI using (vectorized) monte-carlo integration; this is a general method that works for any input.

        :param mu_star: self._gaussian_process.compute_mean_of_points(union_of_points)
        :type mu_star: array of float64 with shape (num_points)
        :param var_star: self._gaussian_process.compute_variance_of_points(union_of_points)
        :type var_star: array of float64 with shape (num_points, num_points)
        :param grad_mu: self._gaussian_process.compute_grad_mean_of_points(union_of_points)
        :type grad_mu: array of float64 with shape (num_points, self.dim)
        :param grad_chol_decomp: self._gaussian_process.compute_grad_cholesky_variance_of_points(union_of_points)
        :type grad_chol_decomp: array of float64 with shape (self.num_to_sample, num_points, num_points, self.dim)
        :return: gradient of EI, ``\pderiv{EI(Xq \cup Xp)}{Xq_{i,d}}`` where ``Xq`` is ``points_to_sample``
          and ``Xp`` is ``points_being_sampled`` (grad EI from sampling ``points_to_sample`` with
          ``points_being_sampled`` concurrent experiments wrt each dimension of the points in ``points_to_sample``)
        :rtype: array of float64 with shape (self.num_to_sample, self.dim)
        """
        num_points = self.num_to_sample + self.num_being_sampled
        union_of_points = numpy.reshape(numpy.append(self._points_to_sample, self._points_being_sampled), (num_points, self.dim))
        mu_star = scalable_gp.compute_mean_of_points(union_of_points)
        var_star = scalable_gp.compute_variance_of_points(union_of_points)
        chol_var = -tf.cholesky(var_star)
        improvement = mu_star[index] - tf.matmul(chol_var, normals)[index]
        aggregate_dx = tf.gradients(improvement, self._points_to_sample)
        return aggregate_dx

    def compute_grad_expected_improvement(self):
        r"""Compute the gradient of expected improvement at ``points_to_sample`` wrt ``points_to_sample``, with ``points_being_sampled`` concurrent samples.

        .. Note:: These comments were copied from
          :meth:`moe.optimal_learning.python.interfaces.expected_improvement_interface.ExpectedImprovementInterface.compute_grad_expected_improvement`.

        ``points_to_sample`` is the "q" and ``points_being_sampled`` is the "p" in q,p-EI.

        In general, the expressions for gradients of EI are complex and difficult to evaluate; hence we use
        Monte-Carlo simulation to approximate it. When faster (e.g., analytic) techniques are available, we will prefer them.

        The MC computation of grad EI is similar to the computation of EI (decsribed in
        compute_expected_improvement). We differentiate ``y = \mu + Lw`` wrt ``points_to_sample``;
        only terms from the gradient of ``\mu`` and ``L`` contribute. In EI, we computed:
        ``improvement_per_step = max(max(best_so_far - y), 0.0)``
        and noted that only the smallest component of ``y`` may contribute (if it is > 0.0).
        Call this index ``winner``. Thus in computing grad EI, we only add gradient terms
        that are attributable to the ``winner``-th component of ``y``.

        :param force_monte_carlo: whether to force monte carlo evaluation (vs using fast/accurate analytic eval when possible)
        :type force_monte_carlo: boolean
        :return: gradient of EI, ``\pderiv{EI(Xq \cup Xp)}{Xq_{i,d}}`` where ``Xq`` is ``points_to_sample``
          and ``Xp`` is ``points_being_sampled`` (grad EI from sampling ``points_to_sample`` with
          ``points_being_sampled`` concurrent experiments wrt each dimension of the points in ``points_to_sample``)
        :rtype: array of float64 with shape (num_to_sample, dim)

        """
        pass


    compute_grad_objective_function = compute_grad_expected_improvement

    def compute_hessian_objective_function(self, **kwargs):
        """We do not currently support computation of the (spatial) hessian of Expected Improvement."""
        raise NotImplementedError('Currently we cannot compute the hessian of expected improvement.')