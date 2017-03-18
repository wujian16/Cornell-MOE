# -*- coding: utf-8 -*-
"""Tools to compute KnowledgeGradient and optimize the next best point(s) to sample using KG through C++ calls.

This file contains a class to compute Knowledge Gradient + derivatives and a functions to solve the q,p-KG optimization problem.
The :class:`moe.optimal_learning.python.cpp_wrappers.knowledge_gradient.KnowledgeGradient`
The optimization functions are convenient wrappers around the matching C++ calls.

See gpp_knowledge_gradient_optimization.hpp/cpp for further details on knowledge gradient.

"""
import numpy

import moe.build.GPP as C_GP
from moe.optimal_learning.python.constant import DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS, DEFAULT_MAX_NUM_THREADS
import moe.optimal_learning.python.cpp_wrappers.cpp_utils as cpp_utils
from moe.optimal_learning.python.interfaces.optimization_interface import OptimizableInterface


class PosteriorMeanMCMC(OptimizableInterface):
    def __init__(
            self,
            gaussian_process_list,
            points_to_sample=None,
            randomness=None,
    ):
        self._gaussian_process_list = gaussian_process_list

        if points_to_sample is None:
            self._points_to_sample = numpy.zeros((1, self._gaussian_process_list[0].dim))

        if randomness is None:
            self._randomness = C_GP.RandomnessSourceContainer(1)  # create randomness for only 1 thread
            # Set seed based on less repeatable factors (e.g,. time)
            self._randomness.SetRandomizedUniformGeneratorSeed(0)
            self._randomness.SetRandomizedNormalRNGSeed(0)
        else:
            self._randomness = randomness

        self.objective_type = None  # Not used for KG, but the field is expected in C++

    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return self._gaussian_process_list[0].dim

    @property
    def problem_size(self):
        """Return the number of independent parameters to optimize."""
        return self.dim

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

    def compute_posterior_mean_mcmc(self, force_monte_carlo=False):
        r"""Compute the knowledge gradient at ``points_to_sample``, with ``points_being_sampled`` concurrent points being sampled.

        .. Note:: These comments were copied from
          :meth:`moe.optimal_learning.python.interfaces.expected_improvement_interface.ExpectedImprovementInterface.compute_expected_improvement`

        ``points_to_sample`` is the "q" and ``points_being_sampled`` is the "p" in q,p-EI.

        Computes the knowledge gradient ``EI(Xs) = E_n[[f^*_n(X) - min(f(Xs_1),...,f(Xs_m))]^+]``, where ``Xs``
        are potential points to sample (union of ``points_to_sample`` and ``points_being_sampled``) and ``X`` are
        already sampled points.  The ``^+`` indicates that the expression in the expectation evaluates to 0 if it
        is negative.  ``f^*(X)`` is the MINIMUM over all known function evaluations (``points_sampled_value``),
        whereas ``f(Xs)`` are *GP-predicted* function evaluations.

        In words, we are computing the knowledge gradient (over the current ``best_so_far``, best known
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
        :type force_monte_carlo: boolean
        :return: the knowledge gradient from sampling ``points_to_sample`` with ``points_being_sampled`` concurrent experiments
        :rtype: float64

        """
        posterior_mean_mcmc = 0
        for gp in self._gaussian_process_list:
            posterior_mean_mcmc += C_GP.compute_posterior_mean(
                                        gp._gaussian_process,
                                        cpp_utils.cppify(self._points_to_sample),
                                       )
        return posterior_mean_mcmc/len(self._gaussian_process_list)

    compute_objective_function = compute_posterior_mean_mcmc

    def compute_grad_posterior_mean_mcmc(self, force_monte_carlo=False):
        r"""Compute the gradient of knowledge gradient at ``points_to_sample`` wrt ``points_to_sample``, with ``points_being_sampled`` concurrent samples.

        .. Note:: These comments were copied from
          :meth:`moe.optimal_learning.python.interfaces.expected_improvement_interface.ExpectedImprovementInterface.compute_grad_expected_improvement`

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
        grad_posterior_mean_mcmc = numpy.zeros((1, self.dim))
        for gp in self._gaussian_process_list:
            temp = C_GP.compute_grad_posterior_mean(
                    gp._gaussian_process,
                    cpp_utils.cppify(self._points_to_sample),
            )
            grad_posterior_mean_mcmc += cpp_utils.uncppify(temp, (1, self.dim))
        return grad_posterior_mean_mcmc/len(self._gaussian_process_list)

    compute_grad_objective_function = compute_grad_posterior_mean_mcmc

    def compute_hessian_objective_function(self, **kwargs):
        """We do not currently support computation of the (spatial) hessian of knowledge gradient."""
        raise NotImplementedError('Currently we cannot compute the hessian of knowledge gradient.')


class KnowledgeGradientMCMC(OptimizableInterface):

    r"""Implementation of knowledge gradient computation via C++ wrappers: EI and its gradient at specified point(s) sampled from a GaussianProcess.

    A class to encapsulate the computation of knowledge gradient and its spatial gradient using points sampled from an
    associated GaussianProcess. The general EI computation requires monte-carlo integration; it can support q,p-EI optimization.
    It is designed to work with any GaussianProcess.

    .. Note:: Equivalent methods of ExpectedImprovementInterface and OptimizableInterface are aliased below (e.g.,
      compute_expected_improvement and compute_objective_function, etc).

    See :mod:`moe.optimal_learning.python.interfaces.expected_improvement_interface` docs for further details.

    """

    def __init__(
            self,
            gaussian_process_list,
            inner_optimizer,
            discrete_pts_list,
            num_to_sample,
            points_to_sample=None,
            points_being_sampled=None,
            num_mc_iterations=DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS,
            randomness=None,
    ):
        """Construct a KnowledgeGradient object that supports q,p-KG.
        TODO(GH-56): Allow callers to pass in a source of randomness.
        :param gaussian_process: GaussianProcess describing
        :type gaussian_process: interfaces.gaussian_process_interface.GaussianProcessInterface subclass
        :param discrete_pts: a discrete set of points to approximate the KG
        :type discrete_pts: array of float64 with shape (num_pts, dim)
        :param noise: measurement noise
        :type noise: float64
        :param points_to_sample: points at which to evaluate KG and/or its gradient to check their value in future experiments (i.e., "q" in q,p-KG)
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :param points_being_sampled: points being sampled in concurrent experiments (i.e., "p" in q,p-KG)
        :type points_being_sampled: array of float64 with shape (num_being_sampled, dim)
        :param num_mc_iterations: number of monte-carlo iterations to use (when monte-carlo integration is used to compute KG)
        :type num_mc_iterations: int > 0
        :param randomness: random source(s) used for monte-carlo integration (when applicable) (UNUSED)
        :type randomness: (UNUSED)
        """
        self._num_mc_iterations = num_mc_iterations
        self._gaussian_process_list = gaussian_process_list
        self._num_to_sample = num_to_sample
        self._inner_optimizer = inner_optimizer

        # self._num_derivatives = gaussian_process._historical_data.num_derivatives
        self._discrete_pts_list = []
        self._best_so_far_list = []
        for discrete_pts, gaussian_process in zip(discrete_pts_list, self._gaussian_process_list):
            self._discrete_pts_list.append(numpy.copy(discrete_pts))
            mu_star = gaussian_process.compute_mean_of_additional_points(discrete_pts)
            self._best_so_far_list.append(numpy.amin(mu_star))

        if points_being_sampled is None:
            self._points_being_sampled = numpy.array([])
        else:
            self._points_being_sampled = numpy.copy(points_being_sampled)

        if points_to_sample is None:
            self._points_to_sample = numpy.zeros((self._num_to_sample, self._gaussian_process_list[0].dim))
        else:
            self._points_to_sample = points_to_sample

        self._num_to_sample = self._points_to_sample.shape[0]

        if randomness is None:
            self._randomness = C_GP.RandomnessSourceContainer(1)  # create randomness for only 1 thread
            # Set seed based on less repeatable factors (e.g,. time)
            self._randomness.SetRandomizedUniformGeneratorSeed(0)
            self._randomness.SetRandomizedNormalRNGSeed(0)
        else:
            self._randomness = randomness

        self.objective_type = None  # Not used for KG, but the field is expected in C++

    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return self._gaussian_process_list[0].dim

    @property
    def num_to_sample(self):
        """Number of points at which to compute/optimize KG, aka potential points to sample in future experiments; i.e., the ``q`` in ``q,p-kg``."""
        return self._points_to_sample.shape[0]

    @property
    def num_being_sampled(self):
        """Number of points being sampled in concurrent experiments; i.e., the ``p`` in ``q,p-KG``."""
        return self._points_being_sampled.shape[0]

    @property
    def discrete(self):
        return self._discrete_pts_list[0].shape[0]

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

    def evaluate_at_point_list(
            self,
            points_to_evaluate,
            randomness=None,
            max_num_threads=DEFAULT_MAX_NUM_THREADS,
            status=None,
    ):
        """Evaluate knowledge gradient (1,p-EI) over a specified list of ``points_to_evaluate``.

        .. Note:: We use ``points_to_evaluate`` instead of ``self._points_to_sample`` and compute the EI at those points only.
            ``self._points_to_sample`` is unchanged.

        Generally gradient descent is preferred but when they fail to converge this may be the only "robust" option.
        This function is also useful for plotting or debugging purposes (just to get a bunch of EI values).

        :param points_to_evaluate: points at which to compute EI
        :type points_to_evaluate: array of float64 with shape (num_to_evaluate, self.dim)
        :param randomness: RNGs used by C++ to generate initial guesses and as the source of normal random numbers when monte-carlo is used
        :type randomness: RandomnessSourceContainer (C++ object; e.g., from C_GP.RandomnessSourceContainer())
        :param max_num_threads: maximum number of threads to use, >= 1
        :type max_num_threads: int > 0
        :param status: (output) status messages from C++ (e.g., reporting on optimizer success, etc.)
        :type status: dict
        :return: EI evaluated at each of points_to_evaluate
        :rtype: array of float64 with shape (points_to_evaluate.shape[0])

        """
        # Create enough randomness sources if none are specified.
        if randomness is None:
            if max_num_threads == 1:
                randomness = self._randomness
            else:
                randomness = C_GP.RandomnessSourceContainer(max_num_threads)
                # Set seeds based on less repeatable factors (e.g,. time)
                randomness.SetRandomizedUniformGeneratorSeed(0)
                randomness.SetRandomizedNormalRNGSeed(0)

        # status must be an initialized dict for the call to C++.
        if status is None:
            status = {}

        # num_to_sample need not match ei_evaluator.num_to_sample since points_to_evaluate
        # overrides any data inside ei_evaluator
        num_to_evaluate, num_to_sample, _ = points_to_evaluate.shape

        kg_values_mcmc = numpy.zeros(num_to_evaluate)
        for gp, discrete_pts, best_so_far in zip(self._gaussian_process_list, self._discrete_pts_list, self._best_so_far_list):
            kg_values = C_GP.evaluate_KG_at_point_list(
                    gp._gaussian_process,
                    self._inner_optimizer.optimizer_parameters,
                    cpp_utils.cppify(self._inner_optimizer.domain.domain_bounds),
                    cpp_utils.cppify(discrete_pts),
                    cpp_utils.cppify(points_to_evaluate),
                    cpp_utils.cppify(self._points_being_sampled),
                    num_to_evaluate,
                    self.discrete,
                    num_to_sample,
                    self.num_being_sampled,
                    best_so_far,
                    self._num_mc_iterations,
                    max_num_threads,
                    randomness,
                    status,
            )
            kg_values_mcmc += numpy.array(kg_values)
        kg_values_mcmc /= len(self._gaussian_process_list)
        return kg_values_mcmc

    def compute_knowledge_gradient_mcmc(self, force_monte_carlo=False):
        r"""Compute the knowledge gradient at ``points_to_sample``, with ``points_being_sampled`` concurrent points being sampled.

        .. Note:: These comments were copied from
          :meth:`moe.optimal_learning.python.interfaces.expected_improvement_interface.ExpectedImprovementInterface.compute_expected_improvement`

        ``points_to_sample`` is the "q" and ``points_being_sampled`` is the "p" in q,p-EI.

        Computes the knowledge gradient ``EI(Xs) = E_n[[f^*_n(X) - min(f(Xs_1),...,f(Xs_m))]^+]``, where ``Xs``
        are potential points to sample (union of ``points_to_sample`` and ``points_being_sampled``) and ``X`` are
        already sampled points.  The ``^+`` indicates that the expression in the expectation evaluates to 0 if it
        is negative.  ``f^*(X)`` is the MINIMUM over all known function evaluations (``points_sampled_value``),
        whereas ``f(Xs)`` are *GP-predicted* function evaluations.

        In words, we are computing the knowledge gradient (over the current ``best_so_far``, best known
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
        :type force_monte_carlo: boolean
        :return: the knowledge gradient from sampling ``points_to_sample`` with ``points_being_sampled`` concurrent experiments
        :rtype: float64

        """
        knowledge_gradient_mcmc = 0.0
        for gp, discrete_pts, best_so_far in zip(self._gaussian_process_list, self._discrete_pts_list, self._best_so_far_list):
            knowledge_gradient_mcmc += C_GP.compute_knowledge_gradient(
                    gp._gaussian_process,
                    self._inner_optimizer.optimizer_parameters,
                    cpp_utils.cppify(self._inner_optimizer.domain.domain_bounds),
                    cpp_utils.cppify(discrete_pts),
                    cpp_utils.cppify(self._points_to_sample),
                    cpp_utils.cppify(self._points_being_sampled),
                    self.discrete,
                    self.num_to_sample,
                    self.num_being_sampled,
                    self._num_mc_iterations,
                    best_so_far,
                    self._randomness,
            )
        return (knowledge_gradient_mcmc/len(self._gaussian_process_list))

    compute_objective_function = compute_knowledge_gradient_mcmc

    def compute_grad_knowledge_gradient_mcmc(self, force_monte_carlo=False):
        r"""Compute the gradient of knowledge gradient at ``points_to_sample`` wrt ``points_to_sample``, with ``points_being_sampled`` concurrent samples.

        .. Note:: These comments were copied from
          :meth:`moe.optimal_learning.python.interfaces.expected_improvement_interface.ExpectedImprovementInterface.compute_grad_expected_improvement`

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
        grad_knowledge_gradient_mcmc = numpy.zeros((self.num_to_sample, self.dim))
        for gp, discrete_pts, best_so_far in zip(self._gaussian_process_list, self._discrete_pts_list, self._best_so_far_list):
            temp = C_GP.compute_grad_knowledge_gradient(
                    gp._gaussian_process,
                    self._inner_optimizer.optimizer_parameters,
                    cpp_utils.cppify(self._inner_optimizer.domain.domain_bounds),
                    cpp_utils.cppify(discrete_pts),
                    cpp_utils.cppify(self._points_to_sample),
                    cpp_utils.cppify(self._points_being_sampled),
                    self.discrete,
                    self.num_to_sample,
                    self.num_being_sampled,
                    self._num_mc_iterations,
                    best_so_far,
                    self._randomness,
            )
            grad_knowledge_gradient_mcmc += cpp_utils.uncppify(temp, (self.num_to_sample, self.dim))
        return (grad_knowledge_gradient_mcmc/len(self._gaussian_process_list))

    compute_grad_objective_function = compute_grad_knowledge_gradient_mcmc

    def compute_hessian_objective_function(self, **kwargs):
        """We do not currently support computation of the (spatial) hessian of knowledge gradient."""
        raise NotImplementedError('Currently we cannot compute the hessian of knowledge gradient.')