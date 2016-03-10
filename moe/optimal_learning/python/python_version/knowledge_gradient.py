# -*- coding: utf-8 -*-
"""
Classes (Python) to compute the knowledge gradient (monte carlo)
"""

from collections import namedtuple
import logging

import numpy

import scipy.linalg
import scipy.stats

from moe.optimal_learning.python.constant import DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS, DEFAULT_MAX_NUM_THREADS
from moe.optimal_learning.python.interfaces.optimization_interface import OptimizableInterface
#from moe.optimal_learning.python.python_version.gaussian_process import MINIMUM_STD_DEV_GRAD_CHOLESKY
from moe.optimal_learning.python.python_version.optimization import multistart_optimize, NullOptimizer


# See MVNDSTParameters (below) for docstring.
_BaseMVNDSTParameters = namedtuple('_BaseMVNDSTParameters', [
    'releps',
    'abseps',
    'maxpts_per_dim',
])


class MVNDSTParameters(_BaseMVNDSTParameters):

    """Container to hold parameters that specify the behavior of mvndst, which qKG uses to calculate KG.
    For more information about these parameters, consult: http://www.math.wsu.edu/faculty/genz/software/fort77/mvndstpack.f
    .. NOTE:: The actual accuracy used in mvndst is MAX(abseps, FINEST * releps), where FINEST is the estimate of the cdf integral.
        Because of this, it is almost always the case that abseps should be set to 0 for releps to be used.
    :ivar releps: (*float > 0.0*) relative accuracy at which to calculate the cdf of the multivariate gaussian (suggest: 1.0e-9)
    :ivar abseps: (*float > 0.0*) absolute accuracy at which to calculate the cdf of the multivariate gaussian (suggest: 1.0e-9)
    :ivar maxpts_per_dim: (*int > 0*) the maximum number of iterations mvndst will do is num_dimensions * maxpts_per_dim (suggest: 20000)
    """

    __slots__ = ()


DEFAULT_MVNDST_PARAMS = MVNDSTParameters(
        releps=1.0e-9,
        abseps=1.0e-9,
        maxpts_per_dim=20000,
        )



def multistart_knowledge_gradient_optimization(
        kg_optimizer,
        num_multistarts,
        num_to_sample,
        randomness=None,
        max_num_threads=DEFAULT_MAX_NUM_THREADS,
        status=None,
):
    """Solve the q,p-KG problem, returning the optimal set of q points to sample CONCURRENTLY in future experiments.
    .. NOTE:: The following comments are modified from
      :func:`moe.optimal_learning.python.cpp_wrappers.expected_improvement.multistart_expected_improvement_optimization`.
    This is the primary entry-point for KG optimization in the optimal_learning library. It offers our best shot at
    improving robustness by combining higher accuracy methods like gradient descent with fail-safes like random/grid search.
    Returns the optimal set of q points to sample CONCURRENTLY by solving the q,p-KG problem.  That is, we may want to run 4
    experiments at the same time and maximize the KG across all 4 experiments at once while knowing of 2 ongoing experiments
    (4,2-KG). This function handles this use case. Evaluation of q,p-KG (and its gradient) for q > 1 or p > 1 is expensive
    (requires monte-carlo iteration), so this method is usually very expensive.

    TODO(GH-56): Allow callers to pass in a source of randomness.
    :param kg_optimizer: object that optimizes (e.g., gradient descent, newton) KG over a domain
    :type kg_optimizer: interfaces.optimization_interfaces.OptimizerInterface subclass
    :param num_multistarts: number of times to multistart ``ei_optimizer``
    :type num_multistarts: int > 0
    :param num_to_sample: how many simultaneous experiments you would like to run (i.e., the q in q,p-KG) (UNUSED, specify through ei_optimizer)
    :type num_to_sample: int >= 1
    :param randomness: random source(s) used to generate multistart points and perform monte-carlo integration (when applicable) (UNUSED)
    :type randomness: (UNUSED)
    :param max_num_threads: maximum number of threads to use, >= 1 (UNUSED)
    :type max_num_threads: int > 0
    :param status: (output) status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict
    :return: point(s) that maximize the knowledge gradient (solving the q,p-KG problem)
    :rtype: array of float64 with shape (num_to_sample, ei_evaluator.dim)
    """
    random_starts = kg_optimizer.domain.generate_uniform_random_points_in_domain(num_points=num_multistarts)
    best_point, _ = multistart_optimize(kg_optimizer, starting_points=random_starts)

    # TODO(GH-59): Have GD actually indicate whether updates were found.
    found_flag = True
    if status is not None:
        status["gradient_descent_found_update"] = found_flag

    return best_point


class KnowledgeGradient(OptimizableInterface):

    r"""Implementation of Knowledge Gradient computation in Python: KG and its gradient at specified point(s) sampled from a GaussianProcess.
    A class to encapsulate the computation of knowledge gradient and its spatial gradient using points sampled from an
    associated GaussianProcess. The general KG computation requires monte-carlo integration; it can support q,p-KG optimization.
    It is designed to work with any GaussianProcess.
    """

    def __init__(
            self,
            gaussian_process,
            discrete_pts,
            noise,
            num_to_sample,
            points_to_sample=None,
            points_being_sampled=None,
            num_mc_iterations=DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS,
            randomness=None,
            mvndst_parameters=None
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
        self._gaussian_process = gaussian_process
        self._discrete_pts = numpy.copy(discrete_pts)
        self._noise = noise
        self._num_to_sample = num_to_sample
        self._mu_star = self._gaussian_process.compute_mean_of_points(self._discrete_pts)
        self._best_so_far = numpy.amin(self._mu_star)

        if points_being_sampled is None:
            self._points_being_sampled = numpy.array([])
        else:
            self._points_being_sampled = numpy.copy(points_being_sampled)

        if points_to_sample is None:
            self._points_to_sample = numpy.zeros((self._num_to_sample, self._gaussian_process.dim))
        else:
            self._points_to_sample = points_to_sample

        if mvndst_parameters is None:
            self._mvndst_parameters = DEFAULT_MVNDST_PARAMS
        else:
            self._mvndst_parameters = mvndst_parameters

        self.log = logging.getLogger(__name__)
        self.objective_type = None  # Not used for KG, but the field is expected in C++
 
    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return self._gaussian_process.dim

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
        return self._discrete_pts.shape[0]

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
        """Evaluate Knowledge Gradient (q,p-KG) over a specified list of ``points_to_evaluate``.
        .. Note:: We use ``points_to_evaluate`` instead of ``self._points_to_sample`` and compute the KG at those points only.
            ``self._points_to_sample`` will be changed.
        Generally gradient descent is preferred but when it fails to converge this may be the only "robust" option.
        This function is also useful for plotting or debugging purposes (just to get a bunch of KG values).
        TODO(GH-56): Allow callers to pass in a source of randomness.
        :param ei_evaluator: object specifying how to evaluate the expected improvement
        :type ei_evaluator: interfaces.expected_improvement_interface.ExpectedImprovementInterface subclass
        :param points_to_evaluate: points at which to compute KG
        :type points_to_evaluate: array of float64 with shape (num_to_evaluate, num_to_sample, ei_evaluator.dim)
        :param randomness: random source(s) used for monte-carlo integration (when applicable) (UNUSED)
        :type randomness: (UNUSED)
        :param max_num_threads: maximum number of threads to use, >= 1 (UNUSED)
        :type max_num_threads: int > 0
        :param status: (output) status messages from C++ (e.g., reporting on optimizer success, etc.)
        :type status: dict
        :return: KG evaluated at each of points_to_evaluate
        :rtype: array of float64 with shape (points_to_evaluate.shape[0])
        """
        null_optimizer = NullOptimizer(None, self)
        _, values = multistart_optimize(null_optimizer, starting_points=points_to_evaluate)

        # TODO(GH-59): Have multistart actually indicate whether updates were found.
        found_flag = True
        if status is not None:
            status["evaluate_EI_at_point_list"] = found_flag

        return values

    def _compute_knowledge_gradient_naive(self, force_monte_carlo=True, force_1d_ei=False):
        r"""
        Compute the knowledge gradient at ``points_to_sample``, with ``points_being_sampled`` concurrent points being sampled.
        :return: the knowledge gradient from sampling ``points_to_sample`` with ``points_being_sampled`` concurrent experiments
        :rtype: float64
        """
        noise=self._noise

        num_points = self.num_to_sample + self.num_being_sampled
        union_of_points = numpy.reshape(numpy.append(self._points_to_sample, self._points_being_sampled), (num_points, self.dim))
        num_combination = num_points+self.discrete
        combination = numpy.reshape(numpy.append(union_of_points, self._discrete_pts), (num_combination, self.dim))

        var_star = self._gaussian_process.compute_covariance_of_points(combination, union_of_points)
        K=var_star[num_points:num_combination, :num_points]
        chol_var = scipy.linalg.cholesky(var_star[:num_points, :num_points]+numpy.diag(noise*numpy.ones(num_points)), lower=True, overwrite_a=True)
        #sigma = numpy.mat(K) * (((numpy.mat(chol_var)).T).I)
        sigma = (scipy.linalg.solve_triangular(
                chol_var,
                K.T,
                lower=True,
                overwrite_b=True,
        )).T

        normals = numpy.random.normal(size=(self._num_mc_iterations, num_points))

        aggregate = 0.0
        for normal_draws in normals:
            sample_values_this_iter = self._mu_star + numpy.dot(sigma, normal_draws.T)
            improvement_this_iter = numpy.amax(self._best_so_far-sample_values_this_iter)
            aggregate += improvement_this_iter
        return aggregate / float(self._num_mc_iterations)

    def _compute_knowledge_gradient_monte_carlo(self, force_monte_carlo=True, force_1d_ei=False):
        r"""Compute KG using (vectorized) monte-carlo integration; this is a general method that works for any input.
        This function cal support the computation of q,p-KG.
        This function requires access to a random number generator.

        For performance, this function vectorizes the monte-carlo integration loop, using numpy's mask feature to skip
        iterations where the improvement is not positive.
        """
        noise=self._noise

        num_points = self.num_to_sample + self.num_being_sampled
        union_of_points = numpy.reshape(numpy.append(self._points_to_sample, self._points_being_sampled), (num_points, self.dim))
        num_combination = num_points+self.discrete
        combination = numpy.reshape(numpy.append(union_of_points, self._discrete_pts), (num_combination, self.dim))

        #mu_star = self._gaussian_process.compute_mean_of_points(self._discrete_pts)
        var_star = self._gaussian_process.compute_covariance_of_points(combination, union_of_points)
        K=var_star[num_points:num_combination, :num_points]
        try:
            chol_var = -scipy.linalg.cholesky(var_star[:num_points, :num_points]+numpy.diag(noise*numpy.ones(num_points)), lower=True, overwrite_a=True)
        except scipy.linalg.LinAlgError as exception:
            self.log.info('GP-variance matrix (size {0:d} is singular; scipy.linalg.cholesky failed. Error: {1:s}'.format(num_points, exception))
            # TOOD(GH-325): Investigate whether the SVD is the best option here
            # var_star is singular or near-singular and cholesky failed.
            # Instead, use the SVD: U * E * V^H = A, which can be computed extremely reliably.
            # See: http://en.wikipedia.org/wiki/Singular_value_decomposition
            # U, V are unitary and E is diagonal with all non-negative entries.
            # If A is SPSD, U = V.
            _, E, VH = scipy.linalg.svd(var_star[:num_points, :num_points]+numpy.diag(noise*numpy.ones(num_points)))
            # Then form factor Q * R = sqrt(E) * V^H.
            # See: http://en.wikipedia.org/wiki/QR_decomposition
            # (Q * R)^T * (Q * R) = R^T * Q * Q^T * R = R^T * R
            # and (Q * R)^T * (Q * R) = (sqrt(E) * V^T)^T * (sqrt(E) * V^T)
            # = V * sqrt(E) * sqrt(E) * V^T = A (using U = V).
            # Hence R^T * R = L * L^T = A is a cholesky factorization.
            # Note: we do not always use this approach b/c it is extremely expensive.
            R = scipy.linalg.qr(numpy.dot(numpy.diag(numpy.sqrt(E)), VH), mode='r')[0]
            chol_var = -R.T
        #sigma = numpy.mat(K) * (((numpy.mat(chol_var)).T).I)
        sigma = (scipy.linalg.solve_triangular(
                chol_var,
                K.T,
                lower=True,
                overwrite_b=True,
        )).T

        normals = numpy.random.normal(size=(self._num_mc_iterations, num_points))
        # TODO(GH-60): Partition num_mc_iterations up into smaller blocks if it helps.
        # so that we don't waste as much mem bandwidth (since each entry of normals is
        # only used once)
        mu_star = self._best_so_far - self._mu_star
        # Compute Ls * w; note the shape is (self._num_mc_iterations, num_points)
        improvement_each_iter = numpy.einsum('kj, ij', sigma, normals)
        # Now we have improvement = best_so_far - y = best_so_far - (mus + Ls * w)
        improvement_each_iter += mu_star
        # We want the maximum improvement each step; note the shape is (self._num_mc_iterations)
        best_improvement_each_iter = numpy.amax(improvement_each_iter, axis=1)
        result = best_improvement_each_iter.sum(dtype=numpy.float64) / float(self._num_mc_iterations)
        return result

    
    compute_objective_function = _compute_knowledge_gradient_monte_carlo

    def _compute_grad_knowledge_gradient_test(self, force_monte_carlo=False):
        r"""Need to test the grad of KG
        :param force_monte_carlo:
        :return: finite difference estimation
        """
        aggregate_dx = numpy.zeros_like(self._points_to_sample)
        #data=numpy.copy(self._points_to_sample)
        fx = self.compute_objective_function()
        curPoint=numpy.copy(self._points_to_sample)
        for i in xrange(self.num_to_sample):
            for j in xrange(self.dim):
                temp=curPoint
                temp[i,j]+=0.01
                self.set_current_point(temp)
                fy = self.compute_objective_function()
                aggregate_dx[i,j]=(fy-fx)/0.01
        return aggregate_dx

 
    def _compute_grad_knowledge_gradient_naive(self, force_monte_carlo=True):
        r"""Compute the gradient of KG using (naive) monte carlo integration.
        :return: gradient of KG, ``\pderiv{KG(Xq \cup Xp)}{Xq_{i,d}}`` where ``Xq`` is ``points_to_sample``
          and ``Xp`` is ``points_being_sampled`` (grad KG from sampling ``points_to_sample`` with
          ``points_being_sampled`` concurrent experiments wrt each dimension of the points in ``points_to_sample``)
        :rtype: array of float64 with shape (self.num_to_sample, self.dim)
        """
        noise=self._noise

        num_points = self.num_to_sample + self.num_being_sampled
        union_of_points = numpy.reshape(numpy.append(self._points_to_sample, self._points_being_sampled), (num_points, self.dim))
        num_combination = num_points+self.discrete
        combination = numpy.reshape(numpy.append(union_of_points, self._discrete_pts), (num_combination, self.dim))

        var_star = self._gaussian_process.compute_covariance_of_points(combination, union_of_points)
        #K is the covariance between discrete_pts and the points_to_sample
        K=var_star[num_points:num_combination, :num_points]
        #chol_var is the D(x_1, ..., x_q) in the document
        chol_var = scipy.linalg.cholesky(var_star[:num_points, :num_points]+numpy.diag(noise*numpy.ones(num_points)), lower=True, overwrite_a=True)

        #compute derivative of D in the document
        grad_chol_decomp = self._gaussian_process.compute_grad_cholesky_variance_of_points(
                union_of_points,
                chol_var=chol_var,
                num_derivatives=self.num_to_sample,
        )
        #compute derivative of (D^{-1})^T in the document
        for i in xrange(self.num_to_sample):
            for j in xrange(self.dim):
                tmp = scipy.linalg.solve_triangular(
                        chol_var,
                        grad_chol_decomp[i, ..., j],
                        lower = True,
                        overwrite_b = True,
                )
                grad_chol_decomp[i, ..., j] = -scipy.linalg.solve_triangular(
                        chol_var.T,
                        tmp.T,
                        lower = False,
                        overwrite_b = True,
                )
                #grad_chol_decomp[i, ..., j]=-((numpy.mat(chol_var).I)*numpy.mat(grad_chol_decomp[i, ..., j])*(numpy.mat(chol_var).I)).T

        #sigma = numpy.mat(K) * (((numpy.mat(chol_var)).T).I)
        sigma = (scipy.linalg.solve_triangular(
                chol_var,
                K.T,
                lower=True,
                overwrite_b=True,
        )).T

        normals = numpy.random.normal(size=(self._num_mc_iterations, num_points))

        # Differentiating wrt each point of self._points_to_sample
        aggregate_dx = numpy.zeros_like(self._points_to_sample)
        for normal_draws in normals:
            sample_values_this_iter = self._mu_star + numpy.dot(sigma, normal_draws.T)
            improvement_this_iter = self._best_so_far-sample_values_this_iter
            winner = numpy.argmax(improvement_this_iter)

            #tmp_grad_var is the covariace among the best point in discrete_pts and points_to_sample wrt points_to_sample
            tmp_grad_var = self._gaussian_process.compute_grad_covariance_of_points(union_of_points, self._discrete_pts[winner,], num_derivatives=self.num_to_sample)
            for diff_index in xrange(self.num_to_sample):
                tmp_winner = numpy.einsum('i, ijk -> jk',K[winner, ...], grad_chol_decomp[diff_index,...])
                #print "test 1st component", tmp_winner, numpy.mat(K[winner, ...])*numpy.mat(grad_chol_decomp[diff_index,...,0])
                tmp_winner += scipy.linalg.solve_triangular(
                        chol_var,
                        tmp_grad_var[diff_index, ...,0,...],
                        lower=True,
                        overwrite_b=True,
                )
                '''
                print "test 2nd component", scipy.linalg.solve_triangular(
                        chol_var,
                        tmp_grad_var[diff_index, num_points, :num_points, ...],
                        lower=True,
                        overwrite_b=True,
                ), numpy.mat(tmp_grad_var[diff_index, num_points, :num_points, 0])*(numpy.mat(chol_var).I).T
                '''
                aggregate_dx[diff_index, ...] -= numpy.dot(tmp_winner.T, normal_draws)

        return aggregate_dx / float(self._num_mc_iterations)

    def _compute_grad_knowledge_gradient_monte_carlo(self, force_monte_carlo=True):
        noise=self._noise

        num_points = self.num_to_sample + self.num_being_sampled
        union_of_points = numpy.reshape(numpy.append(self._points_to_sample, self._points_being_sampled), (num_points, self.dim))

        num_combination = num_points+self.discrete
        combination = numpy.reshape(numpy.append(union_of_points, self._discrete_pts), (num_combination, self.dim))

        var_star = self._gaussian_process.compute_covariance_of_points(combination, union_of_points)
        K=var_star[num_points:num_combination, :num_points]
        
        try:
            chol_var = -scipy.linalg.cholesky(var_star[:num_points, :num_points]+numpy.diag(noise*numpy.ones(num_points)), lower=True, overwrite_a=True)
        except scipy.linalg.LinAlgError as exception:
            self.log.info('GP-variance matrix (size {0:d} is singular; scipy.linalg.cholesky failed. Error: {1:s}'.format(num_points, exception))
            # TOOD(GH-325): Investigate whether the SVD is the best option here
            # var_star is singular or near-singular and cholesky failed.
            # Instead, use the SVD: U * E * V^H = A, which can be computed extremely reliably.
            # See: http://en.wikipedia.org/wiki/Singular_value_decomposition
            # U, V are unitary and E is diagonal with all non-negative entries.
            # If A is SPSD, U = V.
            _, E, VH = scipy.linalg.svd(var_star[:num_points, :num_points]+numpy.diag(noise*numpy.ones(num_points)))
            # Then form factor Q * R = sqrt(E) * V^H.
            # See: http://en.wikipedia.org/wiki/QR_decomposition
            # (Q * R)^T * (Q * R) = R^T * Q * Q^T * R = R^T * R
            # and (Q * R)^T * (Q * R) = (sqrt(E) * V^T)^T * (sqrt(E) * V^T)
            # = V * sqrt(E) * sqrt(E) * V^T = A (using U = V).
            # Hence R^T * R = L * L^T = A is a cholesky factorization.
            # Note: we do not always use this approach b/c it is extremely expensive.
            R = scipy.linalg.qr(numpy.dot(numpy.diag(numpy.sqrt(E)), VH), mode='r')[0]
            chol_var = -R.T

        grad_chol_decomp = self._gaussian_process.compute_grad_cholesky_variance_of_points(
                union_of_points,
                chol_var=chol_var,
                num_derivatives=self.num_to_sample,
        )

        for i in xrange(self.num_to_sample):
            for j in xrange(self.dim):
                tmp = scipy.linalg.solve_triangular(
                        chol_var,
                        grad_chol_decomp[i, ..., j],
                        lower = True,
                        overwrite_b = True,
                )
                grad_chol_decomp[i, ..., j] = -scipy.linalg.solve_triangular(
                        chol_var.T,
                        tmp.T,
                        lower = False,
                        overwrite_b = True,
                )
                #grad_chol_decomp[i, ..., j]=-((numpy.mat(chol_var).I)*numpy.mat(grad_chol_decomp[i, ..., j])*(numpy.mat(chol_var).I)).T

        #sigma = numpy.mat(K) * (((numpy.mat(chol_var)).T).I)
        sigma = (scipy.linalg.solve_triangular(
                chol_var,
                K.T,
                lower=True,
                overwrite_b=True,
        )).T

        normals = numpy.random.normal(size=(self._num_mc_iterations, num_points))
        # TODO(GH-60): Partition num_mc_iterations up into smaller blocks if it helps.
        # so that we don't waste as much mem bandwidth (since each entry of normals is
        # only used once)
        mu_star = self._best_so_far - self._mu_star
        # Compute Ls * w; note the shape is (self._num_mc_iterations, num_points)
        improvement_each_iter = numpy.einsum('kj, ij', sigma, normals)
        # Now we have improvement = best_so_far - y = best_so_far - (mus + Ls * w)
        improvement_each_iter += mu_star
        # We want the maximum improvement each step; note the shape is (self._num_mc_iterations)
        best_improvement_each_iter = numpy.amax(improvement_each_iter, axis=1)
        winner_indexes = numpy.argmax(improvement_each_iter, axis=1)

        aggregate_dx = numpy.zeros_like(self._points_to_sample)
        #dimension of grad_chol_decomp_tiled: _num_mc_iterations * num_to_sample * dim
        grad_chol_decomp_tiled = numpy.empty((normals.shape[0], grad_chol_decomp.shape[0], grad_chol_decomp.shape[3]))

        grad_var = self._gaussian_process.compute_grad_covariance_of_points(union_of_points, self._discrete_pts[winner_indexes,], self.num_to_sample)

        for diff_index in xrange(self.num_to_sample):
            grad_chol_decomp_tiled[...] = 0.0
            for k in xrange(self._num_mc_iterations):
                grad_chol_decomp_tiled[k, ...] = numpy.einsum('i, ijk -> jk', K[winner_indexes[k], ...], grad_chol_decomp[diff_index,...])
                #tmp_winner += (tmp_grad_var[diff_index, num_points, :num_points, ...].T * (numpy.mat(chol_var).I).T).T
                grad_chol_decomp_tiled[k, ...] += scipy.linalg.solve_triangular(
                        chol_var,
                        grad_var[diff_index,...,k,...],
                        lower=True,
                        overwrite_b=True,
                )
            aggregate_dx[diff_index, ...] -= numpy.einsum('ki, kij', normals, grad_chol_decomp_tiled)
        aggregate_dx /= float(self._num_mc_iterations)
        return aggregate_dx

    compute_grad_objective_function = _compute_grad_knowledge_gradient_monte_carlo

    def compute_hessian_objective_function(self, **kwargs):
        """We do not currently support computation of the (spatial) hessian of Expected Improvement."""
        raise NotImplementedError('Currently we cannot compute the hessian of expected improvement.')