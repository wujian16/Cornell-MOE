# -*- coding: utf-8 -*-
"""Implementation (Python) of GaussianProcessInterface.
This file contains a class to manipulate a Gaussian Process through numpy/scipy.
See :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface` for more details.
"""
import logging

import copy

import numpy

import tensorflow as tf

import edward as ed
from edward.models import MultivariateNormalTriL, Normal, Empirical

import emcee

import moe.build.GPP as C_GP
from moe.optimal_learning.python.cpp_wrappers import cpp_utils
from moe.optimal_learning.python.cpp_wrappers.covariance import SquareExponential
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess
from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient_mcmc import GaussianProcessMCMC

class AdditiveKernelMCMC(object):

    r"""Class for computing log likelihood-like measures of deep kernel model fit.
    """

    def __init__(self, historical_data, derivatives, prior=None, chain_length=1e4, burnin_steps=2e3, n_hypers=20,
                 log_likelihood_type=C_GP.LogLikelihoodTypes.log_marginal_likelihood, stride = 100, rng = None):
        """Construct a LogLikelihood object that knows how to call C++ for evaluation of member functions.
        :param covariance_function: covariance object encoding assumptions about the GP's behavior on our data
        :type covariance_function: :class:`moe.optimal_learning.python.interfaces.covariance_interface.CovarianceInterface` subclass
          (e.g., from :mod:`moe.optimal_learning.python.cpp_wrappers.covariance`).
        :param historical_data: object specifying the already-sampled points, the objective value at those points, and the noise variance associated with each observation
        :type historical_data: :class:`moe.optimal_learning.python.data_containers.HistoricalData` object
        """
        self._historical_data = copy.deepcopy(historical_data)

        self._derivatives = copy.deepcopy(derivatives)
        self._num_derivatives = len(cpp_utils.cppify(self._derivatives))

        self.objective_type = log_likelihood_type

        self.prior = prior
        self.chain_length = chain_length
        self.burned = False
        self.burnin_steps = burnin_steps
        self._models = []

        if rng is None:
            self.rng = numpy.random.RandomState(numpy.random.randint(0, 10000))
        else:
            self.rng = rng
        self.stride_length = stride
        self.n_hypers = max(n_hypers, 2*(2*self._historical_data.dim+1+len(self._derivatives)))

    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return self._historical_data.dim

    @property
    def _num_sampled(self):
        """Return the number of sampled points."""
        return self._historical_data.num_sampled

    @property
    def _points_sampled(self):
        """Return the coordinates of the already-sampled points; see :class:`moe.optimal_learning.python.data_containers.HistoricalData`."""
        return self._historical_data.points_sampled

    @property
    def _points_sampled_value(self):
        """Return the function values measured at each of points_sampled; see :class:`moe.optimal_learning.python.data_containers.HistoricalData`."""
        return self._historical_data.points_sampled_value

    @property
    def _points_sampled_noise_variance(self):
        """Return the noise variance associated with points_sampled_value; see :class:`moe.optimal_learning.python.data_containers.HistoricalData`."""
        return self._historical_data.points_sampled_noise_variance

    @property
    def models(self):
        return self._models

    def get_historical_data_copy(self):
        """Return the data (points, function values, noise) specifying the prior of the Gaussian Process.
        :return: object specifying the already-sampled points, the objective value at those points, and the noise variance associated with each observation
        :rtype: data_containers.HistoricalData
        """
        return copy.deepcopy(self._historical_data)

    def logistic(self, x, a, b):
        ex = tf.exp(-x)
        return a + (b - a) / (1. + ex)

    def train_hmc(self, **kwargs):
        """
        Trains the model on the historical data.
        """
        with tf.Graph().as_default():
            noise = Normal(tf.zeros(1), tf.ones(1))
            sigma = Normal(tf.zeros(self.dim), tf.ones(self.dim))
            l = Normal(tf.zeros(self.dim), tf.ones(self.dim))


            param = [tf.nn.softplus(noise), tf.nn.softplus(sigma),
                     self.logistic(l, 0.04, 10)]

            N = self._points_sampled.shape[0]  # number of data points

            X_input = tf.placeholder(tf.float32, [N, self.dim])
            f = MultivariateNormalTriL(loc=tf.constant([numpy.mean(self._points_sampled_value)]*N),
                                       scale_tril=tf.cholesky(self.covariance(param, X_input)))

            qnoise = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 1])))
            qsigma = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, self.dim])))
            ql = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, self.dim])))

            # from (N, 1) to (N, )
            logging.info("Starting sampling")
            inference = ed.HMC({noise: qnoise,
                                sigma: qsigma,
                                l: ql}, data={X_input: self._points_sampled,
                                              f: self._points_sampled_value.ravel()})

            inference.run(step_size = 1e-2, n_steps = 10)

            qnoise = tf.nn.softplus(qnoise.params[self.burnin_steps:self.chain_length:self.stride_length]).eval()
            qsigma = tf.nn.softplus(qsigma.params[self.burnin_steps:self.chain_length:self.stride_length]).eval()
            ql = self.logistic(ql.params[self.burnin_steps:self.chain_length:self.stride_length], 0.04, 10).eval()
            sess = ed.get_session()
            if sess is not None:
                sess.close()
                del sess
            self.n_hypers = ql.shape[0]
            self._models = []
            hypers_list = []
            noises_list = []
            for n_sample in xrange(self.n_hypers):
                param = numpy.concatenate([qsigma[n_sample].ravel(), ql[n_sample].ravel()]).astype(numpy.float64)
                hypers_list.append(param)
                cov = SquareExponential(param)
                model = GaussianProcess(cov, qnoise[n_sample].astype(numpy.float64), self._historical_data, [])
                noises_list.append(1e-6)
                self._models.append(model)
            self._gaussian_process_mcmc = GaussianProcessMCMC(numpy.array(hypers_list), numpy.array(noises_list),
                                                              self._historical_data, [])
            self.is_trained = True

    def train(self, do_optimize=True, **kwargs):
        """
        Performs MCMC sampling to sample hyperparameter configurations from the
        likelihood and trains for each sample a GP on X and y

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.
        do_optimize: boolean
            If set to true we perform MCMC sampling otherwise we just use the
            hyperparameter specified in the kernel.
        """

        if do_optimize:
            # We have one walker for each hyperparameter configuration
            sampler = emcee.EnsembleSampler(self.n_hypers,
                                            2*self.dim + 2 + self._num_derivatives + 1,
                                            self.compute_log_likelihood)

            # Do a burn-in in the first iteration
            if not self.burned:
                # Initialize the walkers by sampling from the prior
                if self.prior is None:
                    self.p0 = numpy.random.rand(self.n_hypers, 2*self.dim + 2 + self._num_derivatives + 1)
                else:
                    self.p0 = self.prior.sample_from_prior(self.n_hypers)
                # Run MCMC sampling
                self.p0, _, _ = sampler.run_mcmc(self.p0,
                                                 self.burnin_steps,
                                                 rstate0=self.rng)

                self.burned = True

            # Start sampling
            pos, _, _ = sampler.run_mcmc(self.p0,
                                         self.chain_length,
                                         rstate0=self.rng)

            # Save the current position, it will be the start point in
            # the next iteration
            self.p0 = pos

            # Take the last samples from each walker
            self.hypers = sampler.chain[:, -1]

        self.is_trained = True
        self._models = []
        hypers_list = []
        noises_list = []
        for sample in self.hypers:
            if numpy.any((-10 > sample) + (sample > 10)):
                continue
            sample = numpy.exp(sample)
            # Instantiate a GP for each hyperparameter configuration
            cov_hyps = sample[:(2*self.dim + 2)]
            hypers_list.append(cov_hyps)
            se = SquareExponential(cov_hyps)
            noise = numpy.array([1e-6]*(1+len(self._derivatives)))
            noises_list.append(noise)
            model = GaussianProcess(se, noise,
                                    self._historical_data,
                                    self._derivatives)
            self._models.append(model)

        self._gaussian_process_mcmc = GaussianProcessMCMC(numpy.array(hypers_list), numpy.array(noises_list),
                                                          self._historical_data, self._derivatives)

    def compute_log_likelihood(self, hyps):
        r"""Compute the objective_type measure at the specified hyperparameters.

        :return: value of log_likelihood evaluated at hyperparameters (``LL(y | X, \theta)``)
        :rtype: float64

        """
        # Bound the hyperparameter space to keep things sane. Note all
        # hyperparameters live on a log scale
        if numpy.any((-10 > hyps) + (hyps > 10)):
            return -numpy.inf

        hyps = numpy.exp(hyps)
        cov_hyps = hyps[:(2*self.dim + 2)]
        noise = hyps[(2*self.dim + 2):]

        if self.prior is not None:
            posterior = self.prior.lnprob(numpy.log(hyps))
            return posterior + C_GP.compute_log_likelihood(
                    cpp_utils.cppify(self._points_sampled),
                    cpp_utils.cppify(self._points_sampled_value),
                    self.dim,
                    self._num_sampled,
                    self.objective_type,
                    cpp_utils.cppify(cov_hyps),
                    cpp_utils.cppify(self._derivatives), self._num_derivatives,
                    cpp_utils.cppify(noise),
            )
        else:
            return C_GP.compute_log_likelihood(
                    cpp_utils.cppify(self._points_sampled),
                    cpp_utils.cppify(self._points_sampled_value),
                    self.dim,
                    self._num_sampled,
                    self.objective_type,
                    cpp_utils.cppify(cov_hyps),
                    cpp_utils.cppify(self._derivatives), self._num_derivatives,
                    cpp_utils.cppify(noise),
            )

    def square_dist(self, param, points_one, points_two = None):
        r"""Compute the square distances of two sets of points, cov(``points_one``, ``points_two``).
        Square Exponential: ``dis(x_1, x_2) = ((x_1 - x_2)^T * L * (x_1 - x_2)) ``
        :param points_one: first input, the point ``x``
        :type point_one: array of float64 with shape (N1, dim)
        :param points_two: second input, the point ``y``
        :type point_two: array of float64 with shape (N2, dim)
        :return: the square distance matrix (tensor) between the input points
        :rtype: tensor of float64 with shape (N1, N2)
        """
        noise, sigma, l = param
        #W_0, b_0, noise, sigma, l = param
        points_one = tf.divide(points_one, l)

        results = []
        for i in xrange(points_one.shape[1]):
            set_sum1 = tf.reduce_sum(tf.square(points_one[:,i:(i+1)]), 1)
            temp = -2 * tf.matmul(points_one[:,i:(i+1)], tf.transpose(points_one[:,i:(i+1)])) + \
                   tf.reshape(set_sum1, (-1, 1)) + tf.reshape(set_sum1, (1, -1))
            results.append(temp)
        return results

    def covariance(self, param, points_one, points_two=None):
        r"""Compute the square exponential covariance function of two points, cov(``point_one``, ``point_two``).
        Square Exponential: ``cov(x_1, x_2) = \alpha * \exp(-1/2 * ((x_1 - x_2)^T * L * (x_1 - x_2)) )``
        .. Note:: comments are copied from the matching method comments of
          :class:`moe.optimal_learning.python.interfaces.covariance_interface.CovarianceInterface`.
        The covariance function is guaranteed to be symmetric by definition: ``covariance(x, y) = covariance(y, x)``.
        This function is also positive definite by definition.
        :param points_one: first input, the point ``x``
        :type point_one: array of float64 with shape (N1, dim)
        :param points_two: second input, the point ``y``
        :type point_two: array of float64 with shape (N2, dim)
        :return: the covariance tensor between the input points
        :rtype: tensor of float64 with shape (N1, N2)
        """
        noise, sigma, l = param
        dist = self.square_dist(param, points_one, points_two)
        #W_0, b_0, noise, sigma, l = param
        result = 1e-1*tf.constant(numpy.identity(points_one.shape[0]), dtype=tf.float32)
        for i in xrange(points_one.shape[1]):
            result += sigma[i] * tf.exp(-0.5 * dist[i])
        return result

    def add_sampled_points(self, sampled_points):
        r"""Add sampled point(s) (point, value, noise) to the GP's prior data.
        Also forces recomputation of all derived quantities for GP to remain consistent.
        :param sampled_points: :class:`moe.optimal_learning.python.SamplePoint` objects to load
          into the GP (containing point, function value, and noise variance)
        :type sampled_points: list of :class:`~moe.optimal_learning.python.SamplePoint` objects (or SamplePoint-like iterables)
        """
        # TODO(GH-159): When C++ can pass back numpy arrays, we can stop keeping a duplicate in self._historical_data.
        self._historical_data.append_sample_points(sampled_points)
        if len(self.models) > 0:
            for model in self._models:
                model.add_sampled_points(sampled_points)