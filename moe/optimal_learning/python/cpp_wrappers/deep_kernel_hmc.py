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

from moe.optimal_learning.python.cpp_wrappers.covariance import SquareExponential
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess
from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient_mcmc import GaussianProcessMCMC

class DeepKernelHMC(object):

    r"""Class for computing log likelihood-like measures of deep kernel model fit.
    """

    def __init__(self, historical_data, chain_length = 10**4, burnin_steps = 2*10**3, stride = 100):
        """Construct a LogLikelihood object that knows how to call C++ for evaluation of member functions.
        :param covariance_function: covariance object encoding assumptions about the GP's behavior on our data
        :type covariance_function: :class:`moe.optimal_learning.python.interfaces.covariance_interface.CovarianceInterface` subclass
          (e.g., from :mod:`moe.optimal_learning.python.cpp_wrappers.covariance`).
        :param historical_data: object specifying the already-sampled points, the objective value at those points, and the noise variance associated with each observation
        :type historical_data: :class:`moe.optimal_learning.python.data_containers.HistoricalData` object
        """
        self._historical_data = copy.deepcopy(historical_data)
        self.chain_length = chain_length
        self.burned = False
        self.burnin_steps = burnin_steps
        self.stride_length = stride
        self.n_nets = 80

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

    def train(self, **kwargs):
        """
        Trains the model on the historical data.
        """
        with tf.Graph().as_default():
            W_0 = Normal(tf.zeros([10, self.dim]), tf.ones([10, self.dim]))
            W_1 = Normal(tf.zeros([10, 10]), tf.ones([10, 10]))
            W_2 = Normal(tf.zeros([10, 10]), tf.ones([10, 10]))
            W_3 = Normal(tf.zeros([1, 10]), tf.ones([1, 10]))
            b_0 = Normal(tf.zeros(10), tf.ones(10))
            b_1 = Normal(tf.zeros(10), tf.ones(10))
            b_2 = Normal(tf.zeros(10), tf.ones(10))
            b_3 = Normal(tf.zeros(1), tf.ones(1))

            noise = Normal(tf.zeros(1), tf.ones(1))
            sigma = Normal(tf.zeros(1), tf.ones(1))
            l = Normal(tf.zeros(1), tf.ones(1))


            param = [tf.transpose(W_0), b_0,
                     tf.transpose(W_1), b_1,
                     tf.transpose(W_2), b_2,
                     tf.transpose(W_3), b_3,
                     tf.nn.softplus(noise), tf.nn.softplus(sigma),
                     self.logistic(l, 0.04, 10)]

            N = self._points_sampled.shape[0]  # number of data points

            X_input = tf.placeholder(tf.float32, [N, self.dim])
            f = MultivariateNormalTriL(loc=tf.constant([numpy.mean(self._points_sampled_value)]*N),
                                       scale_tril=tf.cholesky(self.covariance(param, X_input)))

            qw_0 = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 10, self.dim])))
            qw_1 = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 10, 10])))
            qw_2 = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 10, 10])))
            qw_3 = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 1, 10])))
            qb_0 = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 10])))
            qb_1 = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 10])))
            qb_2 = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 10])))
            qb_3 = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 1])))

            qnoise = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 1])))
            qsigma = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 1])))
            ql = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 1])))

            # from (N, 1) to (N, )
            logging.info("Starting sampling")
            inference = ed.HMC({W_0: qw_0,
                                W_1: qw_1,
                                W_2: qw_2,
                                W_3: qw_3,
                                b_0: qb_0,
                                b_1: qb_1,
                                b_2: qb_2,
                                b_3: qb_3,
                                noise: qnoise,
                                sigma: qsigma,
                                l: ql}, data={X_input: self._points_sampled,
                                              f: self._points_sampled_value.ravel()})
            # inference = ed.HMC({W_0: qw_0,
            #                     b_0: qb_0,
            #                     noise: qnoise,
            #                     sigma: qsigma,
            #                     l: ql}, data={X_input: self._points_sampled, f: self._points_sampled_value.ravel()})

            inference.run(step_size = 4*1e-5, n_steps = 10)

            qw_0 = qw_0.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
            qw_1 = qw_1.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
            qw_2 = qw_2.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
            qw_3 = qw_3.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
            qb_0 = qb_0.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
            qb_1 = qb_1.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
            qb_2 = qb_2.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
            qb_3 = qb_3.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
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
                param = numpy.concatenate([
                         qw_0[n_sample].ravel(), qb_0[n_sample].ravel(),
                         qw_1[n_sample].ravel(), qb_1[n_sample].ravel(),
                         qw_2[n_sample].ravel(), qb_2[n_sample].ravel(),
                         qw_3[n_sample].ravel(), qb_3[n_sample].ravel(),
                         qsigma[n_sample].ravel(), ql[n_sample].ravel()]).astype(numpy.float64)
                hypers_list.append(param)
                cov = SquareExponential(param)
                model = GaussianProcess(cov, qnoise[n_sample].astype(numpy.float64), self._historical_data, [])
                noises_list.append(1e-6)
                self._models.append(model)
            self._gaussian_process_mcmc = GaussianProcessMCMC(numpy.array(hypers_list), numpy.array(noises_list),
                                                              self._historical_data, [])
            self.is_trained = True

    def train_KL(self, **kwargs):
        """
        Trains the model on the historical data.
        """
        with tf.Graph().as_default():
            W_0 = Normal(tf.zeros([10, self.dim]), tf.ones([10, self.dim]))
            W_1 = Normal(tf.zeros([10, 10]), tf.ones([10, 10]))
            W_2 = Normal(tf.zeros([10, 10]), tf.ones([10, 10]))
            W_3 = Normal(tf.zeros([1, 10]), tf.ones([1, 10]))
            b_0 = Normal(tf.zeros(10), tf.ones(10))
            b_1 = Normal(tf.zeros(10), tf.ones(10))
            b_2 = Normal(tf.zeros(10), tf.ones(10))
            b_3 = Normal(tf.zeros(1), tf.ones(1))

            noise = Normal(tf.zeros(1), tf.ones(1))
            sigma = Normal(tf.zeros(1), tf.ones(1))
            l = Normal(tf.zeros(1), tf.ones(1))


            param = [tf.transpose(W_0), b_0,
                     tf.transpose(W_1), b_1,
                     tf.transpose(W_2), b_2,
                     tf.transpose(W_3), b_3,
                     tf.nn.softplus(noise), tf.nn.softplus(sigma),
                     self.logistic(l, 0.04, 10)]

            N = self._points_sampled.shape[0]  # number of data points

            X_input = tf.placeholder(tf.float32, [N, self.dim])
            f = MultivariateNormalTriL(loc=tf.constant([numpy.mean(self._points_sampled_value)]*N),
                                       scale_tril=tf.cholesky(self.covariance(param, X_input)))

            qw_0 = Normal(loc=tf.Variable(tf.random_normal([10, self.dim])),
                          scale=tf.nn.softplus(tf.Variable(tf.random_normal([10, self.dim]))))
            qw_1 = Normal(loc=tf.Variable(tf.random_normal([10, 10])),
                          scale=tf.nn.softplus(tf.Variable(tf.random_normal([10, 10]))))
            qw_2 = Normal(loc=tf.Variable(tf.random_normal([10, 10])),
                          scale=tf.nn.softplus(tf.Variable(tf.random_normal([10, 10]))))
            qw_3 = Normal(loc=tf.Variable(tf.random_normal([1, 10])),
                          scale=tf.nn.softplus(tf.Variable(tf.random_normal([1, 10]))))
            qb_0 = Normal(loc=tf.Variable(tf.random_normal([10])),
                          scale=tf.nn.softplus(tf.Variable(tf.random_normal([10]))))
            qb_1 = Normal(loc=tf.Variable(tf.random_normal([10])),
                          scale=tf.nn.softplus(tf.Variable(tf.random_normal([10]))))
            qb_2 = Normal(loc=tf.Variable(tf.random_normal([10])),
                          scale=tf.nn.softplus(tf.Variable(tf.random_normal([10]))))
            qb_3 = Normal(loc=tf.Variable(tf.random_normal([1])),
                          scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

            qnoise = Normal(loc=tf.Variable(tf.random_normal([1])),
                            scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))
            qsigma = Normal(loc=tf.Variable(tf.random_normal([1])),
                            scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))
            ql = Normal(loc=tf.Variable(tf.random_normal([1])),
                        scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

            # from (N, 1) to (N, )
            logging.info("Starting sampling")
            inference = ed.KLqp({W_0: qw_0,
                                 W_1: qw_1,
                                 W_2: qw_2,
                                 W_3: qw_3,
                                 b_0: qb_0,
                                 b_1: qb_1,
                                 b_2: qb_2,
                                 b_3: qb_3,
                                 noise: qnoise,
                                 sigma: qsigma,
                                 l: ql}, data={X_input: self._points_sampled,
                                               f: self._points_sampled_value.ravel()})

            inference.run(n_iter=int(2*1e3))

            qw_0 = qw_0.sample(self.n_nets).eval()
            qw_1 = qw_1.sample(self.n_nets).eval()
            qw_2 = qw_2.sample(self.n_nets).eval()
            qw_3 = qw_3.sample(self.n_nets).eval()
            qb_0 = qb_0.sample(self.n_nets).eval()
            qb_1 = qb_1.sample(self.n_nets).eval()
            qb_2 = qb_2.sample(self.n_nets).eval()
            qb_3 = qb_3.sample(self.n_nets).eval()
            qnoise = tf.nn.softplus(qnoise.sample(self.n_nets)).eval()
            qsigma = tf.nn.softplus(qsigma.sample(self.n_nets)).eval()
            ql = self.logistic(ql.sample(self.n_nets), 0.04, 10).eval()
            sess = ed.get_session()
            if sess is not None:
                sess.close()
                del sess
            self.n_hypers = ql.shape[0]
            self._models = []
            hypers_list = []
            noises_list = []
            for n_sample in xrange(self.n_hypers):
                param = numpy.concatenate([
                    qw_0[n_sample].ravel(), qb_0[n_sample].ravel(),
                    qw_1[n_sample].ravel(), qb_1[n_sample].ravel(),
                    qw_2[n_sample].ravel(), qb_2[n_sample].ravel(),
                    qw_3[n_sample].ravel(), qb_3[n_sample].ravel(),
                    qsigma[n_sample].ravel(), ql[n_sample].ravel()]).astype(numpy.float64)
                hypers_list.append(param)
                cov = SquareExponential(param)
                model = GaussianProcess(cov, qnoise[n_sample].astype(numpy.float64), self._historical_data, [])
                noises_list.append(1e-6)
                self._models.append(model)
            self._gaussian_process_mcmc = GaussianProcessMCMC(numpy.array(hypers_list), numpy.array(noises_list),
                                                              self._historical_data, [])
            self.is_trained = True

    def neural_network(self, X, param):
        """define the neural network part
        Parameters
        ----------
        X: numpy.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        """
        W_0, b_0, W_1, b_1, W_2, b_2, W_3, b_3, noise, sigma, l = param
        #>W_0, b_0, noise, sigma, l = param
        h = tf.tanh(tf.matmul(X, W_0) + b_0)
        h = tf.tanh(tf.matmul(h, W_1) + b_1)
        h = tf.tanh(tf.matmul(h, W_2) + b_2)
        h = tf.tanh(tf.matmul(h, W_3) + b_3)
        return h

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
        W_0, b_0, W_1, b_1, W_2, b_2, W_3, b_3, noise, sigma, l = param
        #W_0, b_0, noise, sigma, l = param
        points_one = self.neural_network(points_one, param)
        points_one = tf.divide(points_one, l)
        set_sum1 = tf.reduce_sum(tf.square(points_one), 1)
        if points_two is None:
            return -2 * tf.matmul(points_one, tf.transpose(points_one)) + \
                   tf.reshape(set_sum1, (-1, 1)) + tf.reshape(set_sum1, (1, -1))
        else:
            points_two = self.neural_network(points_two, param)
            points_two = tf.divide(points_two, self._lengths_scale)
            set_sum2 = tf.reduce_sum(tf.square(points_two), 1)
            return -2 * tf.matmul(points_one, tf.transpose(points_two)) + \
                   tf.reshape(set_sum1, (-1, 1)) + tf.reshape(set_sum2, (1, -1))

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
        W_0, b_0, W_1, b_1, W_2, b_2, W_3, b_3, noise, sigma, l = param
        #W_0, b_0, noise, sigma, l = param
        return 1e-1*tf.constant(numpy.identity(points_one.shape[0]), dtype=tf.float32) + sigma * tf.exp(-0.5 * self.square_dist(param, points_one, points_two))

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