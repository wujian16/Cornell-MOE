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
from edward.models import MultivariateNormalCholesky, Normal, Empirical

from moe.optimal_learning.python.python_version.deep_kernel import DeepKernel
from moe.optimal_learning.python.python_version.scalable_gaussian_process import ScalableDeepKernel

class DeepKernelHMC(object):

    r"""Class for computing log likelihood-like measures of deep kernel model fit.
    """

    def __init__(self, historical_data, chain_length = 10**4, burnin_steps = 5000, stride = 100):
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

    def train(self, **kwargs):
        """
        Trains the model on the historical data.
        """

        self.W_0 = Normal(mu=tf.zeros([self.dim, 50]), sigma=tf.ones([self.dim, 50]))
        self.W_1 = Normal(mu=tf.zeros([50, 50]), sigma=tf.ones([50, 50]))
        self.W_2 = Normal(mu=tf.zeros([50, 50]), sigma=tf.ones([50, 50]))
        self.W_3 = Normal(mu=tf.zeros([50, 1]), sigma=tf.ones([50, 1]))
        self.b_0 = Normal(mu=tf.zeros(50), sigma=tf.ones(50))
        self.b_1 = Normal(mu=tf.zeros(50), sigma=tf.ones(50))
        self.b_2 = Normal(mu=tf.zeros(50), sigma=tf.ones(50))
        self.b_3 = Normal(mu=tf.zeros(1), sigma=tf.ones(1))

        self.sigma = Normal(mu=tf.zeros(1), sigma=tf.ones(1))
        self.l = Normal(mu=tf.zeros(1), sigma=tf.ones(1))

        param = [self.W_0, self.W_1, self.W_2, self.W_3,
                 self.b_0, self.b_1, self.b_2, self.b_3,
                 tf.nn.softplus(self.sigma), tf.nn.softplus(self.l)]

        N = self._points_sampled.shape[0]  # number of data ponts
        dk = DeepKernel(param)

        X_input = tf.placeholder(tf.float32, [N, self.dim])
        f = MultivariateNormalCholesky(mu=tf.constant([numpy.mean(self._points_sampled_value)]*N), chol=tf.cholesky(dk.covariance(X_input)))

        qw_0 = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, self.dim, 50])))
        qw_1 = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 50, 50])))
        qw_2 = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 50, 50])))
        qw_3 = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 50, 1])))
        qb_0 = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 50])))
        qb_1 = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 50])))
        qb_2 = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 50])))
        qb_3 = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 1])))

        qsigma = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 1])))
        ql = Empirical(params=tf.Variable(tf.random_normal([self.chain_length, 1])))

        # from (N, 1) to (N, )
        logging.info("Starting sampling")
        inference = ed.HMC({self.W_0: qw_0,
                            self.W_1: qw_1,
                            self.W_2: qw_2,
                            self.W_3: qw_3,
                            self.b_0: qb_0,
                            self.b_1: qb_1,
                            self.b_2: qb_2,
                            self.b_3: qb_3,
                            self.sigma: qsigma,
                            self.l: ql}, data={X_input: self._points_sampled, f: self._points_sampled_value.ravel()})
        inference.run(step_size = 1e-3)
        qw_0 = qw_0.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
        qw_1 = qw_1.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
        qw_2 = qw_2.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
        qw_3 = qw_3.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
        qb_0 = qb_0.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
        qb_1 = qb_1.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
        qb_2 = qb_2.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
        qb_3 = qb_3.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
        qsigma = qsigma.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
        ql = ql.params[self.burnin_steps:self.chain_length:self.stride_length].eval()
        sess = tf.get_default_session()
        if sess is not None:
            sess.close()
        self.n_hypers = ql.shape[0]
        self._models = []
        for n_sample in xrange(self.n_hypers):
            param = [qw_0[n_sample], qw_1[n_sample], qw_2[n_sample], qw_3[n_sample],
                     qb_0[n_sample], qb_1[n_sample], qb_2[n_sample], qb_3[n_sample],
                     tf.nn.softplus(qsigma[n_sample]), tf.nn.softplus(ql[n_sample])]
            cov = DeepKernel(param)
            model = ScalableDeepKernel(cov, self._historical_data)
            self._models.append(model)
        self.is_trained = True

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