# -*- coding: utf-8 -*-
"""Implementation (Python) of GaussianProcessInterface.
This file contains a class to manipulate a Gaussian Process through numpy/scipy.
See :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface` for more details.
"""
import logging

import copy

import numpy

import tensorflow as tf

import emcee

import moe.build.GPP as C_GP
from moe.optimal_learning.python.cpp_wrappers import cpp_utils
from moe.optimal_learning.python.cpp_wrappers.covariance import SquareExponential
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess
from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient_mcmc import GaussianProcessMCMC

class DeepAdditiveKernelMCMC(object):

    r"""Class for computing log likelihood-like measures of deep kernel model fit.
    """

    def __init__(self, historical_data, derivatives, prior, chain_length, burnin_steps, n_hypers,
                 log_likelihood_type=C_GP.LogLikelihoodTypes.log_marginal_likelihood, noisy = True, rng = None):
        """Construct a LogLikelihood object that knows how to call C++ for evaluation of member functions.

        :param covariance_function: covariance object encoding assumptions about the GP's behavior on our data
        :type covariance_function: :class:`moe.optimal_learning.python.interfaces.covariance_interface.CovarianceInterface` subclass
          (e.g., from :mod:`moe.optimal_learning.python.cpp_wrappers.covariance`).
        :param historical_data: object specifying the already-sampled points, the objective value at those points, and the noise variance associated with each observation
        :type historical_data: :class:`moe.optimal_learning.python.data_containers.HistoricalData` object
        :param log_likelihood_type: enum specifying which log likelihood measure to compute
        :type log_likelihood_type: GPP.LogLikelihoodTypes

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
        self.noisy = noisy

        if rng is None:
            self.rng = numpy.random.RandomState(numpy.random.randint(0, 10000))
        else:
            self.rng = rng
        self.n_hypers = n_hypers
        self.n_chains = max(n_hypers, 2*(self._historical_data.dim+1+1+self._num_derivatives))
        self._nn_hypers = None

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

    def train_MLP(self):
        """
        Trains the model on the historical data.
        """
        with tf.Graph().as_default():
            # Parameters
            learning_rate = 0.001
            training_epochs = 15
            batch_size = 100
            display_step = 1

            qw_0 = tf.Variable(tf.random_normal([10, self.dim]))
            qw_1 = tf.Variable(tf.random_normal([10, 10]))
            qw_2 = tf.Variable(tf.random_normal([10, 10]))
            qw_3 = tf.Variable(tf.random_normal([1, 10]))
            qb_0 = tf.Variable(tf.random_normal([10]))
            qb_1 = tf.Variable(tf.random_normal([10]))
            qb_2 = tf.Variable(tf.random_normal([10]))
            qb_3 = tf.Variable(tf.random_normal([1]))

            N = self._points_sampled.shape[0]  # number of data points

            x = tf.placeholder(tf.float32, [None, self.dim])
            y = tf.placeholder(tf.float32, [None, 1])
            param = [tf.transpose(qw_0), tf.transpose(qw_1), tf.transpose(qw_2), tf.transpose(qw_3),
                     qb_0, qb_1, qb_2, qb_3]

            # Construct model
            pred = self.neural_network(x, param)
            # Define loss and optimizer
            cost = tf.reduce_mean(tf.square(tf.subtract(y, pred)))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

            # Initializing the variables
            init = tf.global_variables_initializer()

            # Launch the graph
            with tf.Session() as sess:
                sess.run(init)

                # Training cycle
                for epoch in xrange(training_epochs):
                    avg_cost = 0.
                    total_batch = int(N/batch_size)
                    # Loop over all batches
                    for i in xrange(total_batch):
                        batch_x, batch_y = self._points_sampled[i*batch_size:min((i+1)*batch_size, N)], self._points_sampled_value[i*batch_size:min((i+1)*batch_size, N)]
                        # Run optimization op (backprop) and cost op (to get loss value)
                        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                                      y: batch_y})
                        # Compute average loss
                        avg_cost += c / total_batch
                    # Display logs per epoch step
                    if epoch % display_step == 0:
                        print("Epoch:", '%04d' % (epoch+1), "cost=", \
                              "{:.9f}".format(avg_cost))
                print("Optimization Finished!")
                self._nn_hypers = numpy.concatenate([
                        qw_0.eval().ravel(), qb_0.eval().ravel(),
                        qw_1.eval().ravel(), qb_1.eval().ravel(),
                        qw_2.eval().ravel(), qb_2.eval().ravel()])

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
            self.train_MLP()
            # We have one walker for each hyperparameter configuration
            sampler = emcee.EnsembleSampler(self.n_chains, 2*self.dim + self._num_derivatives + 1,
                                            self.compute_log_likelihood)

            # Do a burn-in in the first iteration
            if not self.burned:
                # Initialize the walkers by sampling from the prior
                if self.prior is None:
                    self.p0 = numpy.random.rand(self.n_chains, 2*self.dim + self._num_derivatives + 1)
                else:
                    self.p0 = self.prior.sample_from_prior(self.n_chains)
                # Run MCMC sampling
                self.p0, _, _ = sampler.run_mcmc(self.p0, self.burnin_steps,
                                                 rstate0=self.rng)

                self.burned = True

            # Start sampling
            pos, _, _ = sampler.run_mcmc(self.p0, self.chain_length,
                                         rstate0=self.rng)

            # Save the current position, it will be the start point in
            # the next iteration
            self.p0 = pos

            # Take the last samples from each walker
            self.hypers = sampler.chain[numpy.random.choice(self.n_chains, self.n_hypers), -1]

        self.is_trained = True
        self._models = []
        hypers_list = []
        noises_list = []
        for sample in self.hypers:
            if numpy.any((-20 > sample) + (sample > 20)):
                continue
            sample = numpy.exp(sample)
            print sample
            # Instantiate a GP for each hyperparameter configuration
            hypers_list.append(self._nn_hypers)
            cov_hyps = sample[:20]
            hypers_list.append(cov_hyps)
            se = SquareExponential(cov_hyps)
            if self.noisy:
                noise = sample[20:]
            else:
                noise = numpy.array((1+self._num_derivatives)*[1.e-8])
            noises_list.append(noise)
            model = GaussianProcess(se, noise,
                                    self._historical_data,
                                    self.derivatives)
            self._models.append(model)

        self._gaussian_process_mcmc = GaussianProcessMCMC(numpy.array(hypers_list), numpy.array(noises_list),
                                                          self._historical_data, self.derivatives)

    def compute_log_likelihood(self, hyps):
        r"""Compute the objective_type measure at the specified hyperparameters.

        :return: value of log_likelihood evaluated at hyperparameters (``LL(y | X, \theta)``)
        :rtype: float64

        """
        # Bound the hyperparameter space to keep things sane. Note all
        # hyperparameters live on a log scale
        if numpy.any((-20 > hyps) + (hyps > 20)):
            return -numpy.inf

        hyps = numpy.exp(hyps)
        cov_hyps = hyps[:20]
        noise = hyps[20:]
        if not self.noisy:
            noise = numpy.array((1+self._num_derivatives)*[1.e-8])

        try:
            if self.prior is not None:
                posterior = self.prior.lnprob(numpy.log(hyps))
                return posterior + C_GP.compute_log_likelihood(
                        cpp_utils.cppify(self._points_sampled),
                        cpp_utils.cppify(self._points_sampled_value),
                        self.dim,
                        self._num_sampled,
                        self.objective_type,
                        cpp_utils.cppify_hyperparameters(self._nn_hypers),
                        cpp_utils.cppify_hyperparameters(cov_hyps),
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
                        cpp_utils.cppify_hyperparameters(self._nn_hypers),
                        cpp_utils.cppify_hyperparameters(cov_hyps),
                        cpp_utils.cppify(self._derivatives), self._num_derivatives,
                        cpp_utils.cppify(noise),
                )
        except:
            return -numpy.inf

    def neural_network(self, X, param):
        """define the neural network part
        Parameters
        ----------
        X: numpy.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        """
        #W_0, b_0, W_1, b_1, W_2, b_2, W_3, b_3, noise, sigma, l = param
        W_0, W_1, W_2, W_3, b_0, b_1, b_2, b_3 = param
        h = tf.tanh(tf.matmul(X, W_0) + b_0)
        h = tf.tanh(tf.matmul(h, W_1) + b_1)
        h = tf.tanh(tf.matmul(h, W_2) + b_2)
        h = tf.tanh(tf.matmul(h, W_3) + b_3)
        return h

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
