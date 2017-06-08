# -*- coding: utf-8 -*-
r"""Tools to compute log likelihood-like measures of model fit and optimize them (wrt the hyperparameters of covariance).

See the file comments in :mod:`moe.optimal_learning.python.interfaces.log_likelihood_interface`
for an overview of log likelihood-like metrics and their role
in model selection. This file provides hooks to implementations of two such metrics in C++: Log Marginal Likelihood and
Leave One Out Cross Validation Log Pseudo-Likelihood.

.. Note:: This is a copy of the file comments in gpp_model_selection.hpp.
  These comments are copied in :mod:`moe.optimal_learning.python.python_version.log_likelihood`.
  See this file's comments and interfaces.log_likelihood_interface for more details as well as the hpp and corresponding .cpp file.

**a. LOG MARGINAL LIKELIHOOD (LML)**

(Rasmussen & Williams, 5.4.1)
The Log Marginal Likelihood measure comes from the ideas of Bayesian model selection, which use Bayesian inference
to predict distributions over models and their parameters.  The cpp file comments explore this idea in more depth.
For now, we will simply state the relevant result.  We can build up the notion of the "marginal likelihood":
probability(observed data GIVEN sampling points (``X``), model hyperparameters, model class (regression, GP, etc.)),
which is denoted: ``p(y | X, \theta, H_i)`` (see the cpp file comments for more).

So the marginal likelihood deals with computing the probability that the observed data was generated from (another
way: is easily explainable by) the given model.

The marginal likelihood is in part paramaterized by the model's hyperparameters; e.g., as mentioned above.  Thus
we can search for the set of hyperparameters that produces the best marginal likelihood and use them in our model.
Additionally, a nice property of the marginal likelihood optimization is that it automatically trades off between
model complexity and data fit, producing a model that is reasonably simple while still explaining the data reasonably
well.  See the cpp file comments for more discussion of how/why this works.

In general, we do not want a model with perfect fit and high complexity, since this implies overfit to input noise.
We also do not want a model with very low complexity and poor data fit: here we are washing the signal out with
(assumed) noise, so the model is simple but it provides no insight on the data.

This is not magic.  Using GPs as an example, if the covariance function is completely mis-specified, we can blindly
go through with marginal likelihood optimization, obtain an "optimal" set of hyperparameters, and proceed... never
realizing that our fundamental assumptions are wrong.  So care is always needed.

**b. LEAVE ONE OUT CROSS VALIDATION (LOO-CV)**

(Rasmussen & Williams, Chp 5.4.2)
In cross validation, we split the training data, X, into two sets--a sub-training set and a validation set.  Then we
train a model on the sub-training set and test it on the validation set.  Since the validation set comes from the
original training data, we can compute the error.  In effect we are examining how well the model explains itself.

Leave One Out CV works by considering n different validation sets, one at a time.  Each point of X takes a turn
being the sole member of the validation set.  Then for each validation set, we compute a log pseudo-likelihood, measuring
how probable that validation set is given the remaining training data and model hyperparameters.

Again, we can maximize this quanitity over hyperparameters to help us choose the "right" set for the GP.

"""

import copy

import numpy
import emcee

import moe.build.GPP as C_GP
from moe.optimal_learning.python.cpp_wrappers import cpp_utils
from moe.optimal_learning.python.cpp_wrappers.covariance import SquareExponential
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess
from moe.optimal_learning.python.cpp_wrappers.knowledge_gradient_mcmc import GaussianProcessMCMC

class GaussianProcessLogLikelihoodMCMC:

    r"""Class for computing log likelihood-like measures of model fit via C++ wrappers (currently log marginal and leave one out cross validation).

    See :class:`moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLogMarginalLikelihood` and
    :class:`moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLeaveOneOutLogLikelihood`
    classes below for some more details on these metrics. Users may find it more convenient to
    construct these objects instead of a :class:`~moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLogLikelihood`
    object directly. Since these various metrics are fairly different, the member function docs
    in this class will remain generic.

    .. Note:: Equivalent methods of :class:`moe.optimal_learning.python.interfaces.log_likelihood_interface.GaussianProcessLogLikelihoodInterface` and
      :class:`moe.optimal_learning.python.interfaces.optimization_interface.OptimizableInterface`
      are aliased below (e.g., :class:`~moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLogLikelihood.problem_size` and
      :class:`~moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLogLikelihood.num_hyperparameters`,
      :class:`~moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLogLikelihood.compute_log_likelihood` and
      :class:`~moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLogLikelihood.compute_objective_function`, etc).

    See gpp_model_selection.hpp/cpp for further overview and in-depth discussion, respectively.

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
        self.n_chains = max(n_hypers, 2*(2*self._historical_data.dim+1+self._num_derivatives))

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
    def derivatives(self):
        return self._derivatives

    @property
    def num_derivatives(self):
        return self._num_derivatives

    @property
    def models(self):
        return self._models

    def get_historical_data_copy(self):
        """Return the data (points, function values, noise) specifying the prior of the Gaussian Process.

        :return: object specifying the already-sampled points, the objective value at those points, and the noise variance associated with each observation
        :rtype: data_containers.HistoricalData

        """
        return copy.deepcopy(self._historical_data)

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
            sampler = emcee.EnsembleSampler(self.n_chains, 1+self.dim + self._num_derivatives + 1,
                                            self.compute_log_likelihood)

            # Do a burn-in in the first iteration
            if not self.burned:
                # Initialize the walkers by sampling from the prior
                if self.prior is None:
                    self.p0 = numpy.random.rand(self.n_chains, 1+self.dim + self._num_derivatives + 1)
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
            # Instantiate a GP for each hyperparameter configuration
            cov_hyps = sample[:(self.dim+1)]
            hypers_list.append(cov_hyps)
            se = SquareExponential(cov_hyps)
            if self.noisy:
                noise = sample[(self.dim+1):]
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
        cov_hyps = hyps[:(self.dim+1)]
        noise = hyps[(self.dim+1):]
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
                        cpp_utils.cppify_hyperparameters(cov_hyps),
                        cpp_utils.cppify(self._derivatives), self._num_derivatives,
                        cpp_utils.cppify(noise),
                )
        except:
            return -numpy.inf

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