import copy

import numpy
import emcee

import moe.build.GPP as C_GP
from moe.optimal_learning.python.cpp_wrappers import cpp_utils
from moe.optimal_learning.python.cpp_wrappers.covariance import SquareExponential
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess

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
                 log_likelihood_type=C_GP.LogLikelihoodTypes.log_marginal_likelihood, rng = None):
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

        if rng is None:
            self.rng = numpy.random.RandomState(numpy.random.randint(0, 10000))
        else:
            self.rng = rng
        self.n_hypers = n_hypers

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
            sampler = emcee.EnsembleSampler(self.n_hypers,
                                            1 + self.dim + self._num_derivatives + 1,
                                            self.compute_log_likelihood)

            # Do a burn-in in the first iteration
            if not self.burned:
                # Initialize the walkers by sampling from the prior
                if self.prior is None:
                    self.p0 = numpy.random.rand(self.n_hypers, 1 + self.dim + self._num_derivatives + 1)
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
        for sample in self.hypers:
            sample = numpy.exp(sample)
            # Instantiate a GP for each hyperparameter configuration
            cov_hyps = sample[:(self.dim+1)]
            se = SquareExponential(cov_hyps)
            noise = sample[(self.dim+1):]
            model = GaussianProcess(se, noise,
                                    self._historical_data,
                                    self.derivatives)
            self._models.append(model)

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
        cov_hyps = hyps[:(self.dim+1)]
        noise = hyps[(self.dim+1):]

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