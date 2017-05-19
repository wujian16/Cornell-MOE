'''
Created on Oct 14, 2015

@author: Aaron Klein
'''
import numpy

from moe.optimal_learning.python.base_prior import BasePrior, TophatPrior, HorseshoePrior, NormalPrior

class DefaultPrior(BasePrior):

    def __init__(self, n_dims, num_noise):
        # The number of hyperparameters
        self.n_dims = n_dims
        # The number of noises
        self.num_noise = num_noise
        # Prior for the Matern52 lengthscales
        self.tophat = TophatPrior(-5, 2)

        # Prior for the covariance amplitude
        self.ln_prior = NormalPrior(mean=0.0, sigma=1.0)

        # Prior for the noise
        self.horseshoe = HorseshoePrior(scale=0.1)

    def lnprob(self, theta):
        lp = 0
        # Covariance amplitude
        lp += self.ln_prior.lnprob(theta[0])
        # Lengthscales
        lp += self.tophat.lnprob(theta[1:-self.num_noise])
        # Noise
        for i in xrange(self.num_noise, 0, -1):
            lp += self.horseshoe.lnprob(theta[-i])
        return lp

    def sample_from_prior(self, n_samples):
        p0 = numpy.zeros([n_samples, self.n_dims])
        # Covariance amplitude
        p0[:, 0] = self.ln_prior.sample_from_prior(n_samples)[:, 0]
        # Lengthscales
        ls_sample = numpy.array([self.tophat.sample_from_prior(n_samples)[:, 0]
                              for _ in xrange(1, (self.n_dims - self.num_noise))]).T
        p0[:, 1:(self.n_dims - self.num_noise)] = ls_sample
        # Noise
        ns_sample = numpy.array([self.horseshoe.sample_from_prior(n_samples)[:, 0]
                              for _ in xrange(self.num_noise)]).T
        p0[:, -self.num_noise:] = ns_sample

        return p0
