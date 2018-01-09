/*!
  \file gpp_model_selection.cpp
  \rst
  Table of Contents:

  1. FILE OVERVIEW
  2. MATHEMATICAL OVERVIEW

     a. LOG LIKELIHOOD METRICS OF MODEL QUALITY

        i. BAYESIAN MODEL SELECTION

           A. LOG MARGINAL LIKELIHOOD (LML)

        ii. CROSS VALIDATION (CV)

           A. LEAVE ONE OUT CROSS VALIDATION (LOO-CV)

        iii. REMARKS

  3. CODE HIERARCHY

  **1. FILE OVERVIEW**

  As a preface, if you are not already familiar with GPs and their implementation, you should read the file comments for
  gpp_math.hpp/cpp first.  If you are unfamiliar with the concept of model selection or optimization methods, please read
  the file comments for gpp_model_selection.hpp first.

  This file provides implementations for various log likelihood measures of model quality (marginal likelihood,
  leave one out cross validation).  The functions to optimize these measures all live in the header file (they are
  templated) and the hearts of the optimization routines are in gpp_optimization.hpp.

  **2. MATHEMATICAL OVERVIEW**

  **2a. LOG LIKELIHOOD METRICS OF MODEL QUALITY**

  **2a, i. BAYESIAN MODEL SELECTION**

  (Rasmussen & Williams, 5.2)
  Bayesian model selection uses Bayesian inference to predict various distributional properties of models and their parameters.
  This analysis is usually performed hierarchically.  The pararameters, w, are at the lowest level; e.g., the weights
  in linear regression.  The hyperparameters, ``\theta``, sit at the next level; these are free-floating parameters that
  control model performance such as length scales.  Finally, the highest level is a discrete set of models, ``H_i``; e.g.,
  GPs resulting from different classes of covariance or entirely different models.

  Then the "posterior over the parameters" is:

  ``p(w | y, X, \theta, H_i) = \frac{p(y | X, w, H_i) * p(w | \theta, H_i)}{p(y | X, \theta, H_i)}``

  where ``p(y | X, w, H_i)`` is the "likelihood" (of the model) and ``p(w | \theta, H_i)`` is the "parameter prior" (encoding what
  we know about the model parameters *before* seeing data).  The denominator is the "marginal likelihood."  Using
  total probability, it is given as:

  ``p(y | X, \theta, H_i) = \int p(y | X, w, H_i) * p(w | \theta, H_i) dw``,

  where we have marginalized out ``w ~ p(w | \theta, H_i)`` from the likelihood, ``p(y | X, w, H_i)``, to produce the marginal likelihood.
  This is just the integral of the numerator over ``w``, so it can be viewed as a normalizing constant too--however you
  like think about Bayes' Theorem.

  From that marginal likelihood, we can also produce the posterior over hyperparameters:

  ``p(\theta | y, X, H_i) = \frac{p(y | X, \theta, H_i * p(\theta, H_i)}{p(y | X, H_i)}``

  where ``p(\theta, H_i)`` is called the "hyper-prior" and the denominator is constructed as before.

  And finally, the posterior for the models:

  ``p(H_i | y, X) = \frac{p(y | X, H_i) * p(H_i)}{p(y | X)}``

  Here ``p(y | X)`` is not an integral since ``H_i`` is discrete: ``= \sum_i p(y | X, H_i) * p(H_i)``

  These integrals can be extremely complicated, often requiring Monte-Carlo (MC) integration.  In particular,
  computing the posterior over hyperparameters is usually particularly painful.  This would be the ideal step in
  the Bayesian framework for selecting hyperparameters; with the posterior distribution we can simply choose the most
  likely.  Using the higher level analysis also requires knowing or forming the priors over hyperparameters and/or
  models, which can also be tricky when information is lacking.

  To make the problem more tractable, people usually end up working with maximization of the marginal likelihood wrt
  ``\theta``.  This process can be tricky: if we parameterize everything in our model (many ``\theta``), it is easy to
  overfit and produce nonsense where the model reacts strongly to noise.

  One nice property of the marginal likelihood is that it automatically trades off between model complexity and data fit.
  In the next section, we will make this explicit for GP-based models.  But it is true in general.  First, note
  that marginal likelihood is a probability distribution so it integrates to 1.  A simple model (with few
  parameters), can only explain a few data sets and the likelihood will be high for these and 0 for the rest.  A
  very complex model can explain many data sets, so it will be nonzero over a wider region but never obtain values
  as high as the simple model.  Marginal likelihood optimization trades off between these, in principle automatically
  finding the simplest model that still explains the data.

  Note: we are usually only working with one model (the GP with a specified class of covariance function), so we drop ``H_i``.

  **2a, i, A. LOG MARGINAL LIKELIHOOD (LML)**

  (Rasmussen & Williams, 5.4.1)
  Now we specialize the Bayesian technique for GPs.  We will be working with the log marginal likelihood (LML).
  GPs are non-parametric in the sense that we are not directly computing parameters ``\beta_i`` to evaluate
  ``y_i = x_i\beta_i`` as in linear regression.  However, the function values ``f`` at the training points
  (``points_sampled``) are analogous to parameters; the more sampled points, the more complex the model (more params).

  Then the log marginal likelihood, ``\log(p(y | X, \theta))``, examines the probability of the model given the data.  It is::

    log p(y | X, \theta) = -\frac{1}{2} * y^T * K^-1 * y - \frac{1}{2} * \log(det(K)) - \frac{n}{2} * \log(2*pi)  (Equation 1)

  where ``n`` is ``num_sampled``, ``\theta`` are the hyperparameters, and ``\log`` is the natural logarithm.

  To maximize ``p``, we can equivalently maximize ``log(p)``.

  Since we almost never work with noise-free priors, we drop the subscript ``y`` from ``K_y`` in future discussion; e.g,. in
  ``LogMarginalLikelihoodEvaluator::ComputeLogLikelihood()``.

  Anyway, despite the complex integrals and whatnot in the general Bayesian model inference method, the LML for GPs
  is very easy to derive.  From the discussion in gpp_math.hpp/cpp, it should be clear that the GP is distributed
  like a multi-variate Gaussian:

  ``N(\mu, K) = \frac{1}{\sqrt{(2\pi)^n * det(K)}} * \exp(-\frac{1}{2}*(y-\mu)^T * K^-1 * (y - \mu))``

  And our ``GP ~ N(0, K)``; hence ``p(y | X, \theta) ~ N(0,K)`` by definition.  Take the logarithm and we reach Equation 1.

  Let's look at the terms:

  * ``term1 = -\frac{1}{2} * y^T * K^-1 * y``
  * ``term2 = -\frac{1}{2} * \log(det(K))``
  * ``term3 = -\frac{n}{2} * \log(2*pi)``

  In detail:

  * ``term1``: the only term that depends on the observed function values, ``y``.  This is called the "data-fit."  The data fit
    decreases monotonically as covariance length scales (part of hyperparameters) increase since long lengths force the
    model to change 'slowly', making it less flexible.

  * ``term2``: this term is the complexity penalty, depending only on ``K``.  One can think of complexity as a concrete measure of
    how "bumpy" (short length scales, high frequency) or "not-bumpy" (long length scales, low frequency) the distribution is.\*
    This term increases with length scale; low frequency => low complexity.

  * ``term3``: the simplest term, this is just from normalization (so the hyper-volume under the hyper-surface is 1)

  \* Here we're talking about the variance of the distribution, not the mean since ``term2`` only deals with ``K``; e.g.,
  imagine plotting the variance or the 95% confidence region.

  Hence optimizing LML is a matter of balancing data fit with model complexity.  We made this same observation about
  LML in the general discussion about Bayesian model selection, arguing about properties of distributions; here we see
  the explicit terms responsible for the trade-off.

  This final point is important so here it is again, verbatim from the hpp: This is not magic.  Using GPs as an example,
  if the covariance function is completely mis-specified, we can blindly go through with marginal likelihood
  optimization, obtain an "optimal" set of hyperparameters, and proceed... never realizing that our fundamental
  assumptions are wrong.  So care is always needed.

  **2a, ii. CROSS VALIDATION (CV)**

  (Rasmussen & Williams, Chp 5.3)
  Cross validation deals with estimating the generalization error.  This is done by splitting the training data, ``X``, into
  two disjoint sets: ``X'`` and ``V``.  ``X'`` is the reduced training set and ``V`` is the validation set.  ``X = X' \cup V`` and
  ``X' \cap V = \emptyset``.

  Then we train the model on ``X'`` and evaluate its performance on ``V`` (where we know the answer from direct obsevation, since
  ``V \subset X``).  The errors in prediction on ``V`` serve as a proxy for the generalization error.

  If ``|V|`` is too small, large variance in the estimated error can result (e.g., what if we pick particularly "unlucky"
  data for ``V``?).  But choosing a small ``X'`` leads to a model that is too poorly trained to provide useful outputs.  Instead,
  a common technique is to choose multiple disjoint V and run error estimation on each of them.

  Taking this idea to the extreme, we choose ``n`` sets ``V`` with ``|V| = 1``.  Hence the name "Leave One Out," since each member
  of ``X`` takes a turn being the sole validation point.

  Finally, what measure do we use to evaluate the performance of the model (trained on ``X'``) on ``V``?  According to R&W, the
  most common measure is the squared error loss, but for probabilistic models like GPs, the log probability loss makes more
  sense.

  **2a, ii, A. LEAVE ONE OUT CROSS VALIDATION (LOO-CV)**

  (Rasmussen & Williams, Chp 5.4.2)
  For a GP, LOO-CV, which we denote ``L_{LOO}`` is:

  Let ``\log p(y_i | X_{-i}, y_{-i}, \theta) = -0.5\log(\sigma_i^2) - 0.5*(y_i - \mu_i)^2/\sigma_i^2 - 0.5\log(2\pi)``.
  Then we compute:

  ``L_{LOO}(X ,y, \theta) = \sum_{i = 1}^n \log p(y_i | X_{-i}, y_{-i})``.

  where ``X_{-i}`` and ``y_{-i}`` are the training data with the ``i``-th point removed.  Then ``X_i`` is taken as the point to sample.
  ``\sigma_i^2`` and ``\mu_i`` are the GP (predicted) variance/mean at the point to sample, ``X \ X_{-i}``.

  On the surface, it looks like we would have to form an entirely new GP ``n`` times to compute ``\mu_i, \sigma_i^2`` for ``i = 1..n``.
  However, in each of these cases, the ``X_{-i}, f_{-i}, \sigma_n_{-i}`` inputs to the GP are almost identical, so there is a
  lot of nearly redundant work going on.  If we take K computed with the full training data (``X, f, \sigma_n``), and remove
  the ``i``-th row and ``i``-th column, we get the ``K_{-i}`` associated with ``X_{-i}``, ``f_{-i}, \sigma_n_{-i}``.  And then the fundamental
  GP operations involve applying ``K_{-i}^-1``: see Equation 2, 3 in gpp_math.cpp.

  By partitioning ``K`` into 4 block matrices, we can (after row/column swap) isolate the element being removed, and solve for it
  in terms of ``K^-1``.
  Wikipedia for block matrix inverse: http://en.wikipedia.org/wiki/Invertible_matrix#Blockwise_inversion
  We can then directly show that:

  | ``\mu_i = y_i - \alpha_i / K^{-1}_{ii}``
  | ``\sigma_i^2 = 1/K^{-1}_{ii}``

  where ``\alpha = K^-1 * y``

  **2a, iii. REMARKS**

  We have not been able to find much information on whether LML or LOO-CV should be preferred.  Rasmussen & Williams
  say, "[the LML gives] the probability of the observations \emph{given the assumptions of the model}."  And "the
  [frequentist LOO-CV] gives an estimate for the predictive probability, whether or not the assumptions of the model
  may be fulfilled."  Thus Wahba (1990, sec 4.8) argues that LOO-CV should be more robust to model mis-specification
  (e.g., wrong class of covariance function).

  **3. CODE HIERARCHY**

  There are currently several top-level entry points for model selection (defined in the hpp) including
  'dumb' search, gradient descent, and Newton:

  * LatinHypercubeSearchHyperparameterOptimization:

    * Estimates the best model fit with a 'dumb' search over hyperparameters
    * Selects random guesses based on latin hypercube sampling
    * This calls:

      EvaluateLogLikelihoodAtPointList:

      * Evaluates the selected log likelihood measure at each set of hyperparameters
      * Multithreaded over starting locations
      * This calls:

        MultistartOptimizer<...>::MultistartOptimize(...) for multistarting (see gpp_optimization.hpp)
        with the NullOptimizer

  * MultistartGradientDescentHyperparameterOptimization:

    * Finds the best model by optimizing hyperparmeters to find maxima of log likelihood metrics
    * Selects random starting locations based on latin hypercube sampling
    * Multithreaded over starting locations
    * Optimizes with restarted gradient descent; collects results and updates the solution as new optima are found
    * This calls:

      MultistartOptimizer<...>::MultistartOptimize(...) for multistarting (see gpp_optimization.hpp) together with
      GradientDescentOptimizer::Optimize<ObjectiveFunctionEvaluator, Domain>() (see gpp_optimization.hpp)

  * MultistartNewtonHyperparameterOptimization: (Recommended)

    * Finds the best model by optimizing hyperparmeters to find maxima of log likelihood metrics
    * Selects random starting locations based on latin hypercube sampling
    * Multithreaded over starting locations
    * Optimizes with (modified) Newton's Method; collects results and updates the solution as new optima are found
    * This calls:

      MultistartOptimizer<...>::MultistartOptimize(...) for multistarting (see gpp_optimization.hpp) together with
      NewtonOptimizer::Optimize<ObjectiveFunctionEvaluator, Domain>() (see gpp_optimization.hpp)

  At the moment, we have two choices for the template parameter LogLikelihoodEvaluator: LML and LOO-CV.
  Each of these make additional lower level calls to gpp_linear_algebra routines and gpp_covariance routines.  The
  details (with derivations and optimizations where appropriate) are specified in the function implementation docs and
  will not be repeated here.

  * LogMarginalLikelihoodEvaluator:

    At the bottom level, LogMarginalLikelihoodEvaluator contains member functions for computing the LML, its gradient
    wrt hyperparameters (of covariance), and its hessian wrt hyperparameters.  Its data members are the GP model
    inputs (sampled points, function values at sampled points, noise).  It does not know what a covariance is since
    the covariance (i.e., hyperparameters) is meant to change during optimization.

    Its member functions require a LogMarginalLikelihoodState object which is where the (stateful) covariance is kept, along
    with derived quantities that are a function of covariance and model inputs, and various temporaries.

    Computations optionally use a faster implementation using explicit matrix inverses; this is poorly conditioned
    but several times faster.

  * LeaveOneOutLogLikelihoodEvaluator:

    This class contains member fucntions for computing the LOO-CV measure.  Its structure is essentially the same as
    LogMarginalLikelihoodEvaluator.  Its members require LeaveOneOutLogLikelihoodState, just as before.

    In the discussion of LOO-CV, we indicated two possible methods to compute the quantities ``\mu_i, \sigma_i^2``.
    Both are implemented in the code, although it is currently configured to use the faster computation:

    * LeaveOneOutCoreAccurate() computes \mu_i, \sigma_i^2 the direct way by forming a new GP.  This is slow but well-conditioned.
    * LeaveOneOutCoreWithMatrixInverse() computes \mu_i, \sigma_i^2 using the described "trick".  This is fast but the results
      may be heavily affected by numerical error (if K is poorly conditioned).
\endrst*/

#include "gpp_model_selection.hpp"

#include <cmath>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_domain.hpp"
#include "gpp_exception.hpp"
#include "gpp_linear_algebra.hpp"
#include "gpp_linear_algebra-inl.hpp"
#include "gpp_math.hpp"
#include "gpp_mock_optimization_objective_functions.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimizer_parameters.hpp"

namespace optimal_learning {

namespace {  // utilities for building covariance matrix and its hyperparameter derivatives

/*!\rst
  Computes the covariance matrix of a list of points, ``X_i``.  Matrix is computed as:

  ``A_{i,j} = covariance(X_i, X_j)``.

  Result is SPD assuming covariance operator is SPD and points are unique.

  Generally, this is called from other functions with "points_sampled" as the input and not any
  arbitrary list of points; hence the very specific input name.

  Point list cannot contain duplicates.  Doing so (or providing nearly duplicate points) can lead to
  semi-definite matrices or very poor numerical conditioning.

  \param
    :covariance: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
    :noise_variance[num_sampled]: i-th entry is amt of noise variance to add to i-th diagonal entry; i.e., noise measuring i-th point
    :points_sampled[dim][num_sampled]: list of points
    :dim: spatial dimension of a point
    :num_sampled: number of points
  \output
    :cov_matrix[num_sampled*(length+1)][num_sampled*(length+1)]: computed covariance matrix
\endrst*/
OL_NONNULL_POINTERS void BuildCovarianceMatrixWithNoiseVariance(const CovarianceInterface& covariance,
                                                                double const * restrict points_sampled,
                                                                int dim, int num_sampled,
                                                                double const * restrict noise_variance,
                                                                int const * derivatives,
                                                                int num_derivatives,
                                                                double * restrict cov_matrix) noexcept {
    // we only work with lower triangular parts of symmetric matrices, so only fill half of it
    double * cov_temp = new double [(num_derivatives+1)*(num_derivatives+1)]();
    for (int i = 0; i < num_sampled; ++i) {//col
        for (int j = i; j < num_sampled; ++j) {//row
            covariance.Covariance(points_sampled + j*dim, derivatives, num_derivatives,
                                  points_sampled + i*dim, derivatives, num_derivatives,
                                  cov_temp);
            for (int m = 0; m < num_derivatives+1; ++m){//col
                for (int n = 0; n < num_derivatives+1; ++n){//row
                    int row = j*(num_derivatives+1) + n;
                    int col = i*(num_derivatives+1) + m;
                    if (row>=col){
                        cov_matrix[row + col*num_sampled*(num_derivatives+1)] = cov_temp[n];
                    }
                    if (row == col){
                        cov_matrix[row + col*num_sampled*(num_derivatives+1)] += noise_variance[m];
                    }
                }
                cov_temp += num_derivatives+1;
            }
            cov_temp -= Square(num_derivatives+1);
        }
    }
    delete [] cov_temp;
}

/*!\rst
  Build ``A_{jik} = \pderiv{K_{ij}}{\theta_k}``

  Hence the outer loop structure is identical to BuildCovarianceMatrix().

  Note the structure of the resulting tensor is ``num_hyperparameters`` blocks of size
  ``num_sampled X num_sampled``.  Consumers of this want ``dK/d\theta_k`` located sequentially.
  However, for a given pair of points (x, y), it is more efficient to compute all
  hyperparameter derivatives at once.  Thus the innermost loop writes to all
  ``num_hyperparameters`` blocks at once.

  Consumers of this result generally require complete storage (i.e., will not take advantage
  of its symmetry), so instead of ignoring the upper triangles, we copy them from the
  (already-computed) lower triangles to avoid redundant work.

  Since CovarianceInterface.HyperparameterGradCovariance() returns a vector of size ``|\theta_k|``,
  the inner loop writes all relevant entries of ``A_{jik}`` simultaneously to prevent recomputation.

  \param
    :covariance: the CovarianceFunction object encoding assumptions about the GP's behavior on our data
    :points_sampled[dim][num_sampled]: list of points
    :dim: spatial dimension of a point
    :num_sampled: number of points
  \output
    :grad_cov_matrix[num_sampled][num_sampled][covariance.GetNumberOfHyperparameters()]: gradients of covariance matrix wrt hyperparameters
\endrst*/
OL_NONNULL_POINTERS void BuildHyperparameterGradCovarianceMatrix(const CovarianceInterface& covariance,
                                                                 double const * restrict points_sampled,
                                                                 int dim, int num_sampled,
                                                                 double const * restrict noise_variance,
                                                                 int const * derivatives,
                                                                 int num_derivatives,
                                                                 double * restrict grad_cov_matrix) noexcept {
  int num_hyperparameters = covariance.GetNumberOfHyperparameters();
  num_hyperparameters += num_derivatives+1;

  const int offset = (num_sampled*(num_derivatives+1))*(num_sampled*(num_derivatives+1));

  std::vector<double> grad_covariance((dim+1)*(num_derivatives+1)*(num_derivatives+1), 0.0);

  // pointer to row that we're copying from
  // operator is symmetric, so we compute just the lower triangle & copy into the upper triangle
  for (int i = 0; i < num_sampled; ++i) { // col
      for (int j = 0; j < i; ++j) { // row
          for (int i_hyper = 0; i_hyper < num_hyperparameters; ++i_hyper) {
              for (int m = 0; m < num_derivatives+1; ++m){
                  for (int n = 0; n < num_derivatives+1; ++n){
                      int row = j*(num_derivatives+1)+m;
                      int col = i*(num_derivatives+1)+n;
                      grad_cov_matrix[row + col*num_sampled*(num_derivatives+1) + i_hyper*offset] =
                         grad_cov_matrix[col + row*num_sampled*(num_derivatives+1) + i_hyper*offset];
                  }
              }
          }
      }

      for (int j = i; j < num_sampled; ++j) { //row
          // compute all hyperparameter derivs at once for efficiency
          covariance.HyperparameterGradCovariance(points_sampled+j*dim, derivatives, num_derivatives,
                                                  points_sampled+i*dim, derivatives, num_derivatives,
                                                  grad_covariance.data());
          for (int i_hyper = 0; i_hyper < num_hyperparameters; ++i_hyper) {
              // have to write each deriv to the correct block, due to the block structure of the output
              for (int m = 0; m < num_derivatives+1; ++m){
                  for (int n = 0; n < num_derivatives+1; ++n){
                      int row = j*(num_derivatives+1)+m;
                      int col = i*(num_derivatives+1)+n;
                      if (i_hyper < dim+1){
                          grad_cov_matrix[row + col*num_sampled*(num_derivatives+1) + i_hyper*offset] =
                             grad_covariance[i_hyper + m*(dim+1) + n*(dim+1)*(num_derivatives+1)];
                      }
                      else{
                          if (row == col && i_hyper == (m+dim+1)){
                              grad_cov_matrix[row + col*num_sampled*(num_derivatives+1) + i_hyper*offset] = 1;
                          }
                          else{
                              grad_cov_matrix[row + col*num_sampled*(num_derivatives+1) + i_hyper*offset] = 0;
                          }
                      }
                  }
              }
          }
      }
  }
}
}  // end unnamed namespace

LogMarginalLikelihoodEvaluator::LogMarginalLikelihoodEvaluator(double const * restrict points_sampled_in,
                                                               double const * restrict points_sampled_value_in,
                                                               int const * derivatives_in,
                                                               int num_derivatives_in,
                                                               int dim_in, int num_sampled_in)
    : dim_(dim_in),
      num_sampled_(num_sampled_in),
      num_derivatives_(num_derivatives_in),
      derivatives_(derivatives_in, derivatives_in + num_derivatives_in),
      points_sampled_(points_sampled_in, points_sampled_in + num_sampled_in*dim_in),
      points_sampled_value_(points_sampled_value_in, points_sampled_value_in + (num_derivatives_in+1)*num_sampled_in) {
}

void LogMarginalLikelihoodEvaluator::BuildHyperparameterGradCovarianceMatrix(
    LogMarginalLikelihoodState * log_likelihood_state) const noexcept {
  optimal_learning::BuildHyperparameterGradCovarianceMatrix(*log_likelihood_state->covariance_ptr,
                                                            points_sampled_.data(), dim_, num_sampled_,
                                                            log_likelihood_state->noise_variance.data(),
                                                            derivatives_.data(), num_derivatives_,
                                                            log_likelihood_state->grad_hyperparameter_cov_matrix.data());
}

/*void LogMarginalLikelihoodEvaluator::BuildHyperparameterHessianCovarianceMatrix(
    LogMarginalLikelihoodState * log_likelihood_state,
    double * hessian_hyperparameter_cov_matrix) const noexcept {
  optimal_learning::BuildHyperparameterHessianCovarianceMatrix(*log_likelihood_state->covariance_ptr,
                                                               points_sampled_.data(), dim_, num_sampled_,
                                                               hessian_hyperparameter_cov_matrix);
}*/

void LogMarginalLikelihoodEvaluator::FillLogLikelihoodState(LogMarginalLikelihoodState * log_likelihood_state) const {
  // K_chol
  optimal_learning::BuildCovarianceMatrixWithNoiseVariance(*log_likelihood_state->covariance_ptr, points_sampled_.data(),
                                                           dim_, num_sampled_, log_likelihood_state->noise_variance.data(),
                                                           derivatives_.data(), num_derivatives_, log_likelihood_state->K_chol.data());

  // Adding the variance of measurement noise to the covariance matrix
  for (int i = 0; i < num_sampled_ * (1+num_derivatives_); i++){
     log_likelihood_state->K_chol[i + i*num_sampled_ * (1+num_derivatives_)] += 1.0e-6;
  }

  // TODO(GH-211): Re-examine ignoring singular covariance matrices here
  int OL_UNUSED(chol_info) = ComputeCholeskyFactorL(num_sampled_*(num_derivatives_+1),
                                                    log_likelihood_state->K_chol.data());

  // K_inv_y
  double mean_ = 0.0;
  for (int i=0; i<num_sampled_; ++i){
     mean_ += points_sampled_value_[i*(num_derivatives_+1)];
  }
  mean_ /= num_sampled_;
  std::copy(points_sampled_value_.begin(), points_sampled_value_.end(), log_likelihood_state->y.begin());
  std::copy(points_sampled_value_.begin(), points_sampled_value_.end(), log_likelihood_state->K_inv_y.begin());
  for (int i=0; i<num_sampled_; ++i){
    log_likelihood_state->K_inv_y[i*(num_derivatives_+1)] -= mean_;
    log_likelihood_state->y[i*(num_derivatives_+1)] -= mean_;
  }
  CholeskyFactorLMatrixVectorSolve(log_likelihood_state->K_chol.data(), num_sampled_*(num_derivatives_+1),
                                   log_likelihood_state->K_inv_y.data());
}

/*!\rst
  .. NOTE:: These comments have been copied into the matching method of LogMarginalLikelihood in python_version/log_likelihood.py.

  ``log p(y | X, \theta) = -\frac{1}{2} * y^T * K^-1 * y - \frac{1}{2} * \log(det(K)) - \frac{n}{2} * \log(2*pi)``

  where n is ``num_sampled``, ``\theta`` are the hyperparameters, and ``\log`` is the natural logarithm.  In the following,

  * ``term1 = -\frac{1}{2} * y^T * K^-1 * y``
  * ``term2 = -\frac{1}{2} * \log(det(K))``
  * ``term3 = -\frac{n}{2} * \log(2*pi)``

  For an SPD matrix ``K = L * L^T``,

  ``det(K) = \Pi_i L_ii^2``

  We could compute this directly and then take a logarithm.  But we also know:

  ``\log(det(K)) = 2 * \sum_i \log(L_ii)``

  The latter method is (currently) preferred for computing ``\log(det(K))`` due to reduced chance for overflow
  and (possibly) better numerical conditioning.
\endrst*/
double LogMarginalLikelihoodEvaluator::ComputeLogLikelihood(
    const LogMarginalLikelihoodState& log_likelihood_state) const noexcept {

  // compute term2 = - \frac{1}{2} * \log(det(K)) using the SPD matrix simplification given above (i.e., without computing det directly)
  double log_marginal_term2 = 0.0;
  double const * restrict K_chol_ptr = log_likelihood_state.K_chol.data();
  for (int i = 0; i < num_sampled_*(num_derivatives_+1); ++i) {
    log_marginal_term2 -= std::log(K_chol_ptr[i]);
    K_chol_ptr += num_sampled_*(num_derivatives_+1);
  }

  // compute term1 = -\frac{1}{2} * y^T * K^-1 * y
  // term1 = y^T * K_inv_y
  double log_marginal_term1 = -0.5*DotProduct(log_likelihood_state.y.data(),
                                              log_likelihood_state.K_inv_y.data(), num_sampled_*(num_derivatives_+1));
  // compute term3 = -\frac{n}{2} * \log(2*pi), where log(2*pi) has been precomputed
  double log_marginal_term3 = -0.5*static_cast<double>(num_sampled_*(num_derivatives_+1))*kLog2Pi;

  return log_marginal_term1 + log_marginal_term2 + log_marginal_term3;
}

/*!\rst
  .. NOTE:: These comments have been copied into the matching method of LogMarginalLikelihood in python_version/log_likelihood.py.

  Computes::

    \pderiv{log(p(y | X, \theta))}{\theta_k} = \frac{1}{2} * y_i * \pderiv{K_{ij}}{\theta_k} * y_j -
                                               \frac{1}{2} * trace(K^{-1}_{ij}\pderiv{K_{ij}}{\theta_k})

  Or equivalently::

    = \frac{1}{2} * trace([\alpha_i \alpha_j - K^{-1}_{ij}]*\pderiv{K_{ij}}{\theta_k}),

  where ``\alpha_i = K^{-1}_{ij} * y_j``
\endrst*/
#define OL_USE_INVERSE 0
void LogMarginalLikelihoodEvaluator::ComputeGradLogLikelihood(LogMarginalLikelihoodState * log_likelihood_state,
                                                              double * restrict grad_log_marginal) const noexcept {

#if OL_USE_INVERSE == 1
  std::vector<double> K_inv(num_sampled_*(num_derivatives_+1)*num_sampled_*(num_derivatives_+1));
  SPDMatrixInverse(log_likelihood_state->K_chol.data(), num_sampled_, K_inv.data());
#endif

  // TODO(GH-156): is it more stable to compute:
  //  tr(\alpha\alpha^T dK/d\theta) - tr( K \ dK/d\theta)
  //  OR tr((\alpha\alpha^T - K^-1) dK/d\theta) (UNLIKELY...)
  //  OR tr(\alpha\alpha^T dK/d\theta - K \ dK/d\theta)

  BuildHyperparameterGradCovarianceMatrix(log_likelihood_state);

  double * restrict grad_hyperparameter_cov_matrix_ptr = log_likelihood_state->grad_hyperparameter_cov_matrix.data();
  const int num_hyperparameters = log_likelihood_state->num_hyperparameters;
  // compute gradient  as 0.5 * \alpha^T * (dK/d\theta) * \alpha - 0.5 * tr(K^-1 * dK/d\theta)
  for (int i_hyper = 0; i_hyper < num_hyperparameters; ++i_hyper) {
    // grad_hyperparameter_cov_matrix for the i_hyper-th hyperparameter is not needed after the
    // i_hyper-th iteration of this loop. Thus we overwrite it with K^-1 * grad_hyperparameter_cov_matrix

    // computing 0.5 * \alpha^T * grad_hyperparameter_cov_matrix * \alpha, where \alpha = K^-1 * y (aka K_inv_y)
    // temp_vec := grad_hyperparameter_cov_matrix * K_inv_y
    GeneralMatrixVectorMultiply(grad_hyperparameter_cov_matrix_ptr, 'N', log_likelihood_state->K_inv_y.data(),
                                1.0, 0.0, num_sampled_*(num_derivatives_+1), num_sampled_*(num_derivatives_+1),
                                num_sampled_*(num_derivatives_+1), log_likelihood_state->temp_vec.data());
    // could use dsymv here but it appears to be slightly slower in practice
    // SymmetricMatrixVectorMultiply(grad_hyperparameter_cov_matrix_ptr, K_inv_y.data(), num_sampled_, temp_vec.data());
    // computes 0.5 * K_inv_y^T * temp_vec
    grad_log_marginal[i_hyper] = 0.5*DotProduct(log_likelihood_state->K_inv_y.data(),
                                                log_likelihood_state->temp_vec.data(), num_sampled_*(num_derivatives_+1));

    // compute -0.5 * tr(K^-1 * dK/d\theta)
#if OL_USE_INVERSE == 1
    // avoid performing the matrix product; only calculate terms needed for trace
    grad_log_marginal[i_hyper] -= 0.5*TraceOfGeneralMatrixMatrixMultiply(K_inv.data(), grad_hyperparameter_cov_matrix_ptr,
                                                                         num_sampled_*(num_derivatives_+1));
#else
    // avoid forming the matrix inverse explicitly; improves numerical accuracy
    // overwrites grad_hyperparameter_cov_matrix := K^-1 * grad_hyperparameter_cov_matrix
    CholeskyFactorLMatrixMatrixSolve(log_likelihood_state->K_chol.data(), num_sampled_*(num_derivatives_+1),
                                     num_sampled_*(num_derivatives_+1), grad_hyperparameter_cov_matrix_ptr);
    grad_log_marginal[i_hyper] -= 0.5*MatrixTrace(grad_hyperparameter_cov_matrix_ptr, num_sampled_*(num_derivatives_+1));
#endif

    grad_hyperparameter_cov_matrix_ptr += Square(num_sampled_*(num_derivatives_+1));
  }
}

void LogMarginalLikelihoodState::SetHyperparameters(const EvaluatorType& log_likelihood_eval,
                                                    double const * restrict hyperparameters) {
  // update hyperparameters
  int num_derivatives = log_likelihood_eval.num_derivatives();
  covariance_ptr->SetHyperparameters(hyperparameters);
  hyperparameters += covariance_ptr->GetNumberOfHyperparameters();

  for (int i = 0; i < num_derivatives+1; ++i){
      noise_variance[i] = hyperparameters[i];
  }

  // evaluate derived quantities
  log_likelihood_eval.FillLogLikelihoodState(this);
}

void LogMarginalLikelihoodState::SetupState(const EvaluatorType& log_likelihood_eval,
                                            double const * restrict hyperparameters) {
  if (unlikely(num_sampled != log_likelihood_eval.num_sampled()) || unlikely(num_derivatives != log_likelihood_eval.num_derivatives())) {
    num_sampled = log_likelihood_eval.num_sampled();
    num_derivatives = log_likelihood_eval.num_derivatives();
    num_hyperparameters = covariance_ptr->GetNumberOfHyperparameters() + num_derivatives + 1;
    K_chol.resize(Square(num_sampled*(num_derivatives+1)));
    K_inv_y.resize(num_sampled*(num_derivatives+1));
    y.resize(num_sampled*(num_derivatives+1));
    grad_hyperparameter_cov_matrix.resize(num_hyperparameters*Square(num_sampled*(num_derivatives+1)));
    temp_vec.resize(num_sampled*(num_derivatives+1));
  }

  // set hyperparameters and derived quantities
  SetHyperparameters(log_likelihood_eval, hyperparameters);
}

LogMarginalLikelihoodState::LogMarginalLikelihoodState(const EvaluatorType& log_likelihood_eval,
                                                       const CovarianceInterface& covariance_in,
                                                       const std::vector<double> noise_variance_in)
    : dim(log_likelihood_eval.dim()),
      num_sampled(log_likelihood_eval.num_sampled()),
      num_derivatives(log_likelihood_eval.num_derivatives()),
      num_hyperparameters(covariance_in.GetNumberOfHyperparameters() + num_derivatives + 1),
      covariance_ptr(covariance_in.Clone()),
      noise_variance(noise_variance_in),
      K_chol(Square(num_sampled*(num_derivatives+1))),
      K_inv_y(num_sampled*(num_derivatives+1)),
      y(num_sampled*(num_derivatives+1)),
      grad_hyperparameter_cov_matrix(num_hyperparameters*Square(num_sampled*(num_derivatives+1))),
      temp_vec(num_sampled*(num_derivatives+1)) {
  std::vector<double> hyperparameters(num_hyperparameters);
  covariance_ptr->GetHyperparameters(hyperparameters.data());
  for (int i = 0; i<num_derivatives+1; i++){
      hyperparameters[i+covariance_in.GetNumberOfHyperparameters()] = noise_variance_in[i];
  }
  SetupState(log_likelihood_eval, hyperparameters.data());
}

LogMarginalLikelihoodState::LogMarginalLikelihoodState(LogMarginalLikelihoodState&& OL_UNUSED(other)) = default;

}  // end namespace optimal_learning
