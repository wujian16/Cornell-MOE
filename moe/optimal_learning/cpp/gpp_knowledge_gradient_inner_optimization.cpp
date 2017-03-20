/*!
  \file gpp_knowledge_gradient_inner_optimization.cpp
  \rst
\endrst*/

#include "gpp_knowledge_gradient_inner_optimization.hpp"

#include <cmath>

#include <memory>

#include <stdlib.h>

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_optimizer_parameters.hpp"

namespace optimal_learning {

FuturePosteriorMeanEvaluator::FuturePosteriorMeanEvaluator(const GaussianProcess& gaussian_process_in,
    double const * coefficient,
    double const * to_sample,
    const int num_to_sample,
    int const * to_sample_derivatives,
    int num_derivatives,
    double const * chol,
    double const * train_sample)
    : dim_(gaussian_process_in.dim()),
      gaussian_process_(&gaussian_process_in),
      coeff_(coeff(coefficient, num_to_sample, num_derivatives)),
      to_sample_(to_sample_points(to_sample, num_to_sample)),
      num_to_sample_(num_to_sample),
      to_sample_derivatives_(derivatives(to_sample_derivatives, num_derivatives)),
      num_derivatives_(num_derivatives),
      chol_(cholesky(chol, num_to_sample, num_derivatives)),
      train_sample_(train_sample_precompute(train_sample, num_to_sample, num_derivatives)) {
}

/*!\rst
  Uses analytic formulas to compute EI when ``num_to_sample = 1`` and ``num_being_sampled = 0`` (occurs only in 1,0-EI).
  In this case, the single-parameter (posterior) GP is just a Gaussian.  So the integral in EI (previously eval'd with MC)
  can be computed 'exactly' using high-accuracy routines for the pdf & cdf of a Gaussian random variable.
  See Ginsbourger, Le Riche, and Carraro.
\endrst*/
double FuturePosteriorMeanEvaluator::ComputePosteriorMean(StateType * ps_state) const {
  double to_sample_mean = 0;
  gaussian_process_->ComputeMeanOfPoints(ps_state->points_to_sample_state, &to_sample_mean);

  double * var_star = new double[num_to_sample_*(1+num_derivatives_)]();
  gaussian_process_->ComputeCovarianceOfPoints(&(ps_state->points_to_sample_state), to_sample_.data(), num_to_sample_,
                                               to_sample_derivatives_.data(), num_derivatives_, true, train_sample_.data(), var_star);

  TriangularMatrixVectorSolve(chol_.data(), 'N', num_to_sample_*(1+num_derivatives_), num_to_sample_*(1+num_derivatives_), var_star);
  to_sample_mean += DotProduct(var_star, coeff_.data(), num_to_sample_*(1+num_derivatives_));
  delete [] var_star;

  return -to_sample_mean;
}

/*!\rst
  Differentiates OnePotentialSampleExpectedImprovementEvaluator::ComputeExpectedImprovement wrt
  ``points_to_sample`` (which is just ONE point; i.e., 1,0-EI).
  Again, this uses analytic formulas in terms of the pdf & cdf of a Gaussian since the integral in EI (and grad EI)
  can be evaluated exactly for this low dimensional case.
  See Ginsbourger, Le Riche, and Carraro.
\endrst*/
void FuturePosteriorMeanEvaluator::ComputeGradPosteriorMean(
    StateType * ps_state,
    double * restrict grad_PS) const {
  double * grad_mu = ps_state->grad_mu.data();
  gaussian_process_->ComputeGradMeanOfPoints(ps_state->points_to_sample_state, grad_mu);

  double * grad_cov = new double[dim_*num_to_sample_*(1+num_derivatives_)]();
  gaussian_process_->ComputeGradCovarianceOfPoints(&(ps_state->points_to_sample_state), to_sample_.data(),
                                                   num_to_sample_, to_sample_derivatives_.data(), num_derivatives_,
                                                   false, nullptr, grad_cov);
  std::vector<double> temp(coeff_);

  TriangularMatrixVectorSolve(chol_.data(), 'T', num_to_sample_*(1+num_derivatives_),
                              num_to_sample_*(1+num_derivatives_), temp.data());
  GeneralMatrixVectorMultiply(grad_cov, 'N', temp.data(), 1.0, 1.0,
                              dim_, num_to_sample_*(1+num_derivatives_),
                              dim_, grad_mu);
  delete [] grad_cov;

  for (int i = 0; i < dim_; ++i) {
    grad_PS[i] = -grad_mu[i];
  }
}

void FuturePosteriorMeanState::SetCurrentPoint(const EvaluatorType& ps_evaluator,
                                         double const * restrict point_to_sample_in) {
  // update current point in union_of_points
  std::copy(point_to_sample_in, point_to_sample_in + dim, point_to_sample.data());

  // evaluate derived quantities
  points_to_sample_state.SetupState(*ps_evaluator.gaussian_process(), point_to_sample.data(),
                                    num_to_sample, 0, num_derivatives);
}

FuturePosteriorMeanState::FuturePosteriorMeanState(
    const EvaluatorType& ps_evaluator,
    double const * restrict point_to_sample_in,
    bool configure_for_gradients)
    : dim(ps_evaluator.dim()),
      num_derivatives(configure_for_gradients ? num_to_sample : 0),
      point_to_sample(point_to_sample_in, point_to_sample_in + dim),
      points_to_sample_state(*ps_evaluator.gaussian_process(), point_to_sample.data(), num_to_sample, nullptr, 0, num_derivatives),
      grad_mu(dim*num_derivatives) {
}

FuturePosteriorMeanState::FuturePosteriorMeanState(FuturePosteriorMeanState&& OL_UNUSED(other)) = default;

void FuturePosteriorMeanState::SetupState(const EvaluatorType& ps_evaluator,
                                    double const * restrict point_to_sample_in) {
  if (unlikely(dim != ps_evaluator.dim())) {
    OL_THROW_EXCEPTION(InvalidValueException<int>, "Evaluator's and State's dim do not match!", dim, ps_evaluator.dim());
  }

  SetCurrentPoint(ps_evaluator, point_to_sample_in);
}

/*!\rst
  Perform multistart gradient descent (MGD) to solve the q,p-EI problem (see ComputeOptimalPointsToSample and/or
  header docs), starting from ``num_multistarts`` points selected randomly from the within th domain.
  This function is a simple wrapper around ComputeOptimalPointsToSampleViaMultistartGradientDescent(). It additionally
  generates a set of random starting points and is just here for convenience when better initial guesses are not
  available.
  See ComputeOptimalPointsToSampleViaMultistartGradientDescent() for more details.
  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :optimizer_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_dynamic), chunk_size (0).
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    :normal_rng[thread_schedule.max_num_threads]: a vector of NormalRNG objects that provide
      the (pesudo)random source for MC integration
  \output
    :found_flag[1]: true if best_next_point corresponds to a nonzero EI
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :normal_rng[thread_schedule.max_num_threads]: NormalRNG objects will have their state changed due to random draws
    :best_next_point[dim][num_to_sample]: points yielding the best EI according to MGD
\endrst*/
template <typename DomainType>
void ComputeOptimalFuturePosteriorMean(const GaussianProcess& gaussian_process, double const * coefficient,
                                       double const * to_sample,
                                       const int num_to_sample,
                                       int const * to_sample_derivatives,
                                       int num_derivatives,
                                       double const * chol,
                                       double const * train_sample,
                                       const GradientDescentParameters& optimizer_parameters,
                                       const DomainType& domain, double const * restrict initial_guess,
                                       bool * restrict found_flag, double * restrict best_next_point) {
  if (unlikely(optimizer_parameters.max_num_restarts <= 0)) {
    return;
  }
  bool configure_for_gradients = true;
  OL_VERBOSE_PRINTF("Posterior Mean Optimization via %s:\n", OL_CURRENT_FUNCTION_NAME);

  // special analytic case when we are not using (or not accounting for) multiple, simultaneous experiments
  FuturePosteriorMeanEvaluator ps_evaluator(gaussian_process, coefficient, to_sample,
                                      num_to_sample, to_sample_derivatives,
                                      num_derivatives, chol, train_sample);

  typename FuturePosteriorMeanEvaluator::StateType ps_state(ps_evaluator, initial_guess, configure_for_gradients);

  GradientDescentOptimizer<FuturePosteriorMeanEvaluator, DomainType> gd_opt;
  gd_opt.Optimize(ps_evaluator, optimizer_parameters, domain, &ps_state);
  ps_state.GetCurrentPoint(best_next_point);
}

// template explicit instantiation declarations, see gpp_common.hpp header comments, item 6
template void ComputeOptimalFuturePosteriorMean(const GaussianProcess& gaussian_process, double const * coefficient,
                                                double const * to_sample,
                                                const int num_to_sample,
                                                int const * to_sample_derivatives,
                                                int num_derivatives,
                                                double const * chol,
                                                double const * train_sample,
                                                const GradientDescentParameters& optimizer_parameters,
                                                const TensorProductDomain& domain, double const * restrict initial_guess,
                                                bool * restrict found_flag, double * restrict best_next_point);
template void ComputeOptimalFuturePosteriorMean(const GaussianProcess& gaussian_process, double const * coefficient,
                                                double const * to_sample,
                                                const int num_to_sample,
                                                int const * to_sample_derivatives,
                                                int num_derivatives,
                                                double const * chol,
                                                double const * train_sample,
                                                const GradientDescentParameters& optimizer_parameters,
                                                const SimplexIntersectTensorProductDomain& domain, double const * restrict initial_guess,
                                                bool * restrict found_flag, double * restrict best_next_point);
}  // end namespace optimal_learning