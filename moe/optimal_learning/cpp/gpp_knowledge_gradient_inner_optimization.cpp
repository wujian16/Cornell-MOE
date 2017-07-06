/*!
  \file gpp_knowledge_gradient_inner_optimization.cpp
  \rst
\endrst*/

#include "gpp_knowledge_gradient_inner_optimization.hpp"

#include <cmath>

#include <memory>

#include <stdlib.h>
#include <queue>

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
    to_sample_(to_sample_points(to_sample, num_to_sample)),
    num_to_sample_(num_to_sample),
    to_sample_derivatives_(derivatives(to_sample_derivatives, num_derivatives)),
    num_derivatives_(num_derivatives),
    coeff_(coeff(coefficient, chol, num_to_sample, num_derivatives)),
    coeff_combined_(coeff_combine(coeff_.data(), train_sample, num_to_sample, num_derivatives)) {
}

/*!\rst
  Uses analytic formulas to compute EI when ``num_to_sample = 1`` and ``num_being_sampled = 0`` (occurs only in 1,0-EI).
  In this case, the single-parameter (posterior) GP is just a Gaussian.  So the integral in EI (previously eval'd with MC)
  can be computed 'exactly' using high-accuracy routines for the pdf & cdf of a Gaussian random variable.
  See Ginsbourger, Le Riche, and Carraro.
\endrst*/
double FuturePosteriorMeanEvaluator::ComputePosteriorMean(StateType * ps_state) const {
  double to_sample_mean = gaussian_process_->get_mean();
  int num_observations = gaussian_process_->num_sampled();

  GeneralMatrixVectorMultiply(ps_state->K_star.data(), 'T', coeff_combined_.data(), 1.0, 1.0, num_observations*(gaussian_process_->num_derivatives()+1),
                              1, num_observations*(gaussian_process_->num_derivatives()+1), &to_sample_mean);

  std::vector<double> temp(num_to_sample_*(1+num_derivatives_));
  // Vars = Kst
  optimal_learning::BuildMixCovarianceMatrix(*gaussian_process_->covariance_ptr_, ps_state->point_to_sample.data(),
                                             to_sample_.data(), dim_, 1, num_to_sample_, nullptr, 0, to_sample_derivatives_.data(),
                                             num_derivatives_, temp.data());

  to_sample_mean += DotProduct(temp.data(), coeff_.data(), num_to_sample_*(1+num_derivatives_));

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
    StateType * ps_state, double * restrict grad_PS) const {
  std::vector<double> temp(dim_*num_to_sample_*(1+num_derivatives_));
  for (int i = 0; i < num_to_sample_; ++i){
    gaussian_process_->covariance_ptr_->GradCovariance(ps_state->point_to_sample.data(), nullptr, 0,
                                                       to_sample_.data() + i*dim_, to_sample_derivatives_.data(), num_derivatives_,
                                                       temp.data() + i*dim_*(1 + num_derivatives_));
  }
  GeneralMatrixVectorMultiply(temp.data(), 'N', coeff_.data(), 1.0, 0.0,
                              dim_, num_to_sample_*(1+num_derivatives_), dim_, grad_PS);

  int num_observations = gaussian_process_->num_sampled();
  GeneralMatrixVectorMultiply(ps_state->grad_K_star.data(), 'N', coeff_combined_.data(), 1.0, 1.0,
                              dim_, num_observations*(gaussian_process_->num_derivatives()+1), dim_, grad_PS);

  for (int i = 0; i < dim_-ps_state->num_fidelity; ++i) {
    grad_PS[i] = -grad_PS[i];
  }
}

void FuturePosteriorMeanState::Initialize(const EvaluatorType& ps_evaluator) {
  optimal_learning::BuildMixCovarianceMatrix(*(ps_evaluator.gaussian_process()->covariance_ptr_), point_to_sample.data(),
                                             ps_evaluator.gaussian_process()->points_sampled().data(), dim, 1,
                                             ps_evaluator.gaussian_process()->num_sampled(), nullptr, 0,
                                             ps_evaluator.gaussian_process()->derivatives().data(),
                                             ps_evaluator.gaussian_process()->num_derivatives(), K_star.data());
  if (num_derivatives > 0) {
    double * restrict gKs_temp = grad_K_star.data();
    double * restrict grad_cov_temp = new double[dim*(ps_evaluator.gaussian_process()->num_derivatives()+1)]();
    for (int j = 0; j < ps_evaluator.gaussian_process()->num_sampled(); ++j) {
      ps_evaluator.gaussian_process()->covariance_ptr_->GradCovariance(point_to_sample.data(), nullptr, 0,
                                                                       ps_evaluator.gaussian_process()->points_sampled().data() + j*dim,
                                                                       ps_evaluator.gaussian_process()->derivatives().data(),
                                                                       ps_evaluator.gaussian_process()->num_derivatives(), grad_cov_temp);
      for (int n = 0; n < ps_evaluator.gaussian_process()->num_derivatives()+1; ++n){
        int row = n + j*(ps_evaluator.gaussian_process()->num_derivatives()+1);
        for (int d = 0; d <dim; ++d){
          gKs_temp[d + row*dim] = grad_cov_temp[d+n*dim];
        }
      }
    }
    delete [] grad_cov_temp;
  }
}

void FuturePosteriorMeanState::SetCurrentPoint(const EvaluatorType& ps_evaluator,
                                         double const * restrict point_to_sample_in) {
  // update current point in union_of_points
  std::copy(point_to_sample_in, point_to_sample_in + dim - num_fidelity, point_to_sample.data());
  std::fill(point_to_sample.data() + dim - num_fidelity, point_to_sample.data() + dim, 1.0);
  // evaluate derived quantities
  Initialize(ps_evaluator);
}

FuturePosteriorMeanState::FuturePosteriorMeanState(
  const EvaluatorType& ps_evaluator,
  const int num_fidelity_in,
  double const * restrict point_to_sample_in,
  bool configure_for_gradients)
  : dim(ps_evaluator.dim()),
    num_fidelity(num_fidelity_in),
    num_derivatives(configure_for_gradients ? num_to_sample : 0),
    point_to_sample(BuildUnionOfPoints(point_to_sample_in)),
    K_star(ps_evaluator.gaussian_process()->num_sampled()*(1+ps_evaluator.gaussian_process()->num_derivatives())),
    grad_K_star(dim*ps_evaluator.gaussian_process()->num_sampled()*(1+ps_evaluator.gaussian_process()->num_derivatives())),
    randomGenerator() {
  Initialize(ps_evaluator);
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
  header docs).  Starts a GD run from each point in ``start_point_set``.  The point corresponding to the
  optimal EI\* is stored in ``best_next_point``.

  \* Multistarting is heuristic for global optimization. EI is not convex so this method may not find the true optimum.

  This function wraps MultistartOptimizer<>::MultistartOptimize() (see ``gpp_optimization.hpp``), which provides the multistarting
  component. Optimization is done using restarted Gradient Descent, via GradientDescentOptimizer<...>::Optimize() from
  ``gpp_optimization.hpp``. Please see that file for details on gradient descent and see ``gpp_optimizer_parameters.hpp``
  for the meanings of the GradientDescentParameters.

  This function (or its wrappers, e.g., ComputeOptimalPointsToSampleWithRandomStarts) are the primary entry-points for
  gradient descent based EI optimization in the ``optimal_learning`` library.

  Users may prefer to call ComputeOptimalPointsToSample(), which applies other heuristics to improve robustness.

  Currently, during optimization, we recommend that the coordinates of the initial guesses not differ from the
  coordinates of the optima by more than about 1 order of magnitude. This is a very (VERY!) rough guideline for
  sizing the domain and num_multistarts; i.e., be wary of sets of initial guesses that cover the space too sparsely.

  Solution is guaranteed to lie within the region specified by ``domain``; note that this may not be a
  true optima (i.e., the gradient may be substantially nonzero).

  .. WARNING::
       This function fails ungracefully if NO improvement can be found!  In that case,
       ``best_next_point`` will always be the first point in ``start_point_set``.
       ``found_flag`` will indicate whether this occured.

  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :optimizer_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_dynamic), chunk_size (0).
    :start_point_set[dim][num_to_sample][num_multistarts]: set of initial guesses for MGD (one block of num_to_sample points per multistart)
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_multistarts: number of points in set of initial guesses
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :normal_rng[thread_schedule.max_num_threads]: a vector of NormalRNG objects that provide
      the (pesudo)random source for MC integration
  \output
    :normal_rng[thread_schedule.max_num_threads]: NormalRNG objects will have their state changed due to random draws
    :found_flag[1]: true if ``best_next_point`` corresponds to a nonzero EI
    :best_next_point[dim][num_to_sample]: points yielding the best EI according to MGD
\endrst*/
template <typename DomainType>
void ComputeOptimalFuturePosteriorMean(
  const GaussianProcess& gaussian_process, const int num_fidelity, double const * coefficient,
  double const * to_sample, const int num_to_sample, int const * to_sample_derivatives,
  int num_derivatives, double const * chol, double const * train_sample,
  const GradientDescentParameters& optimizer_parameters, const DomainType& domain,
  int max_num_threads, double const * restrict start_point_set,
  int num_multistarts, double * restrict best_function_value, double * restrict best_next_point) {
  if (unlikely(num_multistarts <= 0)) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "num_multistarts must be > 1", num_multistarts, 1);
  }

  ThreadSchedule thread_schedule(max_num_threads, omp_sched_dynamic);

  bool configure_for_gradients = true;

  FuturePosteriorMeanEvaluator fpm_evaluator(gaussian_process, coefficient, to_sample,
                                             num_to_sample, to_sample_derivatives,
                                             num_derivatives, chol, train_sample);

  std::vector<typename FuturePosteriorMeanEvaluator::StateType> fpm_state_vector;
  SetupFuturePosteriorMeanState(fpm_evaluator, start_point_set, max_num_threads, configure_for_gradients, num_fidelity, &fpm_state_vector);

  std::vector<double> future_mean_starting(num_multistarts);
  for (int i=0; i<num_multistarts; ++i){
    fpm_state_vector[0].SetCurrentPoint(fpm_evaluator, start_point_set + i*gaussian_process.dim());
    future_mean_starting[i] = fpm_evaluator.ComputePosteriorMean(&fpm_state_vector[0]);
  }

  std::priority_queue<std::pair<double, int>> q;
  int k = 10; // number of indices we need
  for (int i = 0; i < future_mean_starting.size(); ++i) {
    if (i < k){
      q.push(std::pair<double, int>(-future_mean_starting[i], i));
    }
    else{
      if (q.top().first > -future_mean_starting[i]){
        q.pop();
        q.push(std::pair<double, int>(-future_mean_starting[i], i));
      }
    }
  }

  std::vector<double> top_k_starting(k*gaussian_process.dim());
  for (int i = 0; i < k; ++i) {
    int ki = q.top().second;
    for (int d = 0; d<gaussian_process.dim(); ++d){
      top_k_starting[i*gaussian_process.dim() + d] = start_point_set[ki*gaussian_process.dim() + d];
    }
    q.pop();
  }

  // init winner to be first point in set and 'force' its value to be -INFINITY; we cannot do worse than this
  OptimizationIOContainer io_container(fpm_state_vector[0].GetProblemSize(), -INFINITY, top_k_starting.data());

  GradientDescentOptimizer<FuturePosteriorMeanEvaluator, DomainType> gd_opt;
  MultistartOptimizer<GradientDescentOptimizer<FuturePosteriorMeanEvaluator, DomainType> > multistart_optimizer;

  multistart_optimizer.MultistartOptimize(gd_opt, fpm_evaluator, optimizer_parameters,
                                          domain, thread_schedule, top_k_starting.data(), k,
                                          fpm_state_vector.data(), nullptr, &io_container);

  *best_function_value = io_container.best_objective_value_so_far;
  std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
}

// template explicit instantiation declarations, see gpp_common.hpp header comments, item 6
template void ComputeOptimalFuturePosteriorMean(const GaussianProcess& gaussian_process, const int num_fidelity, double const * coefficient,
                                                double const * to_sample, const int num_to_sample, int const * to_sample_derivatives,
                                                int num_derivatives, double const * chol, double const * train_sample,
                                                const GradientDescentParameters& optimizer_parameters, const TensorProductDomain& domain,
                                                int max_num_threads, double const * restrict start_point_set,
                                                int num_multistarts, double * restrict best_function_value,
                                                double * restrict best_next_point);
template void ComputeOptimalFuturePosteriorMean(const GaussianProcess& gaussian_process, const int num_fidelity, double const * coefficient,
                                                double const * to_sample, const int num_to_sample, int const * to_sample_derivatives,
                                                int num_derivatives, double const * chol, double const * train_sample,
                                                const GradientDescentParameters& optimizer_parameters, const SimplexIntersectTensorProductDomain& domain,
                                                int max_num_threads, double const * restrict start_point_set,
                                                int num_multistarts, double * restrict best_function_value,
                                                double * restrict best_next_point);
}  // end namespace optimal_learning