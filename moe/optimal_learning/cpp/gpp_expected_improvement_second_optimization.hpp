#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_KNOWLEDGE_GRADIENT_INNER_OPTIMIZATION_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_KNOWLEDGE_GRADIENT_INNER_OPTIMIZATION_HPP_

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include <boost/math/distributions/normal.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_exception.hpp"
#include "gpp_covariance.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimizer_parameters.hpp"
#include "gpp_random.hpp"

namespace optimal_learning {

struct FuturePosteriorMeanState;
/*!\rst
  This is a specialization of the ExpectedImprovementEvaluator class for when the number of potential samples is 1; i.e.,
  ``num_to_sample == 1`` and the number of concurrent samples is 0; i.e. ``num_being_sampled == 0``.
  In other words, this class only supports the computation of 1,0-EI.  In this case, we have analytic formulas
  for computing EI and its gradient.
  Thus this class does not perform any explicit numerical integration, nor do its EI functions require access to a
  random number generator.
  This class's methods have some parameters that are unused or redundant.  This is so that the interface matches that of
  the more general ExpectedImprovementEvaluator.
  For other details, see ExpectedImprovementEvaluator for more complete description of what EI is and the outputs of
  EI and grad EI computations.
\endrst*/
class FuturePosteriorMeanEvaluator final {
 public:
  using StateType = FuturePosteriorMeanState;

  /*!\rst
    Constructs a OnePotentialSampleExpectedImprovementEvaluator object.  All inputs are required; no default constructor nor copy/assignment are allowed.
    \param
      :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
        that describes the underlying GP
      :best_so_far: best (minimum) objective function value (in ``points_sampled_value``)
  \endrst*/
  FuturePosteriorMeanEvaluator(const GaussianProcess& gaussian_process_in,
                               double const * coefficient,
                               double const * to_sample,
                               const int num_to_sample,
                               int const * to_sample_derivatives,
                               int num_derivatives,
                               double const * chol,
                               double const * train_sample);

  int dim() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim_;
  }

  const GaussianProcess * gaussian_process() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return gaussian_process_;
  }

  std::vector<double> to_sample_points(double const * to_sample,
                                       int num_to_sample) noexcept OL_WARN_UNUSED_RESULT {
    std::vector<double> result(num_to_sample*dim_);
    std::copy(to_sample, to_sample + num_to_sample*dim_, result.data());
    return result;
  }

  std::vector<int> derivatives(int const * to_sample_derivatives,
                                  int num_derivatives) noexcept OL_WARN_UNUSED_RESULT {
    std::vector<int> result(num_derivatives);
    std::copy(to_sample_derivatives, to_sample_derivatives + num_derivatives, result.data());
    return result;
  }

  std::vector<double> coeff(double const * coefficient, double const * chol,
                            int num_to_sample, int num_derivatives) noexcept OL_WARN_UNUSED_RESULT {
    std::vector<double> result(num_to_sample*(1+num_derivatives));
    std::copy(coefficient, coefficient + num_to_sample*(1+num_derivatives), result.data());
    TriangularMatrixVectorSolve(chol, 'T', num_to_sample_*(1+num_derivatives_),
                                num_to_sample_*(1+num_derivatives_), result.data());
    return result;
  }

  std::vector<double> coeff_combine(double const * coefficient, double const * train_sample,
                                    int num_to_sample, int num_derivatives) noexcept OL_WARN_UNUSED_RESULT {
    std::vector<double> temp(num_to_sample*(1+num_derivatives));
    std::copy(coefficient, coefficient + num_to_sample*(1+num_derivatives), temp.data());

    std::vector<double> result(gaussian_process_->get_K_inv_y());
    int num_observations = gaussian_process_->num_sampled() * (1 + gaussian_process_->num_derivatives());
    GeneralMatrixVectorMultiply(train_sample, 'N', temp.data(), -1.0, 1.0,
                                num_observations, num_to_sample*(1+num_derivatives),
                                num_observations, result.data());
    return result;
  }

  /*!\rst
    Wrapper for ComputeExpectedImprovement(); see that function for details.
  \endrst*/
  double ComputeObjectiveFunction(StateType * ps_state) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    return ComputePosteriorMean(ps_state);
  }

  /*!\rst
    Wrapper for ComputeGradExpectedImprovement(); see that function for details.
  \endrst*/
  void ComputeGradObjectiveFunction(StateType * ps_state, double * restrict grad_PS) const OL_NONNULL_POINTERS {
    ComputeGradPosteriorMean(ps_state, grad_PS);
  }

  /*!\rst
    Computes the expected improvement ``EI(Xs) = E_n[[f^*_n(X) - min(f(Xs_1),...,f(Xs_m))]^+]``
    Uses analytic formulas to evaluate the expected improvement.
    \param
      :ei_state[1]: properly configured state object
    \output
      :ei_state[1]: state with temporary storage modified
    \return
      the expected improvement from sampling ``point_to_sample``
  \endrst*/
  double ComputePosteriorMean(StateType * ps_state) const;

  /*!\rst
    Computes the (partial) derivatives of the expected improvement with respect to the point to sample.
    Uses analytic formulas to evaluate the spatial gradient of the expected improvement.
    \param
      :ei_state[1]: properly configured state object
    \output
      :ei_state[1]: state with temporary storage modified
      :grad_EI[dim]: gradient of EI, ``\pderiv{EI(x)}{x_d}``, where ``x`` is ``points_to_sample``
  \endrst*/
  void ComputeGradPosteriorMean(StateType * ps_state, double * restrict grad_PS) const;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(FuturePosteriorMeanEvaluator);

 private:
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  const int dim_;

  //! pointer to gaussian process used in computations
  const GaussianProcess * gaussian_process_;

  //! points to sample in the next sampling step
  const std::vector<double> to_sample_;

  //! number of points in the next sampling step
  const int num_to_sample_;

  //! the dims with derivatives
  const std::vector<int> to_sample_derivatives_;

  //! number of derivatives
  const int num_derivatives_;

  //! transpose(Chol)^-1 * (sampled Z)
  const std::vector<double> coeff_;
  const std::vector<double> coeff_combined_;
};

/*!\rst
  State object for OnePotentialSampleExpectedImprovementEvaluator.  This tracks the *ONE* ``point_to_sample``
  being evaluated via expected improvement.
  This is just a special case of ExpectedImprovementState; see those class docs for more details.
  See general comments on State structs in ``gpp_common.hpp``'s header docs.
\endrst*/
struct FuturePosteriorMeanState final {
  using EvaluatorType = FuturePosteriorMeanEvaluator;

  /*!\rst
    Constructs an OnePotentialSampleExpectedImprovementState object for the purpose of computing EI
    (and its gradient) over the specified point to sample.
    This establishes properly sized/initialized temporaries for EI computation, including dependent state from the
    associated Gaussian Process (which arrives as part of the ``ei_evaluator``).
    .. WARNING::
         This object is invalidated if the associated ei_evaluator is mutated.  SetupState() should be called to reset.
    .. WARNING::
         Using this object to compute gradients when ``configure_for_gradients`` := false results in UNDEFINED BEHAVIOR.
    \param
      :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
      :point_to_sample[dim]: point at which to evaluate EI and/or its gradient to check their value in future experiments (i.e., test point for GP predictions)
      :configure_for_gradients: true if this object will be used to compute gradients, false otherwise
  \endrst*/
  FuturePosteriorMeanState(const EvaluatorType& ps_evaluator, const int num_fidelity_in, double const * restrict point_to_sample_in, bool configure_for_gradients);

  FuturePosteriorMeanState(FuturePosteriorMeanState&& other);

  int GetProblemSize() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim - num_fidelity;
  }

  std::vector<double> BuildUnionOfPoints(double const * restrict points_to_sample) noexcept OL_WARN_UNUSED_RESULT {
    std::vector<double> union_of_points(dim);
    std::copy(points_to_sample, points_to_sample + dim - num_fidelity, union_of_points.data());
    std::fill(union_of_points.data() + dim - num_fidelity, union_of_points.data() + dim, 1.0);
    return union_of_points;
  }

  /*!\rst
    Get ``point_to_sample``: the potential future sample whose EI (and/or gradients) is being evaluated
    \output
      :point_to_sample[dim]: potential sample whose EI is being evaluted
  \endrst*/
  void GetCurrentPoint(double * restrict point_to_sample_out) const noexcept OL_NONNULL_POINTERS {
    std::copy(point_to_sample.data(), point_to_sample.data() + dim - num_fidelity, point_to_sample_out);
  }

  void Initialize(const EvaluatorType& ps_evaluator);

  /*!\rst
    Change the potential sample whose EI (and/or gradient) is being evaluated.
    Update the state's derived quantities to be consistent with the new point.
    \param
      :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
      :point_to_sample[dim]: potential future sample whose EI (and/or gradients) is being evaluated
  \endrst*/
  void SetCurrentPoint(const EvaluatorType& ps_evaluator,
                       double const * restrict point_to_sample_in) OL_NONNULL_POINTERS;

  /*!\rst
    Configures this state object with a new ``point_to_sample``, the location of the potential sample whose EI is to be evaluated.
    Ensures all state variables & temporaries are properly sized.
    Properly sets all dependent state variables (e.g., GaussianProcess's state) for EI evaluation.
    .. WARNING::
         This object's state is INVALIDATED if the ei_evaluator (including the GaussianProcess it depends on) used in
         SetupState is mutated! SetupState() should be called again in such a situation.
    \param
      :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
      :point_to_sample[dim]: potential future sample whose EI (and/or gradients) is being evaluated
  \endrst*/
  void SetupState(const EvaluatorType& ps_evaluator,
                  double const * restrict point_to_sample_in) OL_NONNULL_POINTERS;

  // size information
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  const int dim;
  //! dim of the fidelity
  const int num_fidelity;
  //! number of points to sample (i.e., the "q" in q,p-EI); MUST be 1
  const int num_to_sample = 1;
  //! number of derivative terms desired (usually 0 for no derivatives or num_to_sample)
  const int num_derivatives;

  //! point at which to evaluate EI and/or its gradient (e.g., to check its value in future experiments)
  std::vector<double> point_to_sample;

  std::vector<double> K_star;
  std::vector<double> grad_K_star;

  UniformRandomGenerator randomGenerator;
  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(FuturePosteriorMeanState);
};

/*!\rst
  Set up vector of FuturePosteriorMeanState::StateType.

  This is a utility function just for reducing code duplication.

  \param
    :ei_evaluator: evaluator object associated w/the state objects being constructed
    :points_to_sample[dim][num_to_sample]: initial points to load into state (must be a valid point for the problem);
      i.e., points at which to evaluate EI and/or its gradient
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrently experiments
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the p in q,p-EI)
    :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
    :configure_for_gradients: true if these state objects will be used to compute gradients, false otherwise
    :state_vector[arbitrary]: vector of state objects, arbitrary size (usually 0)
    :normal_rng[max_num_threads]: a vector of NormalRNG objects that provide the (pesudo)random source for MC integration
  \output
    :state_vector[max_num_threads]: vector of states containing ``max_num_threads`` properly initialized state objects
\endrst*/
inline OL_NONNULL_POINTERS void SetupFuturePosteriorMeanState(
    const FuturePosteriorMeanEvaluator& fpm_evaluator,
    double const * restrict points_to_sample,
    int max_num_threads,
    bool configure_for_gradients,
    const int num_fidelity,
    std::vector<typename FuturePosteriorMeanEvaluator::StateType> * state_vector) {
  state_vector->reserve(max_num_threads);
  for (int i = 0; i < max_num_threads; ++i) {
    state_vector->emplace_back(fpm_evaluator, num_fidelity, points_to_sample, configure_for_gradients);
  }
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
void RestartedGradientDescentFuturePosteriorMeanOptimization(const GaussianProcess& gaussian_process, const int num_fidelity,
                                                             double const * coefficient, double const * to_sample, const int num_to_sample,
                                                             int const * to_sample_derivatives, int num_derivatives,
                                                             double const * chol, double const * train_sample,
                                                             const GradientDescentParameters& optimizer_parameters,
                                                             const DomainType& domain, double const * restrict initial_guess,
                                                             double& best_objective_value, double * restrict best_next_point) {
  if (unlikely(optimizer_parameters.max_num_restarts <= 0)) {
    return;
  }
  bool configure_for_gradients = true;
  OL_VERBOSE_PRINTF("Posterior Mean Optimization via %s:\n", OL_CURRENT_FUNCTION_NAME);

  // special analytic case when we are not using (or not accounting for) multiple, simultaneous experiments
  FuturePosteriorMeanEvaluator ps_evaluator(gaussian_process, coefficient, to_sample,
                                            num_to_sample, to_sample_derivatives,
                                            num_derivatives, chol, train_sample);

  typename FuturePosteriorMeanEvaluator::StateType ps_state(ps_evaluator, num_fidelity, initial_guess, configure_for_gradients);

  GradientDescentOptimizer<FuturePosteriorMeanEvaluator, DomainType> gd_opt;
  gd_opt.Optimize(ps_evaluator, optimizer_parameters, domain, &ps_state);
  ps_state.GetCurrentPoint(best_next_point);
  best_objective_value = ps_evaluator.ComputeObjectiveFunction(&ps_state);
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
void ComputeOptimalFuturePosteriorMean(const GaussianProcess& gaussian_process, const int num_fidelity, double const * coefficient,
                                       double const * to_sample, const int num_to_sample, int const * to_sample_derivatives,
                                       int num_derivatives, double const * chol, double const * train_sample,
                                       const GradientDescentParameters& optimizer_parameters, const DomainType& domain,
                                       int max_num_threads, double const * restrict start_point_set,
                                       int num_multistarts, double * restrict best_function_value, double * restrict best_next_point);

// template explicit instantiation declarations, see gpp_common.hpp header comments, item 6
extern template void ComputeOptimalFuturePosteriorMean(const GaussianProcess& gaussian_process, const int num_fidelity, double const * coefficient,
                                                       double const * to_sample, const int num_to_sample, int const * to_sample_derivatives,
                                                       int num_derivatives, double const * chol, double const * train_sample,
                                                       const GradientDescentParameters& optimizer_parameters, const TensorProductDomain& domain,
                                                       int max_num_threads, double const * restrict start_point_set,
                                                       int num_multistarts, double * restrict best_function_value,
                                                       double * restrict best_next_point);
extern template void ComputeOptimalFuturePosteriorMean(const GaussianProcess& gaussian_process, const int num_fidelity, double const * coefficient,
                                                       double const * to_sample, const int num_to_sample, int const * to_sample_derivatives,
                                                       int num_derivatives, double const * chol, double const * train_sample,
                                                       const GradientDescentParameters& optimizer_parameters, const SimplexIntersectTensorProductDomain& domain,
                                                       int max_num_threads, double const * restrict start_point_set,
                                                       int num_multistarts, double * restrict best_function_value,
                                                       double * restrict best_next_point);
}  // end namespace optimal_learning
#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_KNOWLEDGE_GRADIENT_INNER_OPTIMIZATION_HPP_