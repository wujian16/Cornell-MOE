/*!
  \file gpp_knowledge_gradient_inner_optimization.hpp
  \rst
  1. OVERVIEW OF KNOWLEDGE GRADIENT WHAT ARE WE TRYING TO DO?
  2. IMPLEMENTATION NOTES
  3. CITATIONS

    **1. OVERVIEW OF KNOWLEDGE GRADIENT; WHAT ARE WE TRYING TO DO?**

    .. Note:: these comments are copied in Python: interfaces/__init__.py

    The optimization process models the objective using a Gaussian process (GP) prior
    (also called a GP predictor) based on the specified covariance and the input
    data (e.g., through member functions ComputeMeanOfPoints, ComputeVarianceOfPoints).  Using the GP,
    we can compute the knowledge gradient (KG) from sampling any particular point.  KG
    is defined relative to the best currently known value, and it represents what the
    algorithm believes is the most likely outcome from sampling a particular point in parameter
    space (aka conducting a particular experiment).

    See KnowledgeGradientEvaluator class docs for further details on computing KG.
    Both support ComputeKnowledgeGradient() and ComputeGradKnowledgeGradient().

    The behavior of the GP is controlled by its underlying
    covariance function and the data/uncertainty of prior points (experiments).

    With the ability of the compute KG, the final step is to optimize
    to find the best KG.  This is done using multistart gradient descent (MGD), in
    ComputeKGOptimalPointsToSample(). This method wraps a MGD call and falls back on random search
    if that fails. See gpp_optimization.hpp for multistart/optimization templates. This method
    can evaluate and optimize KG at serval points simultaneously; e.g., if we wanted to run 4 simultaneous
    experiments, we can use KG to select all 4 points at once.

    Additionally, there are use cases where we have existing experiments that are not yet complete but
    we have an opportunity to start some new trials. For example, maybe we are a drug company currently
    testing 2 combinations of dosage levels. We got some new funding, and can now afford to test
    3 more sets of dosage parameters. Ideally, the decision on the new experiments should depend on
    the existence of the 2 ongoing tests. We may not have any data from the ongoing experiments yet;
    e.g., they are [double]-blind trials. If nothing else, we would not want to duplicate any
    existing experiments! So we want to solve 3-KG using the knowledge of the 2 ongoing experiments.

    We call this q,p-KG, so the previous example would be 3,2-KG. So q is the number of new
    (simultaneous) experiments to select. In code, this would be the size of the output from KG
    optimization (i.e., ``best_points_to_sample``, of which there are ``q = num_to_sample points``).
    p is the number of ongoing/incomplete experiments to take into account (i.e., ``points_being_sampled``
    of which there are ``p = num_being_sampled`` points).

    Back to optimization: the idea behind gradient descent is simple.  The gradient gives us the
    direction of steepest ascent (negative gradient is steepest descent).  So each iteration, we
    compute the gradient and take a step in that direction.  The size of the step is not specified
    by GD and is left to the specific implementation.  Basically if we take steps that are
    too large, we run the risk of over-shooting the solution and even diverging.  If we
    take steps that are too small, it may take an intractably long time to reach the solution.
    Thus the magic is in choosing the step size; we do not claim that our implementation is
    perfect, but it seems to work reasonably.  See ``gpp_optimization.hpp`` for more details about
    GD as well as the template definition.

    For particularly difficult problems or problems where gradient descent's parameters are not
    well-chosen, GD can fail to converge.  If this happens, we can fall back on heuristics;
    e.g., 'dumb' search (i.e., evaluate EI at a large number of random points and take the best
    one). Naive search lives in: ComputeOptimalPointsToSampleViaLatinHypercubeSearch<>().

    **2. IMPLEMENTATION NOTES**

    a. This file has a few primary endpoints for KG optimization:

       i. ComputeKGOptimalPointsToSampleWithRandomStarts<>():

          Solves the q,p-KG problem.

          Takes in a gaussian_process describing the prior, domain, config, etc.; outputs the next best point(s) (experiment)
          to sample (run). Uses gradient descent.

       ii. ComputeKGOptimalPointsToSampleViaLatinHypercubeSearch<>():

           Estimates the q,p-KG problem.

           Takes in a gaussian_process describing the prior, domain, etc.; outputs the next best point(s) (experiment)
           to sample (run). Uses 'dumb' search.

       iii. ComputeKGOptimalPointsToSample<>() (Recommended):

            Solves the q,p-KG problem.

            Wraps the previous two items; relies on gradient descent and falls back to "dumb" search if it fails.

       .. NOTE::
           See ``gpp_knowledge_gradient_optimization.cpp``'s header comments for more detailed implementation notes.

           There are also several other functions with external linkage in this header; these
           are provided primarily to ease testing and to permit lower level access from python.

    b. See ``gpp_common.hpp`` header comments for additional implementation notes.

    **3. CITATIONS**

    a. Gaussian Processes for Machine Learning.
       Carl Edward Rasmussen and Christopher K. I. Williams. 2006.
       Massachusetts Institute of Technology.  55 Hayward St., Cambridge, MA 02142.
       http://www.gaussianprocess.org/gpml/ (free electronic copy)

    b. The Knowledge-Gradient Policy for Correlated Normal Beliefs.
       P.I. Frazier, W.B. Powell & S. Dayanik.
       INFORMS Journal on Computing, 2009.

    c. Differentiation of the Cholesky Algorithm.
       S. P. Smith. 1995.
       Journal of Computational and Graphical Statistics. Volume 4. Number 2. p134-147

    d. A Multi-points Criterion for Deterministic Parallel Global Optimization based on Gaussian Processes.
       David Ginsbourger, Rodolphe Le Riche, and Laurent Carraro.  2008.
       D´epartement 3MI. Ecole Nationale Sup´erieure des Mines. 158 cours Fauriel, Saint-Etienne, France.
       ginsbourger@emse.fr, leriche@emse.fr, carraro@emse.fr
\endrst*/

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
  FuturePosteriorMeanState(const EvaluatorType& ps_evaluator, double const * restrict point_to_sample_in, bool configure_for_gradients);

  FuturePosteriorMeanState(FuturePosteriorMeanState&& other);

  int GetProblemSize() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim;
  }

  /*!\rst
    Get ``point_to_sample``: the potential future sample whose EI (and/or gradients) is being evaluated
    \output
      :point_to_sample[dim]: potential sample whose EI is being evaluted
  \endrst*/
  void GetCurrentPoint(double * restrict point_to_sample_out) const noexcept OL_NONNULL_POINTERS {
    std::copy(point_to_sample.begin(), point_to_sample.end(), point_to_sample_out);
  }

  void Initialize(const EvaluatorType& ps_evaluator);

  /*!\rst
    Change the potential sample whose EI (and/or gradient) is being evaluated.
    Update the state's derived quantities to be consistent with the new point.
    \param
      :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
      :point_to_sample[dim]: potential future sample whose EI (and/or gradients) is being evaluated
  \endrst*/
  void SetCurrentPoint(const EvaluatorType& ei_evaluator,
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
    std::vector<typename FuturePosteriorMeanEvaluator::StateType> * state_vector) {
  state_vector->reserve(max_num_threads);
  for (int i = 0; i < max_num_threads; ++i) {
    state_vector->emplace_back(fpm_evaluator, points_to_sample, configure_for_gradients);
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
void RestartedGradientDescentFuturePosteriorMeanOptimization(const GaussianProcess& gaussian_process, double const * coefficient,
                                                           double const * to_sample,
                                                           const int num_to_sample,
                                                           int const * to_sample_derivatives,
                                                           int num_derivatives,
                                                           double const * chol,
                                                           double const * train_sample,
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

  typename FuturePosteriorMeanEvaluator::StateType ps_state(ps_evaluator, initial_guess, configure_for_gradients);

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
void ComputeOptimalFuturePosteriorMean(const GaussianProcess& gaussian_process, double const * coefficient,
                                       double const * to_sample, const int num_to_sample, int const * to_sample_derivatives,
                                       int num_derivatives, double const * chol, double const * train_sample,
                                       const GradientDescentParameters& optimizer_parameters, const DomainType& domain,
                                       int max_num_threads, double const * restrict start_point_set,
                                       int num_multistarts, double * restrict best_function_value, double * restrict best_next_point);

// template explicit instantiation declarations, see gpp_common.hpp header comments, item 6
extern template void ComputeOptimalFuturePosteriorMean(const GaussianProcess& gaussian_process, double const * coefficient,
                                                       double const * to_sample, const int num_to_sample, int const * to_sample_derivatives,
                                                       int num_derivatives, double const * chol, double const * train_sample,
                                                       const GradientDescentParameters& optimizer_parameters, const TensorProductDomain& domain,
                                                       int max_num_threads, double const * restrict start_point_set,
                                                       int num_multistarts, double * restrict best_function_value,
                                                       double * restrict best_next_point);
extern template void ComputeOptimalFuturePosteriorMean(const GaussianProcess& gaussian_process, double const * coefficient,
                                                       double const * to_sample, const int num_to_sample, int const * to_sample_derivatives,
                                                       int num_derivatives, double const * chol, double const * train_sample,
                                                       const GradientDescentParameters& optimizer_parameters, const SimplexIntersectTensorProductDomain& domain,
                                                       int max_num_threads, double const * restrict start_point_set,
                                                       int num_multistarts, double * restrict best_function_value,
                                                       double * restrict best_next_point);
}  // end namespace optimal_learning
#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_KNOWLEDGE_GRADIENT_INNER_OPTIMIZATION_HPP_