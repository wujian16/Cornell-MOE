/*!
  \file gpp_knowledge_gradient_optimization.cpp
  \rst
\endrst*/


#include "gpp_expected_improvement_mcmc_optimization.hpp"

#include <cmath>

#include <memory>

#include <stdlib.h>

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_domain.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_optimizer_parameters.hpp"

namespace optimal_learning {

ExpectedImprovementMCMCEvaluator::ExpectedImprovementMCMCEvaluator(const GaussianProcessMCMC& gaussian_process_mcmc,
                                                                   int num_mc_iterations,
                                                                   double const * best_so_far,
                                                                   std::vector<typename ExpectedImprovementState::EvaluatorType> * evaluator_vector)
: dim_(gaussian_process_mcmc.dim()),
  num_mcmc_hypers_(gaussian_process_mcmc.num_mcmc()),
  num_mc_iterations_(num_mc_iterations),
  best_so_far_(best_so_far_list(best_so_far)),
  gaussian_process_mcmc_(&gaussian_process_mcmc),
  expected_improvement_evaluator_lst(evaluator_vector) {
    expected_improvement_evaluator_lst->reserve(num_mcmc_hypers_);
    for (int i=0; i<num_mcmc_hypers_; ++i){
      expected_improvement_evaluator_lst->emplace_back(gaussian_process_mcmc_->gaussian_process_lst[i],
                                                       num_mc_iterations_, best_so_far_[i]);
    }
}

/*!\rst
  Compute Knowledge Gradient
  This version requires the discretization of A (the feasibe domain).
  The discretization usually is: some set + points previous sampled + points being sampled + points to sample
\endrst*/
double ExpectedImprovementMCMCEvaluator::ComputeExpectedImprovement(StateType * ei_state) const {
  double ei_value = 0.0;
  for (int i=0; i<num_mcmc_hypers_; ++i){
    ei_value += (*expected_improvement_evaluator_lst)[i].ComputeObjectiveFunction((*(ei_state->ei_state_list)).data()+i);
  }
  return ei_value/static_cast<double>(num_mcmc_hypers_);
}

/*!\rst
  Computes gradient of KG (see KnowledgeGradientEvaluator::ComputeGradKnowledgeGradient) wrt points_to_sample (stored in
  ``union_of_points[0:num_to_sample]``).

  Mechanism is similar to the computation of KG, where points' contributions to the gradient are thrown out of their
  corresponding ``improvement <= 0.0``.

  Thus ``\nabla(\mu)`` only contributes when the ``winner`` (point w/best improvement this iteration) is the current point.
  That is, the gradient of ``\mu`` at ``x_i`` wrt ``x_j`` is 0 unless ``i == j`` (and only this result is stored in
  ``kg_state->grad_mu``).  The interaction with ``kg_state->grad_chol_decomp`` is harder to know a priori (like with
  ``grad_mu``) and has a more complex structure (rank 3 tensor), so the derivative wrt ``x_j`` is computed fully, and
  the relevant submatrix (indexed by the current ``winner``) is accessed each iteration.

  .. Note:: comments here are copied to _compute_grad_knowledge_gradient_monte_carlo() in python_version/knowledge_gradient.py
\endrst*/
void ExpectedImprovementMCMCEvaluator::ComputeGradExpectedImprovement(StateType * ei_state, double * restrict grad_EI) const {
  for (int i=0; i<num_mcmc_hypers_; ++i){
    std::vector<double> temp(ei_state->dim*ei_state->num_to_sample, 0.0);
    (*expected_improvement_evaluator_lst)[i].ComputeGradObjectiveFunction((*(ei_state->ei_state_list)).data()+i, temp.data());
    for (int k = 0; k < ei_state->num_to_sample*dim_; ++k) {
        grad_EI[k] += temp[k];
    }
  }
  for (int k = 0; k < ei_state->num_to_sample*dim_; ++k) {
    grad_EI[k] = grad_EI[k]/static_cast<double>(num_mcmc_hypers_);
  }
}


void ExpectedImprovementMCMCState::SetCurrentPoint(const EvaluatorType& ei_evaluator,
                                                   double const * restrict points_to_sample) {
  // evaluate derived quantities for the GP
  for (int i=0; i<ei_evaluator.num_mcmc();++i){
    (ei_state_list->at(i)).SetCurrentPoint(ei_evaluator.expected_improvement_evaluator_list()->at(i), points_to_sample);
  }
}

ExpectedImprovementMCMCState::ExpectedImprovementMCMCState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample,
                                                           double const * restrict points_being_sampled, int num_to_sample_in,
                                                           int num_being_sampled_in, int const * restrict gradients_in, int num_gradients_in,
                                                           bool configure_for_gradients, NormalRNGInterface * normal_rng_in,
                                                           std::vector<typename ExpectedImprovementEvaluator::StateType> * ei_state_vector)
    : dim(ei_evaluator.dim()),
      num_to_sample(num_to_sample_in),
      num_being_sampled(num_being_sampled_in),
      num_derivatives(configure_for_gradients ? num_to_sample : 0),
      num_union(num_to_sample + num_being_sampled),
      gradients(gradients_in, gradients_in+num_gradients_in),
      num_gradients_to_sample(num_gradients_in),
      union_of_points(BuildUnionOfPoints(points_to_sample, points_being_sampled, num_to_sample, num_being_sampled, dim)),
      ei_state_list(ei_state_vector) {
  ei_state_list->reserve(ei_evaluator.num_mcmc());
  // evaluate derived quantities for the GP
  for (int i=0; i<ei_evaluator.num_mcmc();++i){
    ei_state_list->emplace_back(ei_evaluator.expected_improvement_evaluator_list()->at(i), points_to_sample, points_being_sampled,
                                num_to_sample_in, num_being_sampled_in,
                                configure_for_gradients, normal_rng_in);
  }
}

ExpectedImprovementMCMCState::ExpectedImprovementMCMCState(ExpectedImprovementMCMCState&& OL_UNUSED(other)) = default;

void ExpectedImprovementMCMCState::SetupState(const EvaluatorType& ei_evaluator,
                                              double const * restrict points_to_sample) {
  if (unlikely(dim != ei_evaluator.dim())) {
    OL_THROW_EXCEPTION(InvalidValueException<int>, "Evaluator's and State's dim do not match!", dim, ei_evaluator.dim());
  }

  // update quantities derived from points_to_sample
  SetCurrentPoint(ei_evaluator, points_to_sample);
}

OnePotentialSampleExpectedImprovementMCMCEvaluator::OnePotentialSampleExpectedImprovementMCMCEvaluator(const GaussianProcessMCMC& gaussian_process_mcmc,
                                                                                                       double const * best_so_far,
                                                                                                       std::vector<typename OnePotentialSampleExpectedImprovementState::EvaluatorType> * evaluator_vector)
: dim_(gaussian_process_mcmc.dim()),
  num_mcmc_hypers_(gaussian_process_mcmc.num_mcmc()),
  best_so_far_(best_so_far_list(best_so_far)),
  gaussian_process_mcmc_(&gaussian_process_mcmc),
  expected_improvement_evaluator_lst(evaluator_vector) {
    expected_improvement_evaluator_lst->reserve(num_mcmc_hypers_);
    for (int i=0; i<num_mcmc_hypers_; ++i){
        expected_improvement_evaluator_lst->emplace_back(gaussian_process_mcmc_->gaussian_process_lst[i], best_so_far_[i]);
    }
}

/*!\rst
  Compute Knowledge Gradient
  This version requires the discretization of A (the feasibe domain).
  The discretization usually is: some set + points previous sampled + points being sampled + points to sample
\endrst*/
double OnePotentialSampleExpectedImprovementMCMCEvaluator::ComputeExpectedImprovement(StateType * ei_state) const {
    double ei_value = 0.0;
    for (int i=0; i<num_mcmc_hypers_; ++i){
      ei_value += (*expected_improvement_evaluator_lst)[i].ComputeObjectiveFunction((*(ei_state->ei_state_list)).data()+i);
    }
    return ei_value/static_cast<double>(num_mcmc_hypers_);
}

/*!\rst
  Computes gradient of KG (see KnowledgeGradientEvaluator::ComputeGradKnowledgeGradient) wrt points_to_sample (stored in
  ``union_of_points[0:num_to_sample]``).

  Mechanism is similar to the computation of KG, where points' contributions to the gradient are thrown out of their
  corresponding ``improvement <= 0.0``.

  Thus ``\nabla(\mu)`` only contributes when the ``winner`` (point w/best improvement this iteration) is the current point.
  That is, the gradient of ``\mu`` at ``x_i`` wrt ``x_j`` is 0 unless ``i == j`` (and only this result is stored in
  ``kg_state->grad_mu``).  The interaction with ``kg_state->grad_chol_decomp`` is harder to know a priori (like with
  ``grad_mu``) and has a more complex structure (rank 3 tensor), so the derivative wrt ``x_j`` is computed fully, and
  the relevant submatrix (indexed by the current ``winner``) is accessed each iteration.

  .. Note:: comments here are copied to _compute_grad_knowledge_gradient_monte_carlo() in python_version/knowledge_gradient.py
\endrst*/
void OnePotentialSampleExpectedImprovementMCMCEvaluator::ComputeGradExpectedImprovement(StateType * ei_state, double * restrict grad_EI) const {
  for (int i=0; i<num_mcmc_hypers_; ++i){
    std::vector<double> temp(ei_state->dim*ei_state->num_to_sample, 0.0);
    (*expected_improvement_evaluator_lst)[i].ComputeGradObjectiveFunction((*(ei_state->ei_state_list)).data()+i, temp.data());
    for (int k = 0; k < ei_state->num_to_sample*dim_; ++k) {
        grad_EI[k] += temp[k];
    }
  }
  for (int k = 0; k < ei_state->num_to_sample*dim_; ++k) {
    grad_EI[k] = grad_EI[k]/static_cast<double>(num_mcmc_hypers_);
  }
}


void OnePotentialSampleExpectedImprovementMCMCState::SetCurrentPoint(const EvaluatorType& ei_evaluator,
                                                                     double const * restrict points_to_sample) {
  // evaluate derived quantities for the GP
  for (int i=0; i<ei_evaluator.num_mcmc();++i){
    (ei_state_list->at(i)).SetCurrentPoint(ei_evaluator.expected_improvement_evaluator_list()->at(i), points_to_sample);
  }
}

OnePotentialSampleExpectedImprovementMCMCState::OnePotentialSampleExpectedImprovementMCMCState(
    const EvaluatorType& ei_evaluator,
    double const * restrict point_to_sample_in,
    double const * restrict OL_UNUSED(points_being_sampled),
    int OL_UNUSED(num_to_sample_in),
    int OL_UNUSED(num_being_sampled_in),
    bool configure_for_gradients,
    NormalRNGInterface * OL_UNUSED(normal_rng_in),
    std::vector<typename OnePotentialSampleExpectedImprovementEvaluator::StateType> * ei_state_vector)
    : dim(ei_evaluator.dim()),
      num_derivatives(configure_for_gradients ? num_to_sample : 0),
      point_to_sample(point_to_sample_in, point_to_sample_in + dim),
      ei_state_list(ei_state_vector) {
  ei_state_list->reserve(ei_evaluator.num_mcmc());
  // evaluate derived quantities for the GP
  for (int i=0; i<ei_evaluator.num_mcmc();++i){
    ei_state_list->emplace_back(ei_evaluator.expected_improvement_evaluator_list()->at(i), point_to_sample_in,
                                configure_for_gradients);
  }
}

OnePotentialSampleExpectedImprovementMCMCState::OnePotentialSampleExpectedImprovementMCMCState(OnePotentialSampleExpectedImprovementMCMCState&& OL_UNUSED(other)) = default;

void OnePotentialSampleExpectedImprovementMCMCState::SetupState(const EvaluatorType& ei_evaluator,
                                                                double const * restrict points_to_sample) {
  if (unlikely(dim != ei_evaluator.dim())) {
    OL_THROW_EXCEPTION(InvalidValueException<int>, "Evaluator's and State's dim do not match!", dim, ei_evaluator.dim());
  }

  // update quantities derived from points_to_sample
  SetCurrentPoint(ei_evaluator, points_to_sample);
}

/*!\rst
  Routes the EI computation through MultistartOptimizer + NullOptimizer to perform EI function evaluations at the list of input
  points, using the appropriate EI evaluator (e.g., monte carlo vs analytic) depending on inputs.
\endrst*/
void EvaluateEIMCMCAtPointList(GaussianProcessMCMC& gaussian_process_mcmc,
                               const ThreadSchedule& thread_schedule,
                               double const * restrict initial_guesses,
                               double const * restrict points_being_sampled,
                               int num_multistarts, int num_to_sample,
                               int num_being_sampled, double const * best_so_far,
                               int max_int_steps, bool * restrict found_flag, NormalRNG * normal_rng,
                               double * restrict function_values,
                               double * restrict best_next_point) {
    if (unlikely(num_multistarts <= 0)) {
      OL_THROW_EXCEPTION(LowerBoundException<int>, "num_multistarts must be > 1", num_multistarts, 1);
    }

    using DomainType_dummy = DummyDomain;
    DomainType_dummy dummy_domain;
    bool configure_for_gradients = false;
    if (num_to_sample == 1 && num_being_sampled == 0) {
      std::vector<typename OnePotentialSampleExpectedImprovementState::EvaluatorType> ei_evaluator_lst;

      OnePotentialSampleExpectedImprovementMCMCEvaluator ei_evaluator(gaussian_process_mcmc,
                                                                      best_so_far, &ei_evaluator_lst);

      int num_derivatives = (*ei_evaluator.expected_improvement_evaluator_list())[0].gaussian_process()->num_derivatives();
      std::vector<int> derivatives((*ei_evaluator.expected_improvement_evaluator_list())[0].gaussian_process()->derivatives());

      std::vector<typename OnePotentialSampleExpectedImprovementMCMCEvaluator::StateType> state_vector;
      std::vector<std::vector<typename OnePotentialSampleExpectedImprovementEvaluator::StateType>> ei_state_vector(thread_schedule.max_num_threads);
      SetupExpectedImprovementMCMCState(ei_evaluator, initial_guesses, points_being_sampled,
                                        num_to_sample, num_being_sampled, derivatives.data(), num_derivatives,
                                        thread_schedule.max_num_threads, configure_for_gradients,
                                        normal_rng, ei_state_vector.data(), &state_vector);

      // init winner to be first point in set and 'force' its value to be -INFINITY; we cannot do worse than this
      OptimizationIOContainer io_container(state_vector[0].GetProblemSize(), 0.0, initial_guesses);

      NullOptimizer<OnePotentialSampleExpectedImprovementMCMCEvaluator, DomainType_dummy> null_opt;
      typename NullOptimizer<OnePotentialSampleExpectedImprovementMCMCEvaluator, DomainType_dummy>::ParameterStruct null_parameters;
      MultistartOptimizer<NullOptimizer<OnePotentialSampleExpectedImprovementMCMCEvaluator, DomainType_dummy> > multistart_optimizer;
      multistart_optimizer.MultistartOptimize(null_opt, ei_evaluator, null_parameters,
                                              dummy_domain, thread_schedule, initial_guesses,
                                              num_multistarts, state_vector.data(), function_values, &io_container);
      *found_flag = io_container.found_flag;
      std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
    } else {
      std::vector<typename ExpectedImprovementState::EvaluatorType> ei_evaluator_lst;

      ExpectedImprovementMCMCEvaluator ei_evaluator(gaussian_process_mcmc, max_int_steps,
                                                    best_so_far, &ei_evaluator_lst);

      int num_derivatives = (*ei_evaluator.expected_improvement_evaluator_list())[0].gaussian_process()->num_derivatives();
      std::vector<int> derivatives((*ei_evaluator.expected_improvement_evaluator_list())[0].gaussian_process()->derivatives());

      std::vector<typename ExpectedImprovementMCMCEvaluator::StateType> state_vector;
      std::vector<std::vector<typename ExpectedImprovementEvaluator::StateType>> ei_state_vector(thread_schedule.max_num_threads);
      SetupExpectedImprovementMCMCState(ei_evaluator, initial_guesses, points_being_sampled,
                                        num_to_sample, num_being_sampled, derivatives.data(), num_derivatives,
                                        thread_schedule.max_num_threads, configure_for_gradients,
                                        normal_rng, ei_state_vector.data(), &state_vector);

      // init winner to be first point in set and 'force' its value to be -INFINITY; we cannot do worse than this
      OptimizationIOContainer io_container(state_vector[0].GetProblemSize(), 0.0, initial_guesses);

      NullOptimizer<ExpectedImprovementMCMCEvaluator, DomainType_dummy> null_opt;
      typename NullOptimizer<ExpectedImprovementMCMCEvaluator, DomainType_dummy>::ParameterStruct null_parameters;
      MultistartOptimizer<NullOptimizer<ExpectedImprovementMCMCEvaluator, DomainType_dummy> > multistart_optimizer;
      multistart_optimizer.MultistartOptimize(null_opt, ei_evaluator, null_parameters,
                                              dummy_domain, thread_schedule, initial_guesses,
                                              num_multistarts, state_vector.data(), function_values, &io_container);
      *found_flag = io_container.found_flag;
      std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
    }
}

/*!\rst
  This is a simple wrapper around ComputeKGOptimalPointsToSampleWithRandomStarts() and
  ComputeKGOptimalPointsToSampleViaLatinHypercubeSearch(). That is, this method attempts multistart gradient descent
  and falls back to latin hypercube search if gradient descent fails (or is not desired).
\endrst*/
template <typename DomainType>
void ComputeEIMCMCOptimalPointsToSample(GaussianProcessMCMC& gaussian_process_mcmc,
                                        const GradientDescentParameters& optimizer_parameters,
                                        const DomainType& domain, const ThreadSchedule& thread_schedule,
                                        double const * restrict points_being_sampled,
                                        int num_to_sample, int num_being_sampled,
                                        double const * best_so_far,
                                        int max_int_steps, bool lhc_search_only,
                                        int num_lhc_samples, bool * restrict found_flag,
                                        UniformRandomGenerator * uniform_generator,
                                        NormalRNG * normal_rng, double * restrict best_points_to_sample) {
  if (unlikely(num_to_sample <= 0)) {
    return;
  }

  std::vector<double> next_points_to_sample(gaussian_process_mcmc.dim()*num_to_sample);

  bool found_flag_local = false;
  if (lhc_search_only == false) {
    ComputeEIMCMCOptimalPointsToSampleWithRandomStarts(gaussian_process_mcmc, optimizer_parameters,
                                                       domain, thread_schedule, points_being_sampled,
                                                       num_to_sample, num_being_sampled,
                                                       best_so_far, max_int_steps,
                                                       &found_flag_local, uniform_generator, normal_rng,
                                                       next_points_to_sample.data());
  }

  // if gradient descent EI optimization failed OR we're only doing latin hypercube searches
  if (found_flag_local == false || lhc_search_only == true) {
    if (unlikely(lhc_search_only == false)) {
      OL_WARNING_PRINTF("WARNING: %d,%d-EI opt DID NOT CONVERGE\n", num_to_sample, num_being_sampled);
      OL_WARNING_PRINTF("Attempting latin hypercube search\n");
    }

    if (num_lhc_samples > 0) {
      // Note: using a schedule different than "static" may lead to flakiness in monte-carlo KG optimization tests.
      // Besides, this is the fastest setting.
      ThreadSchedule thread_schedule_naive_search(thread_schedule);
      thread_schedule_naive_search.schedule = omp_sched_static;
      ComputeEIMCMCOptimalPointsToSampleViaLatinHypercubeSearch(gaussian_process_mcmc, domain,
                                                                thread_schedule_naive_search,
                                                                points_being_sampled,
                                                                num_lhc_samples, num_to_sample,
                                                                num_being_sampled, best_so_far, max_int_steps,
                                                                &found_flag_local, uniform_generator,
                                                                normal_rng, next_points_to_sample.data());

      // if latin hypercube 'dumb' search failed
      if (unlikely(found_flag_local == false)) {
        OL_ERROR_PRINTF("ERROR: %d,%d-EI latin hypercube search FAILED on\n", num_to_sample, num_being_sampled);
      }
    } else {
      OL_WARNING_PRINTF("num_lhc_samples <= 0. Skipping latin hypercube search\n");
    }
  }
  // set outputs
  *found_flag = found_flag_local;
  std::copy(next_points_to_sample.begin(), next_points_to_sample.end(), best_points_to_sample);
}

// template explicit instantiation definitions, see gpp_common.hpp header comments, item 6
template void ComputeEIMCMCOptimalPointsToSample(
    GaussianProcessMCMC& gaussian_process_mcmc, const GradientDescentParameters& optimizer_parameters,
    const TensorProductDomain& domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled,
    int num_to_sample, int num_being_sampled,
    double const * best_so_far, int max_int_steps, bool lhc_search_only,
    int num_lhc_samples, bool * restrict found_flag, UniformRandomGenerator * uniform_generator,
    NormalRNG * normal_rng, double * restrict best_points_to_sample);
template void ComputeEIMCMCOptimalPointsToSample(
    GaussianProcessMCMC& gaussian_process_mcmc, const GradientDescentParameters& optimizer_parameters,
    const SimplexIntersectTensorProductDomain& domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled,
    int num_to_sample, int num_being_sampled,
    double const * best_so_far, int max_int_steps, bool lhc_search_only, int num_lhc_samples, bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator, NormalRNG * normal_rng, double * restrict best_points_to_sample);
}  // end namespace optimal_learning