/*!
  \file gpp_knowledge_gradient_optimization.cpp
  \rst
\endrst*/


#include "gpp_knowledge_gradient_mcmc_optimization.hpp"
#include "gpp_knowledge_gradient_inner_optimization.hpp"

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

GaussianProcessMCMC::GaussianProcessMCMC(double const * restrict hypers_mcmc,
                                         double const * restrict noises_mcmc,
                                         int number_mcmc, double const * restrict points_sampled_in,
                                         double const * restrict points_sampled_value_in,
                                         int const * restrict derivatives_in,
                                         int num_derivatives_in,
                                         int dim_in, int num_sampled_in)
    : num_mcmc_(number_mcmc),
      dim_(dim_in),
      num_sampled_(num_sampled_in),
      points_sampled_(points_sampled_in, points_sampled_in + num_sampled_in*dim_in),
      points_sampled_value_(points_sampled_value_in, points_sampled_value_in + num_sampled_in*(num_derivatives_in+1)),
      derivatives_(derivatives_in, derivatives_in + num_derivatives_in),
      num_derivatives_(num_derivatives_in) {
  gaussian_process_lst.reserve(num_mcmc_);
  const double * hypers = hypers_mcmc;
  const double * noises = noises_mcmc;
  for (int i=0; i<num_mcmc_;++i){
    SquareExponential sqexp(dim_, hypers[0], hypers+1);
    gaussian_process_lst.emplace_back(sqexp, points_sampled_.data(), points_sampled_value_.data(),
                                      noises, derivatives_.data(), num_derivatives_,
                                      dim_, num_sampled_);
    hypers += dim_+1;
    noises += num_derivatives_+1;
  }
}

template <typename DomainType>
KnowledgeGradientMCMCEvaluator<DomainType>::KnowledgeGradientMCMCEvaluator(const GaussianProcessMCMC& gaussian_process_mcmc, const int num_fidelity,
                                                                           double const * discrete_pts_lst,
                                                                           int num_pts, int num_mc_iterations,
                                                                           const DomainType& domain,
                                                                           const GradientDescentParameters& optimizer_parameters,
                                                                           double const * best_so_far,
                                                                           std::vector<typename KnowledgeGradientState<DomainType>::EvaluatorType> * evaluator_vector)
: dim_(gaussian_process_mcmc.dim()),
  num_fidelity_(num_fidelity),
  num_mcmc_hypers_(gaussian_process_mcmc.num_mcmc()),
  num_mc_iterations_(num_mc_iterations),
  best_so_far_(best_so_far_list(best_so_far)),
  optimizer_parameters_(optimizer_parameters.num_multistarts, optimizer_parameters.max_num_steps,
                        optimizer_parameters.max_num_restarts, optimizer_parameters.num_steps_averaged,
                        optimizer_parameters.gamma, optimizer_parameters.pre_mult,
                        optimizer_parameters.max_relative_change, optimizer_parameters.tolerance),
  domain_(domain),
  gaussian_process_mcmc_(&gaussian_process_mcmc),
  knowledge_gradient_evaluator_lst(evaluator_vector),
  discrete_pts_lst_(discrete_points_list(discrete_pts_lst, num_pts)),
  num_pts_(num_pts) {
    knowledge_gradient_evaluator_lst->reserve(num_mcmc_hypers_);
    double * discrete_pts = discrete_pts_lst_.data();
    for (int i=0; i<num_mcmc_hypers_; ++i){
      knowledge_gradient_evaluator_lst->emplace_back(gaussian_process_mcmc_->gaussian_process_lst[i], num_fidelity_, discrete_pts,
                                                     num_pts_, num_mc_iterations_, domain_, optimizer_parameters_,
                                                     best_so_far_[i]);
      discrete_pts += num_pts_*(dim_-num_fidelity_);
    }
}

/*!\rst
  Compute Knowledge Gradient
  This version requires the discretization of A (the feasibe domain).
  The discretization usually is: some set + points previous sampled + points being sampled + points to sample
\endrst*/
template <typename DomainType>
double KnowledgeGradientMCMCEvaluator<DomainType>::ComputeKnowledgeGradient(StateType * kg_state) const {
  double kg_value = 0.0;
  for (int i=0; i<num_mcmc_hypers_; ++i){
    kg_value += (*knowledge_gradient_evaluator_lst)[i].ComputeObjectiveFunction((*(kg_state->kg_state_list)).data()+i);
  }
  return kg_value/static_cast<double>(num_mcmc_hypers_);
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
template <typename DomainType>
void KnowledgeGradientMCMCEvaluator<DomainType>::ComputeGradKnowledgeGradient(StateType * kg_state, double * restrict grad_KG) const {
  for (int i=0; i<num_mcmc_hypers_; ++i){
    std::vector<double> temp(kg_state->dim*kg_state->num_to_sample, 0.0);
    (*knowledge_gradient_evaluator_lst)[i].ComputeGradObjectiveFunction((*(kg_state->kg_state_list)).data()+i, temp.data());
    for (int k = 0; k < kg_state->num_to_sample*dim_; ++k) {
        grad_KG[k] += temp[k];
    }
  }
  for (int k = 0; k < kg_state->num_to_sample*dim_; ++k) {
    grad_KG[k] = grad_KG[k]/static_cast<double>(num_mcmc_hypers_);
  }
}

template class KnowledgeGradientMCMCEvaluator<TensorProductDomain>;
template class KnowledgeGradientMCMCEvaluator<SimplexIntersectTensorProductDomain>;

template <typename DomainType>
void KnowledgeGradientMCMCState<DomainType>::SetCurrentPoint(const EvaluatorType& kg_evaluator,
                                                             double const * restrict points_to_sample) {
  // evaluate derived quantities for the GP
  for (int i=0; i<kg_evaluator.num_mcmc();++i){
    (kg_state_list->at(i)).SetCurrentPoint(kg_evaluator.knowledge_gradient_evaluator_list()->at(i), points_to_sample);
  }
}

template <typename DomainType>
KnowledgeGradientMCMCState<DomainType>::KnowledgeGradientMCMCState(const EvaluatorType& kg_evaluator, double const * restrict points_to_sample,
                                                                   double const * restrict points_being_sampled, int num_to_sample_in, int num_being_sampled_in, int num_pts_in,
                                                                   int const * restrict gradients_in, int num_gradients_in,
                                                                   bool configure_for_gradients, NormalRNGInterface * normal_rng_in,
                                                                   std::vector<typename KnowledgeGradientEvaluator<DomainType>::StateType> * kg_state_vector)
  : dim(kg_evaluator.dim()),
    num_to_sample(num_to_sample_in),
    num_being_sampled(num_being_sampled_in),
    num_derivatives(configure_for_gradients ? num_to_sample : 0),
    num_union(num_to_sample + num_being_sampled),
    num_pts(num_pts_in),
    gradients(gradients_in, gradients_in+num_gradients_in),
    num_gradients_to_sample(num_gradients_in),
    union_of_points(BuildUnionOfPoints(points_to_sample, points_being_sampled, num_to_sample, num_being_sampled, dim)),
    kg_state_list(kg_state_vector) {
  kg_state_list->reserve(kg_evaluator.num_mcmc());
  // evaluate derived quantities for the GP
  for (int i=0; i<kg_evaluator.num_mcmc();++i){
    kg_state_list->emplace_back(kg_evaluator.knowledge_gradient_evaluator_list()->at(i), points_to_sample, points_being_sampled,
                                num_to_sample_in, num_being_sampled_in, num_pts_in, gradients_in, num_gradients_in,
                                configure_for_gradients, normal_rng_in);
  }
}

template <typename DomainType>
KnowledgeGradientMCMCState<DomainType>::KnowledgeGradientMCMCState(KnowledgeGradientMCMCState&& OL_UNUSED(other)) = default;

template <typename DomainType>
void KnowledgeGradientMCMCState<DomainType>::SetupState(const EvaluatorType& kg_evaluator,
                                                        double const * restrict points_to_sample) {
  if (unlikely(dim != kg_evaluator.dim())) {
    OL_THROW_EXCEPTION(InvalidValueException<int>, "Evaluator's and State's dim do not match!", dim, kg_evaluator.dim());
  }

  // update quantities derived from points_to_sample
  SetCurrentPoint(kg_evaluator, points_to_sample);
}

template struct KnowledgeGradientMCMCState<TensorProductDomain>;
template struct KnowledgeGradientMCMCState<SimplexIntersectTensorProductDomain>;

/*!\rst
  This is a simple wrapper around ComputeKGOptimalPointsToSampleWithRandomStarts() and
  ComputeKGOptimalPointsToSampleViaLatinHypercubeSearch(). That is, this method attempts multistart gradient descent
  and falls back to latin hypercube search if gradient descent fails (or is not desired).
\endrst*/
template <typename DomainType>
void ComputeKGMCMCOptimalPointsToSample(GaussianProcessMCMC& gaussian_process_mcmc, const int num_fidelity,
                                        const GradientDescentParameters& optimizer_parameters,
                                        const GradientDescentParameters& optimizer_parameters_inner,
                                        const DomainType& domain, const DomainType& inner_domain, const ThreadSchedule& thread_schedule,
                                        double const * restrict points_being_sampled,
                                        double const * discrete_pts,
                                        int num_to_sample, int num_being_sampled,
                                        int num_pts, double const * best_so_far,
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
    ComputeKGMCMCOptimalPointsToSampleWithRandomStarts(gaussian_process_mcmc, num_fidelity, optimizer_parameters, optimizer_parameters_inner,
                                                       domain, inner_domain, thread_schedule, points_being_sampled, discrete_pts,
                                                       num_to_sample, num_being_sampled, num_pts,
                                                       best_so_far, max_int_steps,
                                                       &found_flag_local, uniform_generator, normal_rng,
                                                       next_points_to_sample.data());
  }

  // if gradient descent EI optimization failed OR we're only doing latin hypercube searches
  if (found_flag_local == false || lhc_search_only == true) {
    if (unlikely(lhc_search_only == false)) {
      OL_WARNING_PRINTF("WARNING: %d,%d-KG opt DID NOT CONVERGE\n", num_to_sample, num_being_sampled);
      OL_WARNING_PRINTF("Attempting latin hypercube search\n");
    }

    if (num_lhc_samples > 0) {
      // Note: using a schedule different than "static" may lead to flakiness in monte-carlo KG optimization tests.
      // Besides, this is the fastest setting.
      ThreadSchedule thread_schedule_naive_search(thread_schedule);
      thread_schedule_naive_search.schedule = omp_sched_static;
      ComputeKGMCMCOptimalPointsToSampleViaLatinHypercubeSearch(gaussian_process_mcmc, num_fidelity, optimizer_parameters_inner, domain, inner_domain,
                                                                thread_schedule_naive_search,
                                                                points_being_sampled, discrete_pts,
                                                                num_lhc_samples, num_to_sample,
                                                                num_being_sampled, num_pts, best_so_far, max_int_steps,
                                                                &found_flag_local, uniform_generator,
                                                                normal_rng, next_points_to_sample.data());

      // if latin hypercube 'dumb' search failed
      if (unlikely(found_flag_local == false)) {
        OL_ERROR_PRINTF("ERROR: %d,%d-KG latin hypercube search FAILED on\n", num_to_sample, num_being_sampled);
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
template void ComputeKGMCMCOptimalPointsToSample(
    GaussianProcessMCMC& gaussian_process_mcmc, const int num_fidelity, const GradientDescentParameters& optimizer_parameters,
    const GradientDescentParameters& optimizer_parameters_inner,
    const TensorProductDomain& domain, const TensorProductDomain& inner_domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled, double const * discrete_pts,
    int num_to_sample, int num_being_sampled,
    int num_pts, double const * best_so_far, int max_int_steps, bool lhc_search_only,
    int num_lhc_samples, bool * restrict found_flag, UniformRandomGenerator * uniform_generator,
    NormalRNG * normal_rng, double * restrict best_points_to_sample);
template void ComputeKGMCMCOptimalPointsToSample(
    GaussianProcessMCMC& gaussian_process_mcmc, const int num_fidelity, const GradientDescentParameters& optimizer_parameters,
    const GradientDescentParameters& optimizer_parameters_inner,
    const SimplexIntersectTensorProductDomain& domain, const SimplexIntersectTensorProductDomain& inner_domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled, double const * discrete_pts,
    int num_to_sample, int num_being_sampled,
    int num_pts, double const * best_so_far, int max_int_steps, bool lhc_search_only, int num_lhc_samples, bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator, NormalRNG * normal_rng, double * restrict best_points_to_sample);
}  // end namespace optimal_learning