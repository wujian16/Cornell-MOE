/*!
  \file gpp_two_step_expected_improvement_mcmc_optimization.cpp
  \rst
\endrst*/


#include "gpp_knowledge_gradient_mcmc_optimization.hpp"
#include "gpp_robust_knowledge_gradient_mcmc_optimization.hpp"

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

template <typename DomainType>
RobustKnowledgeGradientMCMCEvaluator<DomainType>::RobustKnowledgeGradientMCMCEvaluator(const GaussianProcessMCMC& gaussian_process_mcmc,
                                                                           const int num_fidelity,
                                                                           double const * discrete_pts_lst,
                                                                           int num_pts, int num_mc_iterations,
                                                                           const DomainType& domain,
                                                                           const GradientDescentParameters& optimizer_parameters,
                                                                           double const * best_so_far,
                                                                           double const factor,
                                                                           std::vector<typename RobustKnowledgeGradientState<DomainType>::EvaluatorType> * evaluator_vector)
: dim_(gaussian_process_mcmc.dim()),
  num_fidelity_(num_fidelity),
  num_mcmc_hypers_(gaussian_process_mcmc.num_mcmc()),
  num_mc_iterations_(num_mc_iterations),
  best_so_far_(best_so_far_list(best_so_far)),
  factor_(factor),
  optimizer_parameters_(optimizer_parameters.num_multistarts, optimizer_parameters.max_num_steps,
                        optimizer_parameters.max_num_restarts, optimizer_parameters.num_steps_averaged,
                        optimizer_parameters.gamma, optimizer_parameters.pre_mult,
                        optimizer_parameters.max_relative_change, optimizer_parameters.tolerance),
  domain_(domain),
  gaussian_process_mcmc_(&gaussian_process_mcmc),
  robust_knowledge_gradient_evaluator_lst(evaluator_vector),
  discrete_pts_lst_(discrete_points_list(discrete_pts_lst, num_pts)),
  num_pts_(num_pts) {
    robust_knowledge_gradient_evaluator_lst->reserve(num_mcmc_hypers_);
    double * discrete_pts = discrete_pts_lst_.data();
    for (int i=0; i<num_mcmc_hypers_; ++i){
      robust_knowledge_gradient_evaluator_lst->emplace_back(gaussian_process_mcmc_->gaussian_process_lst[i], num_fidelity_, discrete_pts,
                                                     num_pts_, num_mc_iterations_, domain_, optimizer_parameters_,
                                                     best_so_far_[i], factor_);
      discrete_pts += num_pts_*(dim_-num_fidelity_);
  }
}

/*!\rst
  compute the cost.
\endrst*/
template <typename DomainType>
double RobustKnowledgeGradientMCMCEvaluator<DomainType>::ComputeCost(StateType * vf_state) const {
  if (num_fidelity_ == 0){
    return 1.0;
  }
  else{
    double cost = 0.0;
    for (int i=0; i<vf_state->num_to_sample; ++i){
      double point_cost = 1.0;
      for (int j=dim_-num_fidelity_; j<dim_; ++j){
        point_cost *= vf_state->union_of_points[i*dim_ + j];
      }
      if (cost < point_cost){
        cost = point_cost;
      }
    }
    return cost;
  }
}

/*!\rst
  compute the gradient of the cost.
\endrst*/
template <typename DomainType>
void RobustKnowledgeGradientMCMCEvaluator<DomainType>::ComputeGradCost(StateType * vf_state, double * restrict grad_cost) const {
  std::fill(vf_state->gradcost.begin(), vf_state->gradcost.end(), 0.0);
  if (num_fidelity_ > 0){
    int index = -1;
    double cost = 0.0;
    for (int i=0; i<vf_state->num_to_sample; ++i){
      double point_cost = 1.0;
      for (int j=dim_-num_fidelity_; j<dim_; ++j){
        point_cost *= vf_state->union_of_points[i*dim_ + j];
      }
      if (cost < point_cost){
        cost = point_cost;
        index = i;
      }
    }
    for (int j=dim_-num_fidelity_; j<dim_; ++j){
      vf_state->gradcost[index*dim_ + j] = cost/vf_state->union_of_points[index*dim_ + j];
    }
  }
}

/*!\rst
  Compute Knowledge Gradient
  This version requires the discretization of A (the feasibe domain).
  The discretization usually is: some set + points previous sampled + points being sampled + points to sample
\endrst*/
template <typename DomainType>
double RobustKnowledgeGradientMCMCEvaluator<DomainType>::ComputeValueFunction(StateType * vf_state) const {
  double vf_value = 0.0;
  for (int i=0; i<num_mcmc_hypers_; ++i){
    vf_value += (*robust_knowledge_gradient_evaluator_lst)[i].ComputeObjectiveFunction((*(vf_state->vf_state_list)).data()+i);
  }
  double cost = ComputeCost(vf_state);
  return vf_value/static_cast<double>(num_mcmc_hypers_*cost);
}

/*!\rst
  Computes gradient of KG (see RobustKnowledgeGradientEvaluator::ComputeGradRobustKnowledgeGradient) wrt points_to_sample (stored in
  ``union_of_points[0:num_to_sample]``).
  Mechanism is similar to the computation of KG, where points' contributions to the gradient are thrown out of their
  corresponding ``improvement <= 0.0``.
  Thus ``\nabla(\mu)`` only contributes when the ``winner`` (point w/best improvement this iteration) is the current point.
  That is, the gradient of ``\mu`` at ``x_i`` wrt ``x_j`` is 0 unless ``i == j`` (and only this result is stored in
  ``vf_state->grad_mu``).  The interaction with ``vf_state->grad_chol_decomp`` is harder to know a priori (like with
  ``grad_mu``) and has a more complex structure (rank 3 tensor), so the derivative wrt ``x_j`` is computed fully, and
  the relevant submatrix (indexed by the current ``winner``) is accessed each iteration.
  .. Note:: comments here are copied to _compute_grad_knowledge_gradient_monte_carlo() in python_version/knowledge_gradient.py
\endrst*/
template <typename DomainType>
void RobustKnowledgeGradientMCMCEvaluator<DomainType>::ComputeGradValueFunction(StateType * vf_state, double * restrict grad_VF) const {
  double VF = 0.0;
  for (int i=0; i<num_mcmc_hypers_; ++i){
    std::vector<double> temp(vf_state->dim*vf_state->num_to_sample, 0.0);
    VF += (*robust_knowledge_gradient_evaluator_lst)[i].ComputeGradValueFunction((*(vf_state->vf_state_list)).data()+i, temp.data());
    for (int k = 0; k < vf_state->num_to_sample*dim_; ++k) {
        grad_VF[k] += temp[k];
    }
  }
  VF /= static_cast<double>(num_mcmc_hypers_);
  // cost and the grad of the cost
  double cost = ComputeCost(vf_state);
  ComputeGradCost(vf_state, vf_state->gradcost.data());

  for (int k = 0; k < vf_state->num_to_sample*dim_; ++k) {
    grad_VF[k] = grad_VF[k]/static_cast<double>(num_mcmc_hypers_);
    grad_VF[k] = (grad_VF[k]*cost - VF*vf_state->gradcost[k])/Square(cost);
  }
}

template class RobustKnowledgeGradientMCMCEvaluator<TensorProductDomain>;
template class RobustKnowledgeGradientMCMCEvaluator<SimplexIntersectTensorProductDomain>;

template <typename DomainType>
void RobustKnowledgeGradientMCMCState<DomainType>::SetCurrentPoint(const EvaluatorType& vf_evaluator,
                                                                      double const * restrict points_to_sample_in) {
  // update current point in union_of_points
  std::copy(points_to_sample_in, points_to_sample_in + dim, union_of_points.data());

  // evaluate derived quantities for the GP
  for (int i=0; i<vf_evaluator.num_mcmc();++i){
    (vf_state_list->at(i)).SetCurrentPoint(vf_evaluator.robust_knowledge_gradient_evaluator_list()->at(i), points_to_sample_in);
  }
}

template <typename DomainType>
RobustKnowledgeGradientMCMCState<DomainType>::RobustKnowledgeGradientMCMCState(const EvaluatorType& vf_evaluator, double const * restrict points_to_sample,
                                                                   double const * restrict points_being_sampled, int num_to_sample_in, int num_being_sampled_in, int num_pts_in,
                                                                   int const * restrict gradients_in, int num_gradients_in,
                                                                   bool configure_for_gradients, NormalRNGInterface * normal_rng_in,
                                                                   std::vector<typename RobustKnowledgeGradientEvaluator<DomainType>::StateType> * vf_state_vector)
  : dim(vf_evaluator.dim()),
    num_to_sample(num_to_sample_in),
    num_being_sampled(num_being_sampled_in),
    num_derivatives(configure_for_gradients ? num_to_sample : 0),
    num_union(num_to_sample + num_being_sampled),
    num_pts(num_pts_in),
    gradients(gradients_in, gradients_in+num_gradients_in),
    num_gradients_to_sample(num_gradients_in),
    union_of_points(BuildUnionOfPoints(points_to_sample, points_being_sampled, num_to_sample, num_being_sampled, dim)),
    gradcost(dim*num_derivatives),
    vf_state_list(vf_state_vector) {
  vf_state_list->reserve(vf_evaluator.num_mcmc());
  // evaluate derived quantities for the GP
  for (int i=0; i<vf_evaluator.num_mcmc();++i){
    vf_state_list->emplace_back(vf_evaluator.robust_knowledge_gradient_evaluator_list()->at(i), points_to_sample, points_being_sampled,
                                num_to_sample_in, num_being_sampled_in, num_pts_in, gradients_in, num_gradients_in,
                                configure_for_gradients, normal_rng_in);
  }
}

template <typename DomainType>
RobustKnowledgeGradientMCMCState<DomainType>::RobustKnowledgeGradientMCMCState(RobustKnowledgeGradientMCMCState&& OL_UNUSED(other)) = default;

template <typename DomainType>
void RobustKnowledgeGradientMCMCState<DomainType>::SetupState(const EvaluatorType& vf_evaluator,
                                                        double const * restrict points_to_sample) {
  if (unlikely(dim != vf_evaluator.dim())) {
    OL_THROW_EXCEPTION(InvalidValueException<int>, "Evaluator's and State's dim do not match!", dim, vf_evaluator.dim());
  }

  // update quantities derived from points_to_sample
  SetCurrentPoint(vf_evaluator, points_to_sample);
}

template struct RobustKnowledgeGradientMCMCState<TensorProductDomain>;
template struct RobustKnowledgeGradientMCMCState<SimplexIntersectTensorProductDomain>;

/*!\rst
  This is a simple wrapper around ComputeKGOptimalPointsToSampleWithRandomStarts() and
  ComputeKGOptimalPointsToSampleViaLatinHypercubeSearch(). That is, this method attempts multistart gradient descent
  and falls back to latin hypercube search if gradient descent fails (or is not desired).
\endrst*/
template <typename DomainType>
void ComputeRKGMCMCOptimalPointsToSample(GaussianProcessMCMC& gaussian_process_mcmc, const int num_fidelity,
                                        const GradientDescentParameters& optimizer_parameters,
                                        const GradientDescentParameters& optimizer_parameters_inner,
                                        const DomainType& domain, const DomainType& inner_domain, const ThreadSchedule& thread_schedule,
                                        double const * restrict points_being_sampled,
                                        double const * discrete_pts,
                                        int num_to_sample, int num_being_sampled,
                                        int num_pts, double const * best_so_far, const double factor,
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
    ComputeRKGMCMCOptimalPointsToSampleWithRandomStarts(gaussian_process_mcmc, num_fidelity, optimizer_parameters, optimizer_parameters_inner,
                                                       domain, inner_domain, thread_schedule, points_being_sampled, discrete_pts,
                                                       num_to_sample, num_being_sampled, num_pts,
                                                       best_so_far, factor, max_int_steps,
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
      ComputeRKGMCMCOptimalPointsToSampleViaLatinHypercubeSearch(gaussian_process_mcmc, num_fidelity, optimizer_parameters_inner, domain, inner_domain,
                                                                thread_schedule_naive_search,
                                                                points_being_sampled, discrete_pts,
                                                                num_lhc_samples, num_to_sample,
                                                                num_being_sampled, num_pts, best_so_far, factor, max_int_steps,
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
template void ComputeRKGMCMCOptimalPointsToSample(
    GaussianProcessMCMC& gaussian_process_mcmc, const int num_fidelity, const GradientDescentParameters& optimizer_parameters,
    const GradientDescentParameters& optimizer_parameters_inner,
    const TensorProductDomain& domain, const TensorProductDomain& inner_domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled, double const * discrete_pts,
    int num_to_sample, int num_being_sampled,
    int num_pts, double const * best_so_far, const double factor, int max_int_steps, bool lhc_search_only,
    int num_lhc_samples, bool * restrict found_flag, UniformRandomGenerator * uniform_generator,
    NormalRNG * normal_rng, double * restrict best_points_to_sample);
template void ComputeRKGMCMCOptimalPointsToSample(
    GaussianProcessMCMC& gaussian_process_mcmc, const int num_fidelity, const GradientDescentParameters& optimizer_parameters,
    const GradientDescentParameters& optimizer_parameters_inner,
    const SimplexIntersectTensorProductDomain& domain, const SimplexIntersectTensorProductDomain& inner_domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled, double const * discrete_pts,
    int num_to_sample, int num_being_sampled,
    int num_pts, double const * best_so_far, const double factor, int max_int_steps, bool lhc_search_only, int num_lhc_samples, bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator, NormalRNG * normal_rng, double * restrict best_points_to_sample);
}  // end namespace optimal_learning