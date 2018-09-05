/*!
  \file gpp_knowledge_gradient_optimization.cpp
  \rst
\endrst*/


#include "gpp_knowledge_gradient_mcmc_optimization.hpp"

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
  for (int i=0; i<num_mcmc_; ++i){
    MaternNu2p5 sqexp(dim_, hypers[0], hypers+1);
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
  compute the cost.
\endrst*/
template <typename DomainType>
double KnowledgeGradientMCMCEvaluator<DomainType>::ComputeCost(StateType * kg_state) const {
  if (num_fidelity_ == 0){
    return 1.0;
  }
  else{
    double cost = 0.0;
    for (int i=0; i<kg_state->num_to_sample; ++i){
      double point_cost = 1.0;
      for (int j=dim_-num_fidelity_; j<dim_; ++j){
        point_cost *= kg_state->union_of_points[i*dim_ + j];
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
void KnowledgeGradientMCMCEvaluator<DomainType>::ComputeGradCost(StateType * kg_state, double * restrict grad_cost) const {
  std::fill(kg_state->gradcost.begin(), kg_state->gradcost.end(), 0.0);
  if (num_fidelity_ > 0){
    int index = -1;
    double cost = 0.0;
    for (int i=0; i<kg_state->num_to_sample; ++i){
      double point_cost = 1.0;
      for (int j=dim_-num_fidelity_; j<dim_; ++j){
        point_cost *= kg_state->union_of_points[i*dim_ + j];
      }
      if (cost < point_cost){
        cost = point_cost;
        index = i;
      }
    }
    for (int j=dim_-num_fidelity_; j<dim_; ++j){
      kg_state->gradcost[index*dim_ + j] = cost/kg_state->union_of_points[index*dim_ + j];
    }
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
  double cost = ComputeCost(kg_state);
  return kg_value/static_cast<double>(num_mcmc_hypers_*cost);
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
  double KG = 0.0;
  for (int i=0; i<num_mcmc_hypers_; ++i){
    std::vector<double> temp(kg_state->dim*kg_state->num_to_sample, 0.0);
    KG += (*knowledge_gradient_evaluator_lst)[i].ComputeGradKnowledgeGradient((*(kg_state->kg_state_list)).data()+i, temp.data());
    for (int k = 0; k < kg_state->num_to_sample*dim_; ++k) {
        grad_KG[k] += temp[k];
    }
  }
  KG /= static_cast<double>(num_mcmc_hypers_);
  // cost and the grad of the cost
  double cost = ComputeCost(kg_state);
  ComputeGradCost(kg_state, kg_state->gradcost.data());

  for (int k = 0; k < kg_state->num_to_sample*dim_; ++k) {
    grad_KG[k] = grad_KG[k]/static_cast<double>(num_mcmc_hypers_);
    grad_KG[k] = (grad_KG[k]*cost - KG*kg_state->gradcost[k])/Square(cost);
  }
}

template class KnowledgeGradientMCMCEvaluator<TensorProductDomain>;
template class KnowledgeGradientMCMCEvaluator<SimplexIntersectTensorProductDomain>;

template <typename DomainType>
void KnowledgeGradientMCMCState<DomainType>::SetCurrentPoint(const EvaluatorType& kg_evaluator,
                                                             double const * restrict points_to_sample_in) {
  // update current point in union_of_points
  std::copy(points_to_sample_in, points_to_sample_in + dim, union_of_points.data());

  // evaluate derived quantities for the GP
  for (int i=0; i<kg_evaluator.num_mcmc();++i){
    (kg_state_list->at(i)).SetCurrentPoint(kg_evaluator.knowledge_gradient_evaluator_list()->at(i), points_to_sample_in);
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
    gradcost(dim*num_derivatives),
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

//PosteriorMeanMCMCEvaluator::PosteriorMeanMCMCEvaluator(
//  const GaussianProcessMCMC& gaussian_process_mcmc,
//  std::vector<typename PosteriorMeanState<DomainType>::EvaluatorType> * evaluator_vector)
//  : dim_(gaussian_process_mcmc.dim()),
//    gaussian_process_mcmc_(&gaussian_process_mcmc),
//    num_mcmc_hypers_(gaussian_process_mcmc.num_mcmc()),
//    posterior_mean_evaluator_lst(evaluator_vector) {
//    posterior_mean_evaluator_lst->reserve(num_mcmc_hypers_);
//    for (int i=0; i<num_mcmc_hypers_; ++i){
//      posterior_mean_evaluator_lst->emplace_back(gaussian_process_mcmc_->gaussian_process_lst[i]);
//  }
//}
//
///*!\rst
//  Uses analytic formulas to compute EI when ``num_to_sample = 1`` and ``num_being_sampled = 0`` (occurs only in 1,0-EI).
//  In this case, the single-parameter (posterior) GP is just a Gaussian.  So the integral in EI (previously eval'd with MC)
//  can be computed 'exactly' using high-accuracy routines for the pdf & cdf of a Gaussian random variable.
//  See Ginsbourger, Le Riche, and Carraro.
//\endrst*/
//double PosteriorMeanMCMCEvaluator::ComputePosteriorMean(StateType * ps_state) const {
//  double ps_value = 0.0;
//  for (int i=0; i<num_mcmc_hypers_; ++i){
//    ps_value += (*posterior_mean_evaluator_lst)[i].ComputeObjectiveFunction((*(ps_state->ps_state_list)).data()+i);
//  }
//  return ps_value/static_cast<double>(num_mcmc_hypers_);
//}
//
///*!\rst
//  Differentiates OnePotentialSampleExpectedImprovementEvaluator::ComputeExpectedImprovement wrt
//  ``points_to_sample`` (which is just ONE point; i.e., 1,0-EI).
//  Again, this uses analytic formulas in terms of the pdf & cdf of a Gaussian since the integral in EI (and grad EI)
//  can be evaluated exactly for this low dimensional case.
//  See Ginsbourger, Le Riche, and Carraro.
//\endrst*/
//void PosteriorMeanMCMCEvaluator::ComputeGradPosteriorMean(
//    StateType * ps_state,
//    double * restrict grad_PS) const {
//  for (int i=0; i<num_mcmc_hypers_; ++i){
//    std::vector<double> temp(dim_-ps_state->num_fidelity, 0.0);
//    (*posterior_mean_evaluator_lst)[i].ComputeGradObjectiveFunction((*(ps_state->ps_state_list)).data()+i, temp.data());
//    for (int k = 0; k < ps_state->dim_-ps_state->num_fidelity; ++k) {
//        grad_PS[k] += temp[k];
//    }
//  }
//  for (int k = 0; k < dim_-ps_state->num_fidelity; ++k) {
//    grad_PS[k] = grad_PS[k]/static_cast<double>(num_mcmc_hypers_);
//  }
//}
//
//void PosteriorMeanMCMCState::SetCurrentPoint(const EvaluatorType& ps_evaluator,
//                                             double const * restrict point_to_sample_in) {
//  // update current point in union_of_points
//  std::copy(point_to_sample_in, point_to_sample_in + dim - num_fidelity, point_to_sample.data());
//  std::fill(point_to_sample.data() + dim - num_fidelity, point_to_sample.data() + dim, 1.0);
//}
//
//PosteriorMeanMCMCState::PosteriorMeanMCMCState(
//  const EvaluatorType& ps_evaluator,
//  const int num_fidelity_in,
//  double const * restrict point_to_sample_in,
//  bool configure_for_gradients,
//  std::vector<typename PosteriorMeanEvaluator<DomainType>::StateType> * ps_state_vector)
//  : dim(ps_evaluator.dim()),
//    num_fidelity(num_fidelity_in),
//    num_derivatives(configure_for_gradients ? num_to_sample : 0),
//    point_to_sample(BuildUnionOfPoints(point_to_sample_in)),
//    grad_mu(dim*num_derivatives),
//    ps_state_list(ps_state_vector) {
//  ps_state_list->reserve(ps_evaluator.num_mcmc());
//  // evaluate derived quantities for the GP
//  for (int i=0; i<ps_evaluator.num_mcmc();++i){
//    ps_state_list->emplace_back(ps_evaluator.posterior_mean_evaluator_lst()->at(i), num_fidelity,
//                                points_to_sample, configure_for_gradients);
//  }
//}
//
//PosteriorMeanMCMCState::PosteriorMeanMCMCState(PosteriorMeanMCMCState&& OL_UNUSED(other)) = default;

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
//template <typename DomainType>
//void ComputeOptimalMCMCPosteriorMean(GaussianProcessMCMC& gaussian_process_mcmc, const int num_fidelity,
//                                 const GradientDescentParameters& optimizer_parameters,
//                                 const DomainType& domain, double const * restrict initial_guess, const int num_starts,
//                                 bool * restrict found_flag, double * restrict best_next_point, double * best_function_value) {
//  if (unlikely(optimizer_parameters.max_num_restarts <= 0)) {
//    return;
//  }
//  bool configure_for_gradients = true;
//  OL_VERBOSE_PRINTF("Posterior Mean Optimization via %s:\n", OL_CURRENT_FUNCTION_NAME);
//
//  // special analytic case when we are not using (or not accounting for) multiple, simultaneous experiments
//  PosteriorMeanEvaluator ps_evaluator(gaussian_process);
//  typename PosteriorMeanEvaluator::StateType ps_state(ps_evaluator, num_fidelity, initial_guess, configure_for_gradients);
//
//  std::priority_queue<std::pair<double, int>> q;
//  double val;
//  int k = std::min(1, num_starts); // number of indices we need
//  for (int i = 0; i < num_starts; ++i) {
//    ps_state.SetCurrentPoint(ps_evaluator, initial_guess + i*(gaussian_process.dim()-num_fidelity));
//    val = ps_evaluator.ComputePosteriorMean(&ps_state);
//    if (i < k){
//      q.push(std::pair<double, int>(-val, i));
//    }
//    else{
//      if (q.top().first > -val){
//        q.pop();
//        q.push(std::pair<double, int>(-val, i));
//      }
//    }
//  }
//
//  std::vector<double> top_k_starting(k*(gaussian_process.dim()-num_fidelity));
//  for (int i = 0; i < k; ++i) {
//    int ki = q.top().second;
//    for (int d = 0; d<gaussian_process.dim()-num_fidelity; ++d){
//      top_k_starting[i*(gaussian_process.dim()-num_fidelity) + d] = initial_guess[ki*(gaussian_process.dim()-num_fidelity) + d];
//    }
//    q.pop();
//  }
//
//  GradientDescentOptimizerLineSearch<PosteriorMeanEvaluator, DomainType> gd_opt;
//  double function_value_temp = -INFINITY;
//  *best_function_value = -INFINITY;
//  for (int i = 0; i < k; ++i){
//    ps_state.SetCurrentPoint(ps_evaluator, top_k_starting.data() + i*(gaussian_process.dim()-num_fidelity));
//    gd_opt.Optimize(ps_evaluator, optimizer_parameters, domain, &ps_state);
//    function_value_temp = ps_evaluator.ComputePosteriorMean(&ps_state);
//    if (function_value_temp > *best_function_value){
//      *best_function_value = function_value_temp;
//      ps_state.GetCurrentPoint(best_next_point);
//    }
//  }
//}
//
//// template explicit instantiation definitions, see gpp_common.hpp header comments, item 6
//template void ComputeOptimalMCMCPosteriorMean(GaussianProcessMCMC& gaussian_process_mcmc, const int num_fidelity,
//                                          const GradientDescentParameters& optimizer_parameters,
//                                          const TensorProductDomain& domain, double const * restrict initial_guess, const int num_starts,
//                                          bool * restrict found_flag, double * restrict best_next_point, double * best_function_value);
//template void ComputeOptimalMCMCPosteriorMean(GaussianProcessMCMC& gaussian_process_mcmc, const int num_fidelity,
//                                          const GradientDescentParameters& optimizer_parameters,
//                                          const SimplexIntersectTensorProductDomain& domain, double const * restrict initial_guess, const int num_starts,
//                                          bool * restrict found_flag, double * restrict best_next_point, double * best_function_value);

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