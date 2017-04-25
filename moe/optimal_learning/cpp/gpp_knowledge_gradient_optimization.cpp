/*!
  \file gpp_knowledge_gradient_optimization.cpp
  \rst
\endrst*/


#include "gpp_knowledge_gradient_optimization.hpp"
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

template <typename DomainType>
KnowledgeGradientEvaluator<DomainType>::KnowledgeGradientEvaluator(const GaussianProcess& gaussian_process_in,
                                                                   double const * discrete_pts,
                                                                   int num_pts,
                                                                   int num_mc_iterations,
                                                                   const DomainType& domain,
                                                                   const GradientDescentParameters& optimizer_parameters,
                                                                   double best_so_far)
    : dim_(gaussian_process_in.dim()),
      num_mc_iterations_(num_mc_iterations),
      best_so_far_(best_so_far),
      optimizer_parameters_(optimizer_parameters.num_multistarts, optimizer_parameters.max_num_steps,
                            optimizer_parameters.max_num_restarts, optimizer_parameters.num_steps_averaged,
                            optimizer_parameters.gamma, optimizer_parameters.pre_mult,
                            optimizer_parameters.max_relative_change, optimizer_parameters.tolerance),
      domain_(domain),
      gaussian_process_(&gaussian_process_in),
      discrete_pts_(discrete_points(discrete_pts, num_pts)),
      num_pts_(num_pts),
      to_sample_mean_(mean_value_discrete(discrete_pts, num_pts)),
      train_discrete_(train_sample(discrete_pts, num_pts, nullptr, 0)){
}

template <typename DomainType>
KnowledgeGradientEvaluator<DomainType>::KnowledgeGradientEvaluator(KnowledgeGradientEvaluator&& other)
    : dim_(other.dim()),
      num_mc_iterations_(other.num_mc_iterations()),
      best_so_far_(other.best_so_far()),
      optimizer_parameters_(other.gradient_descent_params().num_multistarts, other.gradient_descent_params().max_num_steps,
                            other.gradient_descent_params().max_num_restarts, other.gradient_descent_params().num_steps_averaged,
                            other.gradient_descent_params().gamma, other.gradient_descent_params().pre_mult,
                            other.gradient_descent_params().max_relative_change, other.gradient_descent_params().tolerance),
      domain_(other.domain()),
      gaussian_process_(other.gaussian_process()),
      discrete_pts_(other.discrete_pts_copy()),
      num_pts_(other.number_discrete_pts()),
      to_sample_mean_(other.mean_value_discrete_copy()),
      train_discrete_(other.train_discrete_points_copy()){
}

/*!\rst
  Compute Knowledge Gradient
  This version requires the discretization of A (the feasibe domain).
  The discretization usually is: some set + points previous sampled + points being sampled + points to sample
\endrst*/
template <typename DomainType>
double KnowledgeGradientEvaluator<DomainType>::ComputeKnowledgeGradient(StateType * kg_state) const {
    int num_union = kg_state->num_union;
    int num_gradients_to_sample = kg_state->num_gradients_to_sample;

    std::copy(to_sample_mean_.data(), to_sample_mean_.data() + num_pts_, kg_state->to_sample_mean_.data());
    gaussian_process_->ComputeMeanOfAdditionalPoints(kg_state->union_of_points.data(), num_union, nullptr, 0,
                                                     kg_state->to_sample_mean_.data()+num_pts_);

    gaussian_process_->ComputeVarianceOfPoints(&(kg_state->points_to_sample_state), kg_state->gradients.data(),
                                               num_gradients_to_sample, kg_state->cholesky_to_sample_var.data());

    // copy the variance of the num_union * num_union (the variance of the pints to sample + points being sampled)
    // to the computation of q-KG, doing this way includes the current points to the discretized A.
    for (int i = 0; i < num_union; i++){
        for (int j = 0; j < 1+num_gradients_to_sample; ++j){
            for (int col = 0; col < num_union; ++col){
                int row = i*(1+num_gradients_to_sample)+j;
                kg_state->inverse_cholesky_covariance[row + (num_pts_ + col) * num_union * (1+num_gradients_to_sample)] =
                kg_state->cholesky_to_sample_var[row + col*(1+num_gradients_to_sample) * num_union * (1+num_gradients_to_sample)];
            }
        }
    }

    //Adding the variance of measurement noise to the covariance matrix
    for (int i = 0;i < num_union; i++){
        for (int j = 0; j < 1+num_gradients_to_sample; ++j){
            int row = i*(1+num_gradients_to_sample)+j;
            kg_state->cholesky_to_sample_var[row+row*num_union*(1+num_gradients_to_sample)] += gaussian_process_->noise_variance()[j] + 1e-8;
        }
    }

    int leading_minor_index = ComputeCholeskyFactorL(num_union*(1+num_gradients_to_sample), kg_state->cholesky_to_sample_var.data());
    if (unlikely(leading_minor_index != 0)) {
        OL_THROW_EXCEPTION(SingularMatrixException,
        "GP-Variance matrix singular. Check for duplicate points_to_sample/being_sampled or points_to_sample/being_sampled duplicating points_sampled with 0 noise.",
        kg_state->cholesky_to_sample_var.data(), num_union*(1+num_gradients_to_sample), leading_minor_index);
    }
    ZeroUpperTriangle(num_union*(1+num_gradients_to_sample), kg_state->cholesky_to_sample_var.data());

    // matrix: (p+q) * N
    gaussian_process_->ComputeCovarianceOfPoints(&(kg_state->points_to_sample_state), discrete_pts_.data(), num_pts_,
                                                 nullptr, 0, true, train_discrete_.data(), kg_state->inverse_cholesky_covariance.data());

    TriangularMatrixMatrixSolve(kg_state->cholesky_to_sample_var.data(), 'N',
                                num_union*(1+num_gradients_to_sample), num_pts_+num_union, num_union*(1+num_gradients_to_sample),
                                kg_state->inverse_cholesky_covariance.data());

    double best_posterior = best_so_far_;
    for (int j = 0; j < num_union; ++j){
        if (kg_state->to_sample_mean_[num_pts_+j] < best_posterior){
            best_posterior = kg_state->to_sample_mean_[num_pts_+j];
        }
    }

    double aggregate = 0.0;
    for (int i = 0; i < num_mc_iterations_; ++i) {
        double improvement_this_step = -INFINITY;
        for (int j = 0; j < num_union*(1+num_gradients_to_sample); ++j) {
            kg_state->normals[j] = (*(kg_state->normal_rng))();
        }
        // compute KG_this_step_from_far = cholesky * normals   as  KG = cholesky * normal
        // b/c normals currently held in KG_this_step_from_var

        GeneralMatrixVectorMultiply(kg_state->inverse_cholesky_covariance.data(), 'T',
                                    kg_state->normals.data(), 1.0, 0.0,
                                    num_union*(1+num_gradients_to_sample),
                                    num_pts_+num_union,
                                    num_union*(1+num_gradients_to_sample),
                                    kg_state->KG_this_step_from_var.data());

        int winner_so_far = -1;
        for (int j = 0; j < num_pts_+num_union; ++j) {
          double KG_total = best_posterior - (kg_state->to_sample_mean_[j] + kg_state->KG_this_step_from_var[j]);
          if (KG_total > improvement_this_step) {
              improvement_this_step = KG_total;
              winner_so_far = j;
          }
        }

        double best_function_value = 0.0;
        if (winner_so_far < num_pts_) {
             ComputeOptimalFuturePosteriorMean(*gaussian_process_, kg_state->normals.data(), kg_state->union_of_points.data(), num_union,
                                               kg_state->gradients.data(), num_gradients_to_sample, kg_state->cholesky_to_sample_var.data(),
                                               kg_state->points_to_sample_state.K_inv_times_K_star.data(), optimizer_parameters_, domain_,
                                               discrete_pts_.data() + winner_so_far*dim_,
                                               best_function_value, kg_state->best_point.data());
        }
        else{
             ComputeOptimalFuturePosteriorMean(*gaussian_process_, kg_state->normals.data(), kg_state->union_of_points.data(), num_union,
                                               kg_state->gradients.data(), num_gradients_to_sample, kg_state->cholesky_to_sample_var.data(),
                                               kg_state->points_to_sample_state.K_inv_times_K_star.data(), optimizer_parameters_, domain_,
                                               kg_state->union_of_points.data() + (winner_so_far-num_pts_)*dim_,
                                               best_function_value, kg_state->best_point.data());
        }
        if (best_function_value < best_posterior - improvement_this_step){
            aggregate += improvement_this_step;
        }
        else{
            aggregate += best_posterior + best_function_value;
        }
        /*
        double best_mean = 0.0;
        gaussian_process_->ComputeMeanOfAdditionalPoints(kg_state->best_point.data(), 1, nullptr, 0, &best_mean);
        gaussian_process_->ComputeCovarianceOfPoints(&(kg_state->points_to_sample_state), kg_state->best_point.data(), 1,
                                                     nullptr, 0, false, nullptr, kg_state->chol_inverse_cov.data());

        TriangularMatrixMatrixSolve(kg_state->cholesky_to_sample_var.data(), 'N', num_union*(1+num_gradients_to_sample), 1,
                                    num_union*(1+num_gradients_to_sample), kg_state->chol_inverse_cov.data());
        GeneralMatrixVectorMultiply(kg_state->chol_inverse_cov.data(), 'T',
                                    kg_state->normals.data(), 1.0, 1.0,
                                    num_union*(1+num_gradients_to_sample),
                                    1, num_union*(1+num_gradients_to_sample),
                                    &best_mean);

        aggregate += best_posterior - best_mean;
        //aggregate += improvement_this_step;
        */
    }
    return aggregate/static_cast<double>(num_mc_iterations_);
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
void KnowledgeGradientEvaluator<DomainType>::ComputeGradKnowledgeGradient(StateType * kg_state, double * restrict grad_KG) const {
    const int num_union = kg_state->num_union;
    int num_gradients_to_sample = kg_state->num_gradients_to_sample;

    std::copy(to_sample_mean_.data(), to_sample_mean_.data() + num_pts_, kg_state->to_sample_mean_.data());
    gaussian_process_->ComputeMeanOfAdditionalPoints(kg_state->union_of_points.data(), num_union, nullptr, 0,
                                                     kg_state->to_sample_mean_.data() + num_pts_);

    std::vector<double> grad_mu_temp(dim_*(kg_state->num_to_sample)*(1+num_gradients_to_sample), 0.0);
    gaussian_process_->ComputeGradMeanOfPoints(kg_state->points_to_sample_state, grad_mu_temp.data());
    for (int i = 0; i < kg_state->num_to_sample; ++i){
        for (int d = 0; d < dim_; ++d){
            kg_state->grad_mu[d + i*dim_] = grad_mu_temp[d + i*(1+num_gradients_to_sample)*dim_];
        }
    }
    gaussian_process_->ComputeVarianceOfPoints(&(kg_state->points_to_sample_state), kg_state->gradients.data(),
                                               kg_state->num_gradients_to_sample, kg_state->cholesky_to_sample_var.data());

    // copy the variance of the num_union * num_union (the variance of the pints to sample + points being sampled)
    // to the computation of q-KG, doing this way includes the current points to the discretized A.
    for (int i = 0; i < num_union; i++){
        for (int j = 0; j < 1+num_gradients_to_sample; ++j){
            for (int col = 0; col < num_union; ++col){
                int row = i*(1+num_gradients_to_sample)+j;
                kg_state->inverse_cholesky_covariance[row + (num_pts_ + col) * num_union * (1+num_gradients_to_sample)] =
                kg_state->cholesky_to_sample_var[row + col*(1+num_gradients_to_sample) * num_union * (1+num_gradients_to_sample)];
            }
        }
    }
    //Adding the variance of measurement noise to the covariance matrix
    for (int i = 0;i < num_union; i++){
        for (int j = 0; j < 1+num_gradients_to_sample; ++j){
            int row = i*(1+num_gradients_to_sample)+j;
            kg_state->cholesky_to_sample_var[row+row*num_union*(1+num_gradients_to_sample)] += gaussian_process_->noise_variance()[j] + 1e-8;
        }
    }

    // compute the D_q: the cholesky factor of the variance of the "points to sample" with noise.
    int leading_minor_index = ComputeCholeskyFactorL(num_union*(1+num_gradients_to_sample), kg_state->cholesky_to_sample_var.data());
    if (unlikely(leading_minor_index != 0)) {
        OL_THROW_EXCEPTION(SingularMatrixException,
        "GP-Variance matrix singular. Check for duplicate points_to_sample/being_sampled or points_to_sample/being_sampled duplicating points_sampled with 0 noise.",
        kg_state->cholesky_to_sample_var.data(), num_union*(1+num_gradients_to_sample), leading_minor_index);
    }

    ZeroUpperTriangle(num_union*(1+num_gradients_to_sample), kg_state->cholesky_to_sample_var.data());

    gaussian_process_->ComputeCovarianceOfPoints(&(kg_state->points_to_sample_state),
                                                 discrete_pts_.data(), num_pts_, nullptr, 0, true, train_discrete_.data(),
                                                 kg_state->inverse_cholesky_covariance.data());

    gaussian_process_->ComputeGradInverseCholeskyVarianceOfPoints(&(kg_state->points_to_sample_state),
                                                                  kg_state->cholesky_to_sample_var.data(),
                                                                  kg_state->inverse_cholesky_covariance.data() + num_pts_ * (1+num_gradients_to_sample) * num_union,
                                                                  nullptr, nullptr, 0, true, nullptr,
                                                                  kg_state->grad_inverse_cholesky_covariance.data());

    // compute the grad of chol among points to sample.
    gaussian_process_->ComputeGradCholeskyVarianceOfPoints(&(kg_state->points_to_sample_state),
                                                           kg_state->cholesky_to_sample_var.data(),
                                                           kg_state->grad_chol_decomp.data());

    TriangularMatrixMatrixSolve(kg_state->cholesky_to_sample_var.data(), 'N',
                                num_union*(1+num_gradients_to_sample), num_pts_+num_union, num_union*(1+num_gradients_to_sample),
                                kg_state->inverse_cholesky_covariance.data());

    int winner_so_far = -1;
    double best_posterior = best_so_far_;
    for (int j = 0; j < num_union; ++j){
        if (kg_state->to_sample_mean_[num_pts_+j]<best_posterior){
            winner_so_far = j;
            best_posterior = kg_state->to_sample_mean_[num_pts_+j];
        }
    }

    std::fill(kg_state->aggregate.begin(), kg_state->aggregate.end(), 0.0);

    if (winner_so_far >= 0 && winner_so_far < kg_state->num_to_sample){
        for (int k = 0; k < dim_; ++k) {
            kg_state->aggregate[winner_so_far*dim_ + k] += num_mc_iterations_ * kg_state->grad_mu[winner_so_far*dim_ + k];
        }
    }

    for (int i = 0; i < num_mc_iterations_; ++i) {
        //double *norm = new double[num_union*(1+num_gradients_to_sample)]();
        for (int j = 0; j < num_union*(1+num_gradients_to_sample); ++j) {
            //norm[j] = (*(kg_state->normal_rng))();
            kg_state->normals[j] = (*(kg_state->normal_rng))();//norm[j];
        }
        // compute KG_this_step_from_far = cholesky * normals   as  KG = inverse_cholesky_covariance^T * normal
        GeneralMatrixVectorMultiply(kg_state->inverse_cholesky_covariance.data(), 'T',
                                    kg_state->normals.data(),
                                    1.0, 0.0, num_union*(1+num_gradients_to_sample),
                                    num_pts_+num_union,
                                    num_union*(1+num_gradients_to_sample),
                                    kg_state->KG_this_step_from_var.data());

        //delete[] norm;
        double improvement_this_step = -INFINITY;
        int winner = num_pts_ + num_union + 1;  // an out of-bounds initial value
        for (int j = 0; j < num_pts_+num_union; ++j) {
            double KG_total = best_posterior - (kg_state->to_sample_mean_[j] + kg_state->KG_this_step_from_var[j]);
            if (KG_total > improvement_this_step) {
                improvement_this_step = KG_total;
                winner = j;
            }
        }
        //printf("iter %d, winner %d\n", i, winner);
        double best_function_value = 0.0;
        if (winner < num_pts_) {
             ComputeOptimalFuturePosteriorMean(*gaussian_process_, kg_state->normals.data(), kg_state->union_of_points.data(), num_union,
                                               kg_state->gradients.data(), num_gradients_to_sample, kg_state->cholesky_to_sample_var.data(),
                                               kg_state->points_to_sample_state.K_inv_times_K_star.data(), optimizer_parameters_, domain_,
                                               discrete_pts_.data() + winner*dim_,
                                               best_function_value, kg_state->best_point.data());
        }
        else{
             ComputeOptimalFuturePosteriorMean(*gaussian_process_, kg_state->normals.data(), kg_state->union_of_points.data(), num_union,
                                               kg_state->gradients.data(), num_gradients_to_sample, kg_state->cholesky_to_sample_var.data(),
                                               kg_state->points_to_sample_state.K_inv_times_K_star.data(), optimizer_parameters_, domain_,
                                               kg_state->union_of_points.data() + (winner-num_pts_)*dim_,
                                               best_function_value, kg_state->best_point.data());
        }
        //printf("the previous VOI %f, \n", improvement_this_step);
        //printf("the after VOI %f\n", best_posterior + best_function_value);
        if (best_posterior + best_function_value < improvement_this_step && winner >= num_pts_ && winner < num_pts_+kg_state->num_to_sample) {
            for (int k = 0; k < dim_; ++k) {
                kg_state->aggregate[(winner-num_pts_)*dim_ + k] -= kg_state->grad_mu[(winner-num_pts_)*dim_ + k];
            }
            double const * restrict grad_chol_decomp_winner_block = kg_state->grad_inverse_cholesky_covariance.data()
                                                                    + dim_*num_union*(1+num_gradients_to_sample)*(winner-num_pts_);
            // kg_state->grad_chol_decomp.data() + winner*dim_*num_union*(1+num_gradients_to_sample);
            for (int k = 0; k < kg_state->num_to_sample; ++k) {
                GeneralMatrixVectorMultiply(grad_chol_decomp_winner_block, 'N', kg_state->normals.data(), -1.0, 1.0,
                                            dim_, num_union*(1+num_gradients_to_sample), dim_, kg_state->aggregate.data() + k*dim_);
                grad_chol_decomp_winner_block += dim_*num_union*(1+num_gradients_to_sample)*num_union;
            }
        }
        else {
            if (best_posterior + best_function_value < improvement_this_step){
                if (winner < num_pts_) {
                    std::copy(discrete_pts_.data() + winner*dim_, discrete_pts_.data() + (winner+1)*dim_, kg_state->best_point.begin());
                }
                else{
                    std::copy(kg_state->union_of_points.data() + (winner-num_pts_)*dim_,
                              kg_state->union_of_points.data() + (winner+1-num_pts_)*dim_,
                              kg_state->best_point.begin());
                }
            }
            //double best_mean = 0.0;
            //gaussian_process_->ComputeMeanOfAdditionalPoints(kg_state->best_point.data(), 1, nullptr, 0, &best_mean);

            gaussian_process_->ComputeCovarianceOfPoints(&(kg_state->points_to_sample_state), kg_state->best_point.data(), 1,
                                                         nullptr, 0, false, nullptr, kg_state->chol_inverse_cov.data());

            gaussian_process_->ComputeGradInverseCholeskyCovarianceOfPoints(&(kg_state->points_to_sample_state),
                                                                            kg_state->cholesky_to_sample_var.data(),
                                                                            kg_state->grad_chol_decomp.data(),
                                                                            kg_state->chol_inverse_cov.data(),
                                                                            kg_state->best_point.data(), 1, false, nullptr,
                                                                            kg_state->grad_chol_inverse_cov.data());
            /*
            TriangularMatrixMatrixSolve(kg_state->cholesky_to_sample_var.data(), 'N', num_union*(1+num_gradients_to_sample), 1,
                                        num_union*(1+num_gradients_to_sample), kg_state->chol_inverse_cov.data());

            GeneralMatrixVectorMultiply(kg_state->chol_inverse_cov.data(), 'T',
                                        kg_state->normals.data(), 1.0, 1.0,
                                        num_union*(1+num_gradients_to_sample),
                                        1, num_union*(1+num_gradients_to_sample),
                                        &best_mean);
            printf("the after VOI %f\n", best_mean);
            */
            // int winner = 0;
            // let L_{d,i,j,k} = grad_chol_decomp, d over dim_, i, j over num_union, k over num_to_sample
            // we want to compute: agg_dx_{d,k} = L_{d,i,j=winner,k} * normals_i
            // TODO(GH-92): Form this as one GeneralMatrixVectorMultiply() call by storing data as L_{d,i,k,j} if it's faster.
            double const * restrict grad_chol_decomp_winner_block = kg_state->grad_chol_inverse_cov.data();
            // kg_state->grad_chol_decomp.data() + winner*dim_*num_union*(1+num_gradients_to_sample);
            for (int k = 0; k < kg_state->num_to_sample; ++k) {
                GeneralMatrixVectorMultiply(grad_chol_decomp_winner_block, 'N', kg_state->normals.data(), -1.0, 1.0,
                                            dim_, num_union*(1+num_gradients_to_sample), dim_, kg_state->aggregate.data() + k*dim_);
                grad_chol_decomp_winner_block += dim_*num_union*(1+num_gradients_to_sample);
            }
        }
    }  // end for i: num_mc_iterations_

    for (int k = 0; k < kg_state->num_to_sample*dim_; ++k) {
        grad_KG[k] = kg_state->aggregate[k]/static_cast<double>(num_mc_iterations_);
    }
}

template class KnowledgeGradientEvaluator<TensorProductDomain>;
template class KnowledgeGradientEvaluator<SimplexIntersectTensorProductDomain>;

template <typename DomainType>
void KnowledgeGradientState<DomainType>::SetCurrentPoint(const EvaluatorType& kg_evaluator,
                                                         double const * restrict points_to_sample) {
  // update points_to_sample in union_of_points
  std::copy(points_to_sample, points_to_sample + num_to_sample*dim, union_of_points.data());

  // evaluate derived quantities for the GP
  points_to_sample_state.SetupState(*kg_evaluator.gaussian_process(), union_of_points.data(),
                                    num_union, num_gradients_to_sample, num_derivatives);
}

template <typename DomainType>
KnowledgeGradientState<DomainType>::KnowledgeGradientState(const EvaluatorType& kg_evaluator, double const * restrict points_to_sample,
                                                           double const * restrict points_being_sampled, int num_to_sample_in, int num_being_sampled_in, int num_pts_in,
                                                           int const * restrict gradients_in, int num_gradients_in, bool configure_for_gradients, NormalRNGInterface * normal_rng_in)
    : dim(kg_evaluator.dim()),
      num_to_sample(num_to_sample_in),
      num_being_sampled(num_being_sampled_in),
      num_derivatives(configure_for_gradients ? num_to_sample : 0),
      num_union(num_to_sample + num_being_sampled),
      num_pts(num_pts_in),
      gradients(gradients_in, gradients_in+num_gradients_in),
      num_gradients_to_sample(num_gradients_in),
      union_of_points(BuildUnionOfPoints(points_to_sample, points_being_sampled, num_to_sample, num_being_sampled, dim)),
      points_to_sample_state(*kg_evaluator.gaussian_process(), union_of_points.data(), num_union, gradients_in, num_gradients_in, num_derivatives),
      normal_rng(normal_rng_in),
      cholesky_to_sample_var(Square(num_union*(1+num_gradients_to_sample))),
      inverse_cholesky_covariance(num_union*(1+num_gradients_to_sample)*(num_pts+num_union)),
      grad_chol_decomp(dim*Square(num_union*(1+num_gradients_to_sample))*num_derivatives),
      grad_inverse_cholesky_covariance(dim*num_union*(1+num_gradients_to_sample)*num_union*num_derivatives),
      to_sample_mean_(num_pts+num_union),
      grad_mu(dim*num_derivatives),
      KG_this_step_from_var(num_pts+num_union),
      aggregate(dim*num_derivatives),
      normals(num_union*(1+num_gradients_to_sample)),
      best_point(dim),
      chol_inverse_cov(num_union*(1+num_gradients_to_sample)),
      grad_chol_inverse_cov(dim*num_union*(1+num_gradients_to_sample)*num_derivatives) {
}

template <typename DomainType>
KnowledgeGradientState<DomainType>::KnowledgeGradientState(KnowledgeGradientState&& OL_UNUSED(other)) = default;

template <typename DomainType>
void KnowledgeGradientState<DomainType>::SetupState(const EvaluatorType& kg_evaluator,
                                                    double const * restrict points_to_sample) {
  if (unlikely(dim != kg_evaluator.dim())) {
    OL_THROW_EXCEPTION(InvalidValueException<int>, "Evaluator's and State's dim do not match!", dim, kg_evaluator.dim());
  }

  // update quantities derived from points_to_sample
  SetCurrentPoint(kg_evaluator, points_to_sample);
}

template struct KnowledgeGradientState<TensorProductDomain>;
template struct KnowledgeGradientState<SimplexIntersectTensorProductDomain>;

PosteriorMeanEvaluator::PosteriorMeanEvaluator(
    const GaussianProcess& gaussian_process_in)
    : dim_(gaussian_process_in.dim()),
      gaussian_process_(&gaussian_process_in) {
}

/*!\rst
  Uses analytic formulas to compute EI when ``num_to_sample = 1`` and ``num_being_sampled = 0`` (occurs only in 1,0-EI).
  In this case, the single-parameter (posterior) GP is just a Gaussian.  So the integral in EI (previously eval'd with MC)
  can be computed 'exactly' using high-accuracy routines for the pdf & cdf of a Gaussian random variable.
  See Ginsbourger, Le Riche, and Carraro.
\endrst*/
double PosteriorMeanEvaluator::ComputePosteriorMean(StateType * ps_state) const {
  double to_sample_mean;
  gaussian_process_->ComputeMeanOfPoints(ps_state->points_to_sample_state, &to_sample_mean);

  return -to_sample_mean;
}

/*!\rst
  Differentiates OnePotentialSampleExpectedImprovementEvaluator::ComputeExpectedImprovement wrt
  ``points_to_sample`` (which is just ONE point; i.e., 1,0-EI).
  Again, this uses analytic formulas in terms of the pdf & cdf of a Gaussian since the integral in EI (and grad EI)
  can be evaluated exactly for this low dimensional case.
  See Ginsbourger, Le Riche, and Carraro.
\endrst*/
void PosteriorMeanEvaluator::ComputeGradPosteriorMean(
    StateType * ps_state,
    double * restrict grad_PS) const {
  double * restrict grad_mu = ps_state->grad_mu.data();

  gaussian_process_->ComputeGradMeanOfPoints(ps_state->points_to_sample_state, grad_mu);

  for (int i = 0; i < dim_; ++i) {
    grad_PS[i] = -grad_mu[i];
  }
}

void PosteriorMeanState::SetCurrentPoint(const EvaluatorType& ps_evaluator,
                                         double const * restrict point_to_sample_in) {
  // update current point in union_of_points
  std::copy(point_to_sample_in, point_to_sample_in + dim, point_to_sample.data());

  // evaluate derived quantities
  points_to_sample_state.SetupState(*ps_evaluator.gaussian_process(), point_to_sample.data(),
                                    num_to_sample, 0, num_derivatives);
}

PosteriorMeanState::PosteriorMeanState(
    const EvaluatorType& ps_evaluator,
    double const * restrict point_to_sample_in,
    bool configure_for_gradients)
    : dim(ps_evaluator.dim()),
      num_derivatives(configure_for_gradients ? num_to_sample : 0),
      point_to_sample(point_to_sample_in, point_to_sample_in + dim),
      points_to_sample_state(*ps_evaluator.gaussian_process(), point_to_sample.data(),
                             num_to_sample, nullptr, 0, num_derivatives, false),
      grad_mu(dim*num_derivatives) {
}

PosteriorMeanState::PosteriorMeanState(PosteriorMeanState&& OL_UNUSED(other)) = default;

void PosteriorMeanState::SetupState(const EvaluatorType& ps_evaluator,
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
void ComputeOptimalPosteriorMean(const GaussianProcess& gaussian_process,
                                 const GradientDescentParameters& optimizer_parameters,
                                 const DomainType& domain, double const * restrict initial_guess,
                                 bool * restrict found_flag, double * restrict best_next_point) {
  if (unlikely(optimizer_parameters.max_num_restarts <= 0)) {
    return;
  }
  bool configure_for_gradients = true;
  OL_VERBOSE_PRINTF("Posterior Mean Optimization via %s:\n", OL_CURRENT_FUNCTION_NAME);

  // special analytic case when we are not using (or not accounting for) multiple, simultaneous experiments
  PosteriorMeanEvaluator ps_evaluator(gaussian_process);

  typename PosteriorMeanEvaluator::StateType ps_state(ps_evaluator, initial_guess, configure_for_gradients);

  GradientDescentOptimizer<PosteriorMeanEvaluator, DomainType> gd_opt;
  gd_opt.Optimize(ps_evaluator, optimizer_parameters, domain, &ps_state);
  ps_state.GetCurrentPoint(best_next_point);
}

// template explicit instantiation definitions, see gpp_common.hpp header comments, item 6
template void ComputeOptimalPosteriorMean(const GaussianProcess& gaussian_process,
                                          const GradientDescentParameters& optimizer_parameters,
                                          const TensorProductDomain& domain, double const * restrict initial_guess,
                                          bool * restrict found_flag, double * restrict best_next_point);
template void ComputeOptimalPosteriorMean(const GaussianProcess& gaussian_process,
                                                          const GradientDescentParameters& optimizer_parameters,
                                                          const SimplexIntersectTensorProductDomain& domain, double const * restrict initial_guess,
                                                          bool * restrict found_flag, double * restrict best_next_point);

/*!\rst
  Routes the KG computation through MultistartOptimizer + NullOptimizer to perform KG function evaluations at the list of input
  points, using the appropriate KG evaluator (e.g., monte carlo vs analytic) depending on inputs.
\endrst*/
/*
void EvaluateKGAtPointList(const GaussianProcess& gaussian_process, const GradientDescentParameters& optimizer_parameters_inner,
                           const DomainType& domain,
                           const ThreadSchedule& thread_schedule,
                           double const * restrict initial_guesses, double const * restrict points_being_sampled,
                           double const * discrete_pts, int num_multistarts, int num_to_sample,
                           int num_being_sampled, int num_pts, double best_so_far,
                           int max_int_steps, bool * restrict found_flag, NormalRNG * normal_rng,
                           double * restrict function_values, double * restrict best_next_point) {
  if (unlikely(num_multistarts <= 0)) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "num_multistarts must be > 1", num_multistarts, 1);
  }

  using DomainType = DummyDomain;
  DomainType dummy_domain;
  bool configure_for_gradients = false;

  KnowledgeGradientEvaluator kg_evaluator(gaussian_process, discrete_pts, num_pts, max_int_steps, best_so_far);

  int num_derivatives = kg_evaluator.gaussian_process()->num_derivatives();
  std::vector<int> derivatives(kg_evaluator.gaussian_process()->derivatives());

  std::vector<typename KnowledgeGradientEvaluator::StateType> kg_state_vector;
  SetupKnowledgeGradientState(kg_evaluator, initial_guesses, points_being_sampled,
                              num_to_sample, num_being_sampled, num_pts, derivatives.data(), num_derivatives,
                              thread_schedule.max_num_threads,
                              configure_for_gradients, normal_rng, &kg_state_vector);

  // init winner to be first point in set and 'force' its value to be -INFINITY; we cannot do worse than this
  OptimizationIOContainer io_container(kg_state_vector[0].GetProblemSize(), -INFINITY, initial_guesses);

  NullOptimizer<KnowledgeGradientEvaluator, DomainType> null_opt;
  typename NullOptimizer<KnowledgeGradientEvaluator, DomainType>::ParameterStruct null_parameters;
  MultistartOptimizer<NullOptimizer<KnowledgeGradientEvaluator, DomainType> > multistart_optimizer;
  multistart_optimizer.MultistartOptimize(null_opt, kg_evaluator, null_parameters,
                                          dummy_domain, thread_schedule, initial_guesses,
                                          num_multistarts, kg_state_vector.data(), function_values, &io_container);
  *found_flag = io_container.found_flag;
  std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
}
*/

/*!\rst
  This is a simple wrapper around ComputeKGOptimalPointsToSampleWithRandomStarts() and
  ComputeKGOptimalPointsToSampleViaLatinHypercubeSearch(). That is, this method attempts multistart gradient descent
  and falls back to latin hypercube search if gradient descent fails (or is not desired).
\endrst*/
template <typename DomainType>
void ComputeKGOptimalPointsToSample(const GaussianProcess& gaussian_process,
                                    const GradientDescentParameters& optimizer_parameters,
                                    const GradientDescentParameters& optimizer_parameters_inner,
                                    const DomainType& domain, const ThreadSchedule& thread_schedule,
                                    double const * restrict points_being_sampled,
                                    double const * discrete_pts,
                                    int num_to_sample, int num_being_sampled,
                                    int num_pts, double best_so_far,
                                    int max_int_steps, bool lhc_search_only,
                                    int num_lhc_samples, bool * restrict found_flag,
                                    UniformRandomGenerator * uniform_generator,
                                    NormalRNG * normal_rng, double * restrict best_points_to_sample) {
  if (unlikely(num_to_sample <= 0)) {
    return;
  }

  std::vector<double> next_points_to_sample(gaussian_process.dim()*num_to_sample);

  bool found_flag_local = false;
  if (lhc_search_only == false) {
    ComputeKGOptimalPointsToSampleWithRandomStarts(gaussian_process, optimizer_parameters, optimizer_parameters_inner,
                                                   domain, thread_schedule, points_being_sampled, discrete_pts,
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
      ComputeKGOptimalPointsToSampleViaLatinHypercubeSearch(gaussian_process, optimizer_parameters_inner, domain,
                                                            thread_schedule_naive_search,
                                                            points_being_sampled, discrete_pts,
                                                            num_lhc_samples, num_to_sample,
                                                            num_being_sampled, num_pts, best_so_far,
                                                            max_int_steps,
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
template void ComputeKGOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimizer_parameters,
    const GradientDescentParameters& optimizer_parameters_inner,
    const TensorProductDomain& domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled, double const * discrete_pts,
    int num_to_sample, int num_being_sampled,
    int num_pts, double best_so_far, int max_int_steps, bool lhc_search_only,
    int num_lhc_samples, bool * restrict found_flag, UniformRandomGenerator * uniform_generator,
    NormalRNG * normal_rng, double * restrict best_points_to_sample);
template void ComputeKGOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const GradientDescentParameters& optimizer_parameters,
    const GradientDescentParameters& optimizer_parameters_inner,
    const SimplexIntersectTensorProductDomain& domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled,double const * discrete_pts,
    int num_to_sample, int num_being_sampled,
    int num_pts, double best_so_far, int max_int_steps, bool lhc_search_only, int num_lhc_samples, bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator, NormalRNG * normal_rng, double * restrict best_points_to_sample);

}  // end namespace optimal_learning
