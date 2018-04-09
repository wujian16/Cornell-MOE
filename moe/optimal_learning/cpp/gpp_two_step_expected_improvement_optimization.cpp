/*!
  \file gpp_two_step_expected_improvement_optimization.cpp
  \rst
\endrst*/


#include "gpp_two_step_expected_improvement_optimization.hpp"

#include <cmath>

#include <algorithm>
#include <memory>
#include <vector>

#include <boost/math/distributions/normal.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_optimizer_parameters.hpp"

namespace optimal_learning {

template <typename DomainType>
TwoStepExpectedImprovementEvaluator<DomainType>::TwoStepExpectedImprovementEvaluator(const GaussianProcess& gaussian_process_in,
                                                                                     const int num_fidelity,
                                                                                     double const * discrete_pts,
                                                                                     int num_pts,
                                                                                     int num_mc_iterations,
                                                                                     const DomainType& domain,
                                                                                     const GradientDescentParameters& optimizer_parameters,
                                                                                     double best_so_far)
  : dim_(gaussian_process_in.dim()),
    num_fidelity_(num_fidelity),
    num_mc_iterations_(num_mc_iterations),
    best_so_far_(best_so_far),
    optimizer_parameters_(optimizer_parameters.num_multistarts, optimizer_parameters.max_num_steps,
                          optimizer_parameters.max_num_restarts, optimizer_parameters.num_steps_averaged,
                          optimizer_parameters.gamma, optimizer_parameters.pre_mult,
                          optimizer_parameters.max_relative_change, optimizer_parameters.tolerance),
    domain_(domain),
    normal_(0.0, 1.0),
    gaussian_process_(&gaussian_process_in),
    discrete_pts_(discrete_points(discrete_pts, num_pts)),
    num_pts_(num_pts){
}

template <typename DomainType>
TwoStepExpectedImprovementEvaluator<DomainType>::TwoStepExpectedImprovementEvaluator(TwoStepExpectedImprovementEvaluator&& other)
  : dim_(other.dim()),
    num_fidelity_(other.num_fidelity()),
    num_mc_iterations_(other.num_mc_iterations()),
    best_so_far_(other.best_so_far()),
    optimizer_parameters_(other.gradient_descent_params().num_multistarts, other.gradient_descent_params().max_num_steps,
                          other.gradient_descent_params().max_num_restarts, other.gradient_descent_params().num_steps_averaged,
                          other.gradient_descent_params().gamma, other.gradient_descent_params().pre_mult,
                          other.gradient_descent_params().max_relative_change, other.gradient_descent_params().tolerance),
    domain_(other.domain()),
    normal_(0.0, 1.0),
    gaussian_process_(other.gaussian_process()),
    discrete_pts_(other.discrete_pts_copy()),
    num_pts_(other.number_discrete_pts()){
}

/*!\rst
  Compute the two-step value function.
  This version requires the discretization of A (the feasibe domain).
  The discretization usually is: some set + points previous sampled + points being sampled + points to sample
\endrst*/
template <typename DomainType>
double TwoStepExpectedImprovementEvaluator<DomainType>::ComputeValueFunction(StateType * vf_state) const {
  const int num_union = vf_state->num_union;
  const int num_gradients_to_sample = vf_state->num_gradients_to_sample;

  double aggregate = 0.0;
  vf_state->normal_rng->ResetToMostRecentSeed();
  for (int i = 0; i < num_mc_iterations_; ++i) {
    if (i % 2 == 1){
      for (int j = 0; j < num_union*(1+num_gradients_to_sample); ++j) {
        vf_state->normals[j + i*num_union*(1+num_gradients_to_sample)] = -vf_state->normals[j + (i-1)*num_union*(1+num_gradients_to_sample)];
      }
    }
    else {
      for (int j = 0; j < num_union*(1+num_gradients_to_sample); ++j) {
        vf_state->normals[j + i*num_union*(1+num_gradients_to_sample)] = (*(vf_state->normal_rng))();
      }
    }

    double best_function_value = 0.0;
    bool found_flag;
    std::vector<double> make_up_function_value(num_union*(1+num_gradients_to_sample));
    gaussian_process_->ComputeMeanOfAdditionalPoints(vf_state->union_of_points.data(), num_union, vf_state->gradients.data(),
                                                     vf_state->num_gradients_to_sample, make_up_function_value.data());
    GeneralMatrixVectorMultiply(vf_state->cholesky_to_sample_var.data(), 'N', vf_state->normals.data() + i*num_union*(1+num_gradients_to_sample),
                                1.0, 1.0, num_union*(1+num_gradients_to_sample), num_union*(1+num_gradients_to_sample), num_union*(1+num_gradients_to_sample),
                                make_up_function_value.data());

    double improvement_this_step = 0.0;
    for (int j = 0; j < num_union; ++j) {
      double EI_step_one = best_so_far_ - make_up_function_value[j*(1+num_gradients_to_sample)];
      if (EI_step_one > improvement_this_step) {
        improvement_this_step = EI_step_one;
      }
    }

    GaussianProcess gaussian_process_after(*gaussian_process_);
    gaussian_process_after.AddPointsToGP(vf_state->union_of_points.data(), make_up_function_value.data(), num_union, false);

    ComputeOptimalOnePotentialSampleExpectedImprovement(gaussian_process_after, num_fidelity_, optimizer_parameters_,
                                                        best_so_far_-improvement_this_step, domain_,
                                                        vf_state->discretized_set.data(), num_union + num_pts_,
                                                        &found_flag, vf_state->best_point.data() + i*dim_, &best_function_value);

    aggregate += improvement_this_step + best_function_value;
  }
  return aggregate/static_cast<double>(num_mc_iterations_);
}

/*!\rst
  Computes gradient of Two-step Value function (see TwoStepExpectedImprovementEvaluator::ComputeGradValueFunction) wrt points_to_sample (stored in
  ``union_of_points[0:num_to_sample]``).
  Mechanism is similar to the computation of Two-step Value function, where points' contributions to the gradient are thrown out of their
  corresponding ``improvement <= 0.0``.
  Thus ``\nabla(\mu)`` only contributes when the ``winner`` (point w/best improvement this iteration) is the current point.
  That is, the gradient of ``\mu`` at ``x_i`` wrt ``x_j`` is 0 unless ``i == j`` (and only this result is stored in
  ``vf_state->grad_mu``).  The interaction with ``vf_state->grad_chol_decomp`` is harder to know a priori (like with
  ``grad_mu``) and has a more complex structure (rank 3 tensor), so the derivative wrt ``x_j`` is computed fully, and
  the relevant submatrix (indexed by the current ``winner``) is accessed each iteration.
  .. Note:: comments here are copied to _compute_grad_knowledge_gradient_monte_carlo() in python_version/knowledge_gradient.py
\endrst*/
template <typename DomainType>
double TwoStepExpectedImprovementEvaluator<DomainType>::ComputeGradValueFunction(StateType * vf_state, double * restrict grad_VF) const {
  const int num_union = vf_state->num_union;
  const int num_gradients_to_sample = vf_state->num_gradients_to_sample;

  std::vector<double> grad_mu_temp(dim_*(vf_state->num_to_sample)*(1+num_gradients_to_sample), 0.0);
  gaussian_process_->ComputeGradMeanOfPoints(vf_state->points_to_sample_state, grad_mu_temp.data());
  for (int i = 0; i < vf_state->num_to_sample; ++i){
    for (int d = 0; d < dim_; ++d){
      vf_state->grad_mu[d + i*dim_] = grad_mu_temp[d + i*(1+num_gradients_to_sample)*dim_];
    }
  }

  // compute the grad of chol among points to sample.
  gaussian_process_->ComputeGradCholeskyVarianceOfPoints(&(vf_state->points_to_sample_state),
                                                         vf_state->cholesky_to_sample_var.data(),
                                                         vf_state->grad_chol_decomp.data());

  std::fill(vf_state->aggregate.begin(), vf_state->aggregate.end(), 0.0);
  std::fill(vf_state->step_one_gradient.begin(), vf_state->step_one_gradient.end(), 0.0);
  std::fill(vf_state->best_point.begin(), vf_state->best_point.end(), 1.0);

  double aggregate = 0.0;
  vf_state->normal_rng->ResetToMostRecentSeed();
  for (int i = 0; i < num_mc_iterations_; ++i) {
    if (i % 2 == 1){
      for (int j = 0; j < num_union*(1+num_gradients_to_sample); ++j) {
        vf_state->normals[j + i*num_union*(1+num_gradients_to_sample)] = -vf_state->normals[j + (i-1)*num_union*(1+num_gradients_to_sample)];
      }
    }
    else {
      for (int j = 0; j < num_union*(1+num_gradients_to_sample); ++j) {
        vf_state->normals[j + i*num_union*(1+num_gradients_to_sample)] = (*(vf_state->normal_rng))();
      }
    }

    double best_function_value = 0.0;
    bool found_flag;
    std::vector<double> make_up_function_value(num_union*(1+num_gradients_to_sample));
    gaussian_process_->ComputeMeanOfAdditionalPoints(vf_state->union_of_points.data(), num_union, vf_state->gradients.data(),
                                                     vf_state->num_gradients_to_sample, make_up_function_value.data());
    GeneralMatrixVectorMultiply(vf_state->cholesky_to_sample_var.data(), 'N', vf_state->normals.data() + i*num_union*(1+num_gradients_to_sample),
                                1.0, 1.0, num_union*(1+num_gradients_to_sample), num_union*(1+num_gradients_to_sample), num_union*(1+num_gradients_to_sample),
                                make_up_function_value.data());

    double improvement_this_step = 0.0;
    int winner = num_union + 1;  // an out of-bounds initial value
    for (int j = 0; j < num_union; ++j) {
      double EI_step_one = best_so_far_ - make_up_function_value[j*(1+num_gradients_to_sample)];
      if (EI_step_one > improvement_this_step) {
        improvement_this_step = EI_step_one;
        winner = j;
      }
    }

    if (improvement_this_step > 0.0) {
      // improvement > 0.0 implies winner will be valid; i.e., in 0: vf_state->num_to_sample
      // recall that grad_mu only stores \frac{d mu_i}{d Xs_i}, since \frac{d mu_j}{d Xs_i} = 0 for i != j.
      // hence the only relevant term from grad_mu is the one describing the gradient wrt winner-th point,
      // and this term only arises if the winner (for most improvement) index is less than num_to_sample
      if (winner < vf_state->num_to_sample) {
        for (int k = 0; k < dim_; ++k) {
          vf_state->step_one_gradient[i*dim_*vf_state->num_to_sample + winner*dim_ + k] = vf_state->grad_mu[winner*dim_ + k];
        }
      }

      double const * restrict grad_chol_decomp_winner_block = vf_state->grad_chol_decomp.data() +
                                                              winner*dim_*(num_union)*Square((1+num_gradients_to_sample));
      for (int k = 0; k < vf_state->num_to_sample; ++k) {
        GeneralMatrixVectorMultiply(grad_chol_decomp_winner_block, 'N',
                                    vf_state->normals.data() + i*num_union*(1+num_gradients_to_sample), 1.0, 1.0,
                                    dim_, num_union*(1+num_gradients_to_sample), dim_,
                                    vf_state->step_one_gradient.data() + i*dim_*vf_state->num_to_sample + k*dim_);
        grad_chol_decomp_winner_block += dim_*Square(num_union*(1+num_gradients_to_sample));
      }

      for (int k = 0; k < vf_state->num_to_sample; ++k) {
        for (int d = 0; d < dim_; ++d) {
          vf_state->aggregate[d + k*dim_] -= vf_state->step_one_gradient[i*dim_*vf_state->num_to_sample + k*dim_ + d];
        }
      }
    } // end if: improvement_this_step > 0.0

    GaussianProcess gaussian_process_after(*gaussian_process_);
    gaussian_process_after.AddPointsToGP(vf_state->union_of_points.data(), make_up_function_value.data(), num_union, false);

    ComputeOptimalOnePotentialSampleExpectedImprovement(gaussian_process_after, num_fidelity_, optimizer_parameters_,
                                                        best_so_far_-improvement_this_step, domain_,
                                                        vf_state->discretized_set.data(), num_union + num_pts_,
                                                        &found_flag, vf_state->best_point.data() + i*dim_, &best_function_value);
    aggregate += improvement_this_step + best_function_value;

    // update the mean difference
    gaussian_process_after.ComputeMeanOfAdditionalPoints(vf_state->best_point.data() + i*dim_, 1, nullptr,
                                                         0, vf_state->best_mean_difference.data()+i);
    vf_state->best_mean_difference[i] = best_so_far_ - improvement_this_step - vf_state->best_mean_difference[i];

    // update the standard deviation
    PointsToSampleState best_point(gaussian_process_after, vf_state->best_point.data() + i*dim_, 1, nullptr, 0, 0);
    gaussian_process_after.ComputeVarianceOfPoints(&(best_point),
                                                   nullptr,
                                                   0,
                                                   vf_state->best_standard_deviation.data()+i);
    vf_state->best_standard_deviation[i] = std::fmax(kMinimumVarianceGradEI, vf_state->best_standard_deviation[i]);
    vf_state->best_standard_deviation[i] = sqrt(vf_state->best_standard_deviation[i]);
  }  // end for i: num_mc_iterations_

  double VF = aggregate/static_cast<double>(num_mc_iterations_);

  gaussian_process_->ComputeCovarianceOfPoints(&(vf_state->points_to_sample_state), vf_state->best_point.data(), num_mc_iterations_,
                                               nullptr, 0, false, nullptr, vf_state->chol_inverse_cov.data());
  TriangularMatrixMatrixSolve(vf_state->cholesky_to_sample_var.data(), 'N', num_union*(1+num_gradients_to_sample), num_mc_iterations_,
                              num_union*(1+num_gradients_to_sample), vf_state->chol_inverse_cov.data());

  gaussian_process_->ComputeGradInverseCholeskyCovarianceOfPoints(&(vf_state->points_to_sample_state),
                                                                  vf_state->cholesky_to_sample_var.data(),
                                                                  vf_state->grad_chol_decomp.data(),
                                                                  vf_state->chol_inverse_cov.data(),
                                                                  vf_state->best_point.data(),
                                                                  num_mc_iterations_, false, nullptr,
                                                                  vf_state->grad_chol_inverse_cov.data());

  // let L_{d,i,j,k} = grad_chol_decomp, d over dim_, i, j over num_union, k over num_to_sample
  // we want to compute: agg_dx_{d,k} = L_{d,i,j=winner,k} * normals_i
  // TODO(GH-92): Form this as one GeneralMatrixVectorMultiply() call by storing data as L_{d,i,k,j} if it's faster.
  double const * restrict grad_chol_decomp_winner_block = vf_state->grad_chol_inverse_cov.data();
  for (int k = 0; k < vf_state->num_to_sample; ++k) {
    for (int i = 0; i < num_mc_iterations_; ++i){
      // mean gradient
      std::vector<double> mean_grad(dim_);
      std::copy(vf_state->step_one_gradient.data() + i*dim_*vf_state->num_to_sample + k*dim_,
                vf_state->step_one_gradient.data() + i*dim_*vf_state->num_to_sample + (k+1)*dim_,
                mean_grad.data());
      GeneralMatrixVectorMultiply(grad_chol_decomp_winner_block, 'N', vf_state->normals.data() + i*num_union*(1+num_gradients_to_sample), -1.0, 1.0,
                                  dim_, num_union*(1+num_gradients_to_sample), dim_, mean_grad.data());

      // std gradient
      std::vector<double> std_grad(dim_);
      GeneralMatrixVectorMultiply(grad_chol_decomp_winner_block, 'N', vf_state->chol_inverse_cov.data() + i*num_union*(1+num_gradients_to_sample), 1.0, 0.0,
                                  dim_, num_union*(1+num_gradients_to_sample), dim_, std_grad.data());
      for (int d = 0; d<dim_; ++d){
        std_grad[d] /= -vf_state->best_standard_deviation[i];
      }

      // need change
      double to_sample_var = Square(vf_state->best_standard_deviation[i]);
      double mu_diff = vf_state->best_mean_difference[i];
      double C = mu_diff/vf_state->best_standard_deviation[i];
      double pdf_C = boost::math::pdf(normal_, C);
      double cdf_C = boost::math::cdf(normal_, C);

      for (int d = 0; d < dim_; ++d) {
        double d_C = (vf_state->best_standard_deviation[i]*mean_grad[d] - std_grad[d]*mu_diff)/to_sample_var;
        double d_A = mean_grad[d]*cdf_C + mu_diff*pdf_C*d_C;
        double d_B = std_grad[d]*pdf_C + vf_state->best_standard_deviation[i]*(-C)*pdf_C*d_C;

        vf_state->aggregate[d + k*dim_] += d_A + d_B;
      }
      grad_chol_decomp_winner_block += dim_*num_union*(1+num_gradients_to_sample);
    }
  }

  for (int k = 0; k < vf_state->num_to_sample*dim_; ++k) {
    grad_VF[k] = vf_state->aggregate[k]/static_cast<double>(num_mc_iterations_);
  }
  return VF;
}

template class TwoStepExpectedImprovementEvaluator<TensorProductDomain>;
template class TwoStepExpectedImprovementEvaluator<SimplexIntersectTensorProductDomain>;

template <typename DomainType>
void TwoStepExpectedImprovementState<DomainType>::SetCurrentPoint(const EvaluatorType& kg_evaluator,
                                                         double const * restrict points_to_sample) {
  // update points_to_sample in union_of_points
  std::copy(points_to_sample, points_to_sample + num_to_sample*dim, union_of_points.data());

  // evaluate derived quantities for the GP
  points_to_sample_state.SetupState(*kg_evaluator.gaussian_process(), union_of_points.data(),
                                    num_union, num_gradients_to_sample, num_derivatives, true, (num_derivatives>0));

  PreCompute(kg_evaluator, points_to_sample);
}

template <typename DomainType>
TwoStepExpectedImprovementState<DomainType>::TwoStepExpectedImprovementState(const EvaluatorType& kg_evaluator, double const * restrict points_to_sample,
                                                           double const * restrict points_being_sampled, int num_to_sample_in, int num_being_sampled_in, int num_pts_in,
                                                           int const * restrict gradients_in, int num_gradients_in, bool configure_for_gradients, NormalRNGInterface * normal_rng_in)
  : dim(kg_evaluator.dim()),
    num_to_sample(num_to_sample_in),
    num_being_sampled(num_being_sampled_in),
    num_derivatives(configure_for_gradients ? num_to_sample : 0),
    num_union(num_to_sample + num_being_sampled),
    num_iterations(kg_evaluator.num_mc_iterations()),
    gradients(gradients_in, gradients_in+num_gradients_in),
    num_gradients_to_sample(num_gradients_in),
    union_of_points(BuildUnionOfPoints(points_to_sample, points_being_sampled,
                                       num_to_sample, num_being_sampled, dim)),
    subset_union_of_points(SubsetData(union_of_points.data(), num_union, kg_evaluator.num_fidelity())),
    discretized_set(BuildUnionOfPoints(subset_union_of_points.data(), kg_evaluator.discrete_pts_copy().data(),
                                       num_union, kg_evaluator.number_discrete_pts(), dim - kg_evaluator.num_fidelity())),
    points_to_sample_state(*kg_evaluator.gaussian_process(), union_of_points.data(), num_union,
                           gradients_in, num_gradients_in, num_derivatives, true, configure_for_gradients),
    normal_rng(normal_rng_in),
    cholesky_to_sample_var(Square(num_union*(1+num_gradients_to_sample))),
    grad_chol_decomp(dim*Square(num_union*(1+num_gradients_to_sample))*num_derivatives),
    to_sample_mean_(num_union),
    grad_mu(dim*num_derivatives),
    aggregate(dim*num_derivatives),
    normals(num_union*(1+num_gradients_to_sample)*num_iterations),
    best_point(dim*num_iterations),
    chol_inverse_cov(num_iterations*num_union*(1+num_gradients_to_sample)),
    grad_chol_inverse_cov(dim*num_iterations*num_union*(1+num_gradients_to_sample)*num_derivatives),
    best_mean_difference(num_iterations),
    best_standard_deviation(num_iterations),
    step_one_gradient(dim*num_derivatives*num_iterations) {
  PreCompute(kg_evaluator, points_to_sample);
}

template <typename DomainType>
TwoStepExpectedImprovementState<DomainType>::TwoStepExpectedImprovementState(TwoStepExpectedImprovementState&& OL_UNUSED(other)) = default;

template <typename DomainType>
void TwoStepExpectedImprovementState<DomainType>::SetupState(const EvaluatorType& kg_evaluator,
                                                    double const * restrict points_to_sample) {
  if (unlikely(dim != kg_evaluator.dim())) {
    OL_THROW_EXCEPTION(InvalidValueException<int>, "Evaluator's and State's dim do not match!", dim, kg_evaluator.dim());
  }

  // update quantities derived from points_to_sample
  SetCurrentPoint(kg_evaluator, points_to_sample);
}

template <typename DomainType>
void TwoStepExpectedImprovementState<DomainType>::PreCompute(const EvaluatorType& kg_evaluator,
                                                             double const * restrict points_to_sample) {
  if (unlikely(dim != kg_evaluator.dim())) {
    OL_THROW_EXCEPTION(InvalidValueException<int>, "Evaluator's and State's dim do not match!", dim, kg_evaluator.dim());
  }

  kg_evaluator.gaussian_process()->ComputeMeanOfAdditionalPoints(union_of_points.data(), num_union, nullptr, 0,
                                                                 to_sample_mean_.data());

  kg_evaluator.gaussian_process()->ComputeVarianceOfPoints(&(points_to_sample_state), gradients.data(),
                                                           num_gradients_to_sample, cholesky_to_sample_var.data());
  //Adding the variance of measurement noise to the covariance matrix
  for (int i = 0;i < num_union; i++){
    for (int j = 0; j < 1+num_gradients_to_sample; ++j){
      int row = i*(1+num_gradients_to_sample)+j;
      cholesky_to_sample_var[row+row*num_union*(1+num_gradients_to_sample)] += kg_evaluator.gaussian_process()->noise_variance()[j];
    }
  }
  int leading_minor_index = ComputeCholeskyFactorL(num_union*(1+num_gradients_to_sample), cholesky_to_sample_var.data());
  if (unlikely(leading_minor_index != 0)) {
    OL_THROW_EXCEPTION(SingularMatrixException,
    "GP-Variance matrix singular. Check for duplicate points_to_sample/being_sampled or points_to_sample/being_sampled duplicating points_sampled with 0 noise.",
    cholesky_to_sample_var.data(), num_union*(1+num_gradients_to_sample), leading_minor_index);
  }
  ZeroUpperTriangle(num_union*(1+num_gradients_to_sample), cholesky_to_sample_var.data());
}

template struct TwoStepExpectedImprovementState<TensorProductDomain>;
template struct TwoStepExpectedImprovementState<SimplexIntersectTensorProductDomain>;
}  // end namespace optimal_learning
