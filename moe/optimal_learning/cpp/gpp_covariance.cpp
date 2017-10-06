/*!
  \file gpp_covariance.cpp
  \rst
  This file contains function definitions for the Covariance, GradCovariance,
  and HyperparameterGradCovariance member
  functions of CovarianceInterface subclasses.  It also contains a few utilities for computing common mathematical quantities
  and initialization.

  Gradient (spatial and hyperparameter) functions return all derivatives at once because there is substantial shared computation.
  The shared results are by far the most expensive part of gradient computations; they typically involve exponentiation and are
  further at least partially shared with the base covariance computation.

  TODO(GH-132): compute fcn, gradient, and hessian simultaneously for covariance (optionally skipping some terms).

  TODO(GH-129): Check expression simplification of gradients/hessians (esp the latter) for the various covariance functions.
  Current math was done by hand and maybe I missed something.
\endrst*/

#include "gpp_covariance.hpp"

#include <cmath>

#include <limits>
#include <vector>

#include "gpp_common.hpp"
#include "gpp_exception.hpp"

namespace optimal_learning {

namespace {
/*!\rst
  Computes ``\sum_{i=0}^{dim} (p1_i - p2_i) / W_i * (p1_i - p2_i)``,
  ``p1, p2 = point 1 & 2``; ``W = weight``.
  Equivalent to ``\|p1 - p2\|_2`` if all entries of W are 1.0.

  \param
    :point_one[size]: the vector p1
    :point_two[size]: the vector p2
    :weights[size]: the vector W, i.e., the scaling to apply to each term of the norm
    :size: number of dimensions in point
  \return
    the weighted ``L_2``-norm of the vector difference ``p1 - p2``.
\endrst*/
OL_PURE_FUNCTION OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT double
NormSquaredWithInverseWeights(double const * restrict point_one,
                              double const * restrict point_two,
                              double const * restrict weights, int size) noexcept {
  // calculates the one norm of two vectors (point_one and point_two of size size)
  double norm = 0.0;

  for (int i = 0; i < size; ++i) {
    norm += Square((point_one[i] - point_two[i]))/weights[i];
  }
  return norm;
}

/*!\rst
  Validate and initialize covariance function data (sizes, hyperparameters).

  \param
    :dim: the number of spatial dimensions
    :alpha: the hyperparameter \alpha, (e.g., signal variance, \sigma_f^2)
    :lengths_in: the input length scales, one per spatial dimension
    :lengths_sq[dim]: pointer to an array of at least dim double
  \output
    :lengths_sq[dim]: first dim entries overwritten with the square of the entries of lengths_in
\endrst*/
OL_NONNULL_POINTERS void InitializeCovariance(int dim, double alpha,
                                              const std::vector<double>& lengths_in,
                                              double * restrict lengths_sq) {
  // validate inputs
  if (dim < 0) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "Negative spatial dimension.", dim, 0);
  }

  if (static_cast<unsigned>(dim) != lengths_in.size()) {
    OL_THROW_EXCEPTION(InvalidValueException<int>, "dim (truth) and length vector size do not match.", lengths_in.size(), dim);
  }

  if (alpha <= 0.0) {
    OL_THROW_EXCEPTION(LowerBoundException<double>, "Invalid hyperparameter (alpha).", alpha, std::numeric_limits<double>::min());
  }

  // fill lengths_sq array
  for (int i = 0; i < dim; ++i) {
    lengths_sq[i] = Square(lengths_in[i]);
    if (unlikely(lengths_in[i] <= 0.0)) {
      OL_THROW_EXCEPTION(LowerBoundException<double>, "Invalid hyperparameter (length).", lengths_in[i],
                         std::numeric_limits<double>::min());
    }
  }
}

}  // end unnamed namespace

void SquareExponential::Initialize() {
  InitializeCovariance(dim_, alpha_, lengths_, lengths_sq_.data());
}

SquareExponential::SquareExponential(int dim, double alpha, std::vector<double> lengths)
    : dim_(dim), alpha_(alpha), lengths_(lengths), lengths_sq_(dim) {
  Initialize();
}

SquareExponential::SquareExponential(int dim, double alpha, double const * restrict lengths)
    : SquareExponential(dim, alpha, std::vector<double>(lengths, lengths + dim)) {
}

SquareExponential::SquareExponential(int dim, double alpha, double length)
    : SquareExponential(dim, alpha, std::vector<double>(dim, length)) {
}

SquareExponential::SquareExponential(const SquareExponential& OL_UNUSED(source)) = default;

/*
  Square Exponential: ``cov(x_1, x_2) = \alpha * \exp(-1/2 * ((x_1 - x_2)^T * L^{-1} * (x_1 - x_2)) )``
  plus the Jessian vector
  plus the Hessian matrix
*/
void SquareExponential::Covariance(double const * restrict point_one,
                                   int const * restrict derivatives_one,
                                   int num_derivatives_one,
                                   double const * restrict point_two,
                                   int const * restrict derivatives_two,
                                   int num_derivatives_two,
                                   double * restrict cov) const noexcept {
  cov[0] = 1.0;
  // the Jessian vector
  int index = 0;
  for (int i = 0; i < num_derivatives_one; ++i) {
    index = derivatives_one[i];
    cov[i+1] = ((point_two[index] - point_one[index])/lengths_sq_[index]);
  }
  for (int i = 0; i < num_derivatives_two; ++i) {
    index = derivatives_two[i];
    cov[(i+1)*(1+num_derivatives_one)] = ((point_one[index] - point_two[index])/lengths_sq_[index]);
  }

  // the Hessian matrix
  int index_one = 0;
  int index_two = 0;
  for (int i = 0; i < num_derivatives_one; ++i) {
    for (int j = 0; j < num_derivatives_two; ++j) {
      index_one = derivatives_one[i];
      index_two = derivatives_two[j];
      cov[(i+1)+(j+1)*(1+num_derivatives_one)] = cov[i+1]*cov[(j+1)*(1+num_derivatives_one)];
      if(index_one == index_two){
        cov[(i+1)+(j+1)*(1+num_derivatives_one)] += 1.0/lengths_sq_[index_two];
      }
    }
  }

  // correct by the kernel value
  const double norm_val = NormSquaredWithInverseWeights(point_one, point_two, lengths_sq_.data(), dim_);
  const double kernel = alpha_*std::exp(-0.5*norm_val);
  for (int i = 0; i < num_derivatives_one+1; ++i) {
    for (int j = 0; j < num_derivatives_two+1; ++j) {
       cov[i+j*(1+num_derivatives_one)] *= kernel;
    }
  }
}

/*
  Gradient of Square Exponential (wrt ``x_1``):
  ``\pderiv{cov(x_1, x_2)}{x_{1,i}} = (x_{2,i} - x_{1,i}) / L_{i}^2 * cov(x_1, x_2)``
  the gradient of the above matrix Cov wrt to the first point x1.
*/
void SquareExponential::GradCovariance(double const * restrict point_one,
                                       int const * restrict derivatives_one,
                                       int num_derivatives_one,
                                       double const * restrict point_two,
                                       int const * restrict derivatives_two,
                                       int num_derivatives_two,
                                       double * restrict grad_cov) const noexcept {
  std::vector<double> cov((1+num_derivatives_one)*(1+num_derivatives_two));
  cov[0] = 1.0;

  // the Jessian vector
  int index = 0;
  for (int i = 0; i < num_derivatives_one; ++i) {
    index = derivatives_one[i];
    cov[i+1] = ((point_two[index] - point_one[index])/lengths_sq_[index]);
  }
  for (int i = 0; i < num_derivatives_two; ++i) {
    index = derivatives_two[i];
    cov[(i+1)*(1+num_derivatives_one)] = ((point_one[index] - point_two[index])/lengths_sq_[index]);
  }

  // the Hessian matrix
  int index_one = 0;
  int index_two = 0;
  for (int i = 0; i < num_derivatives_one; ++i) {
    for (int j = 0; j < num_derivatives_two; ++j) {
      index_one = derivatives_one[i];
      index_two = derivatives_two[j];
      cov[(i+1)+(j+1)*(1+num_derivatives_one)] = cov[i+1]*cov[(j+1)*(1+num_derivatives_one)];
      if(index_one == index_two){
        cov[(i+1)+(j+1)*(1+num_derivatives_one)] += 1.0/lengths_sq_[index_two];
      }
    }
  }

  int index1 = 0;
  int index2 = 0;
  for (int i = 0; i < dim_; ++i) {
  // ditance between point one and point two at the dim i
    const double distance_i = (point_two[i] - point_one[i])/lengths_sq_[i];
    grad_cov[i] = distance_i*cov[0];
    for (int m = 0; m < num_derivatives_one; ++m){
      index1 = derivatives_one[m];
      grad_cov[i + (m+1)*dim_] = distance_i * cov[m+1];
      if (i == index1){
        grad_cov[i + (m+1)*dim_] -= 1.0/lengths_sq_[index1];
      }
    }
    for (int n = 0; n < num_derivatives_two; ++n){
      index2 = derivatives_two[n];
      grad_cov[i + (n+1)*dim_*(num_derivatives_one+1)] = distance_i * cov[(n+1)*(1+num_derivatives_one)];
      if (i == index2){
        grad_cov[i + (n+1)*dim_*(num_derivatives_one+1)] += 1.0/lengths_sq_[index2];
      }
    }
    for (int m = 0; m < num_derivatives_one; ++m){
      index1 = derivatives_one[m];
      for (int n = 0; n < num_derivatives_two; ++n){
        index2 = derivatives_two[n];
        grad_cov[i+ (m+1)*dim_ + (n+1)*dim_*(num_derivatives_one+1)] = distance_i * cov[(m+1)+(n+1)*(1+num_derivatives_one)];
        if (index1 == i){
          grad_cov[i+ (m+1)*dim_ + (n+1)*dim_*(num_derivatives_one+1)] -= cov[(n+1)*(1+num_derivatives_one)]/lengths_sq_[index1];
        }
        if (index2 == i){
          grad_cov[i+ (m+1)*dim_ + (n+1)*dim_*(num_derivatives_one+1)] += cov[m+1]/lengths_sq_[index2];
        }
      }
    }

    const double norm_val = NormSquaredWithInverseWeights(point_one, point_two, lengths_sq_.data(), dim_);
    const double kernel = alpha_*std::exp(-0.5*norm_val);
    for (int m = 0; m < num_derivatives_one+1; ++m) {
      for (int n = 0; n < num_derivatives_two+1; ++n) {
        grad_cov[i+m*dim_+n*dim_*(1+num_derivatives_one)] *= kernel;
      }
    }
  }
}

/*
  Gradient of Square Exponential (wrt hyperparameters (``alpha, L``)):
  ``\pderiv{cov(x_1, x_2)}{\theta_0} = cov(x_1, x_2) / \theta_0``
  ``\pderiv{cov(x_1, x_2)}{\theta_0} = [(x_{1,i} - x_{2,i}) / L_i]^2 / L_i * cov(x_1, x_2)``
  Note: ``\theta_0 = \alpha`` and ``\theta_{1:d} = L_{0:d-1}``

  output:
  double: (dim+1) * (num_derivatives_+1) * (num_derivatives_+1)
*/
void SquareExponential::HyperparameterGradCovariance(double const * restrict point_one, int const * restrict derivatives_one, int num_derivatives_one,
                                                     double const * restrict point_two, int const * restrict derivatives_two, int num_derivatives_two,
                                                     double * restrict grad_hyperparameter_cov) const noexcept {
  double * cov_matrix = new double[(num_derivatives_one+1)*(num_derivatives_two+1)]();
  Covariance(point_one, derivatives_one, num_derivatives_one,
             point_two, derivatives_two, num_derivatives_two, cov_matrix);

  int index1 = 0;
  int index2 = 0;

  // deriv wrt alpha does not have the same form as the length terms, special case it
  grad_hyperparameter_cov[0] = cov_matrix[0]/alpha_;
  for (int i = 0; i < dim_; ++i) {
    grad_hyperparameter_cov[i+1] = cov_matrix[0]*Square((point_one[i] - point_two[i])/lengths_[i])/lengths_[i];
  }

  for (int m = 0; m < num_derivatives_one; ++m){
    grad_hyperparameter_cov[(m+1)*(dim_+1)] = cov_matrix[m+1]/alpha_;
    index1 = derivatives_one[m];
    for (int i = 0; i < dim_; ++i) {
      grad_hyperparameter_cov[i+1+(m+1)*(dim_+1)] = cov_matrix[m+1]*
                                                    Square((point_one[i] - point_two[i])/lengths_[i])/lengths_[i];
      if (index1 == i){
        grad_hyperparameter_cov[i+1+(m+1)*(dim_+1)] -= (2*cov_matrix[0]*(point_two[i]-point_one[i])/lengths_sq_[i])/lengths_[i];
      }
    }
  }

  for (int n = 0; n < num_derivatives_two; ++n){
    grad_hyperparameter_cov[(n+1)*(dim_+1)*(num_derivatives_one+1)] = cov_matrix[(n+1)*(num_derivatives_one+1)]/alpha_;
    index2 = derivatives_two[n];
    for (int i = 0; i < dim_; ++i){
      grad_hyperparameter_cov[i+1+(n+1)*(dim_+1)*(num_derivatives_one+1)] = cov_matrix[(n+1)*(num_derivatives_one+1)]*
                                                                         Square((point_one[i] - point_two[i])/lengths_[i])/lengths_[i];
      if (index2 == i){
        grad_hyperparameter_cov[i+1+(n+1)*(dim_+1)*(num_derivatives_one+1)] -= (2*cov_matrix[0]*(point_one[i]-point_two[i])/lengths_sq_[i])/lengths_[i];
      }
    }
  }

  for (int m = 0; m < num_derivatives_one; ++m){
    index1 = derivatives_one[m];
    for (int n = 0; n < num_derivatives_two; ++n){
      index2 = derivatives_two[n];
      grad_hyperparameter_cov[(m+1)*(dim_+1)+(n+1)*(dim_+1)*(num_derivatives_one+1)] = cov_matrix[m+1+(n+1)*(num_derivatives_one+1)]/alpha_;
      for (int i = 0; i < dim_; ++i){
        grad_hyperparameter_cov[i+1+(m+1)*(dim_+1)+(n+1)*(dim_+1)*(num_derivatives_one+1)] = cov_matrix[m+1+(n+1)*(num_derivatives_one+1)]*
                                                                                    Square((point_one[i] - point_two[i])/lengths_[i])/lengths_[i];
        if (index1 == index2){
          if (index1 == i){
            grad_hyperparameter_cov[i+1+(m+1)*(dim_+1)+(n+1)*(dim_+1)*(num_derivatives_one+1)] += ((4*cov_matrix[0]*Square(point_one[i]-point_two[i])/
                                                                                         lengths_sq_[i])/lengths_sq_[i])/lengths_[i];
            grad_hyperparameter_cov[i+1+(m+1)*(dim_+1)+(n+1)*(dim_+1)*(num_derivatives_one+1)] -= (2*cov_matrix[0]/lengths_sq_[i])/lengths_[i];
          }
        }
        else{
          if (index1 == i){
            grad_hyperparameter_cov[i+1+(m+1)*(dim_+1)+(n+1)*(dim_+1)*(num_derivatives_one+1)] -= (2*cov_matrix[(n+1)*(num_derivatives_one+1)]*
                                                                                         (point_two[i]-point_one[i])/lengths_sq_[i])/lengths_[i];
          }
          if (index2 == i){
            grad_hyperparameter_cov[i+1+(m+1)*(dim_+1)+(n+1)*(dim_+1)*(num_derivatives_one+1)] -= (2*cov_matrix[m+1]*
                                                                                         (point_one[i]-point_two[i])/lengths_sq_[i])/lengths_[i];
          }
        }
      }
    }
  }

  delete [] cov_matrix;
}

CovarianceInterface * SquareExponential::Clone() const {
  return new SquareExponential(*this);
}

//void MaternNu2p5::Initialize() {
//  InitializeCovariance(dim_, alpha_, lengths_, lengths_sq_.data());
//}
//
//MaternNu2p5::MaternNu2p5(int dim, double alpha, std::vector<double> lengths)
//    : dim_(dim), alpha_(alpha), lengths_(lengths), lengths_sq_(dim) {
//  Initialize();
//}
//
//MaternNu2p5::MaternNu2p5(int dim, double alpha, double const * restrict lengths)
//    : MaternNu2p5(dim, alpha, std::vector<double>(lengths, lengths + dim)) {
//}
//
//MaternNu2p5::MaternNu2p5(int dim, double alpha, double length)
//    : MaternNu2p5(dim, alpha, std::vector<double>(dim, length)) {
//}
//
//MaternNu2p5::MaternNu2p5(const MaternNu2p5& OL_UNUSED(source)) = default;
//
//double MaternNu2p5::Covariance(double const * restrict point_one,
//                               double const * restrict point_two) const noexcept {
//  const double norm_val = NormSquaredWithInverseWeights(point_one, point_two, lengths_sq_.data(), dim_);
//  const double matern_arg = kSqrt5 * std::sqrt(norm_val);
//
//  return alpha_*(1.0 + matern_arg + 5.0/3.0*norm_val)*std::exp(-matern_arg);
//}
//
//void MaternNu2p5::GradCovariance(double const * restrict point_one,
//                                 double const * restrict point_two,
//                                 double * restrict grad_cov) const noexcept {
//  const double norm_val = NormSquaredWithInverseWeights(point_one, point_two, lengths_sq_.data(), dim_);
//  if (norm_val == 0.0) {
//    std::fill(grad_cov, grad_cov + dim_, 0.0);
//    return;
//  }
//  const double matern_arg = kSqrt5 * std::sqrt(norm_val);
//  const double poly_part = matern_arg + 5.0/3.0*norm_val;
//  const double exp_part = std::exp(-matern_arg);
//
//  for (int i = 0; i < dim_; ++i) {
//    const double dr2_dxi = 2.0*(point_one[i] - point_two[i])/lengths_sq_[i];
//    const double dr_dxi = 0.5*dr2_dxi/std::sqrt(norm_val);
//    grad_cov[i] = alpha_*exp_part*(5.0/3.0*dr2_dxi - poly_part*kSqrt5*dr_dxi);
//  }
//}
//
//void MaternNu2p5::HyperparameterGradCovariance(double const * restrict point_one,
//                                               double const * restrict point_two,
//                                               double * restrict grad_hyperparameter_cov) const noexcept {
//  const double norm_val = NormSquaredWithInverseWeights(point_one, point_two, lengths_sq_.data(), dim_);
//  if (norm_val == 0.0) {
//    grad_hyperparameter_cov[0] = 1.0;
//    std::fill(grad_hyperparameter_cov+1, grad_hyperparameter_cov + 1+dim_, 0.0);
//    return;
//  }
//  const double matern_arg = kSqrt5 * std::sqrt(norm_val);
//  const double poly_part = matern_arg + 5.0/3.0*norm_val;
//  const double exp_part = std::exp(-matern_arg);
//
//  // deriv wrt alpha does not have the same form as the length terms, special case it
//  grad_hyperparameter_cov[0] = (1.0 + poly_part) * exp_part;
//  // terms from differentiating Covariance wrt spatial dimensions; since exp(x) is the derivative's identity, some cancellation of
//  // analytic 0s is possible (and desired since it reduces compute-time and is more accurate)
//  for (int i = 0; i < dim_; ++i) {
//    const double dr2_dleni = -2.0*Square((point_one[i] - point_two[i])/lengths_[i])/lengths_[i];
//    const double dr_dleni = 0.5*dr2_dleni/std::sqrt(norm_val);
//    grad_hyperparameter_cov[i+1] = alpha_*exp_part*(5.0/3.0*dr2_dleni - poly_part*kSqrt5*dr_dleni);
//  }
//}
//
//CovarianceInterface * MaternNu2p5::Clone() const {
//  return new MaternNu2p5(*this);
//}

}  // end namespace optimal_learning