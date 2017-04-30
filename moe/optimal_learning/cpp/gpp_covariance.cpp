/*!
  \file gpp_covariance.cpp
  \rst
  This file contains function definitions for the Covariance, GradCovariance, and HyperparameterGradCovariance member
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
#include "gpp_linear_algebra.hpp"

namespace optimal_learning {

namespace {

/*!\rst
  Computes ``\sum_{i=0}^{dim} (p1_i - p2_i) / W_i * (p1_i - p2_i)``, ``p1, p2 = point 1 & 2``; ``W = weight``.
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
NormSquaredWithInverseWeights(double const * restrict point_one, double const * restrict point_two,
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
OL_NONNULL_POINTERS void InitializeCovariance(int dim, double alpha, const std::vector<double>& lengths_in,
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
      OL_THROW_EXCEPTION(LowerBoundException<double>, "Invalid hyperparameter (length).", lengths_in[i], std::numeric_limits<double>::min());
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
void SquareExponential::Covariance(double const * restrict point_one, int const * restrict derivatives_one, int num_derivatives_one,
                                   double const * restrict point_two, int const * restrict derivatives_two, int num_derivatives_two,
                                   double * restrict cov) const noexcept {
    const double norm_val = NormSquaredWithInverseWeights(point_one, point_two, lengths_sq_.data(), dim_);

    cov[0] = alpha_*std::exp(-0.5*norm_val);
    int index = 0;
    int index1 = 0;
    int index2 = 0;

    for (int i = 0; i < num_derivatives_one; ++i) {
        index = derivatives_one[i];
        cov[i+1] = ((point_two[index] - point_one[index])/lengths_sq_[index]) * cov[0];
    }
    for (int i = 0; i < num_derivatives_two; ++i) {
        index = derivatives_two[i];
        cov[(i+1)*(1+num_derivatives_one)] = ((point_one[index] - point_two[index])/lengths_sq_[index]) * cov[0];
    }

    for (int i = 0; i < num_derivatives_one; ++i) {
        for (int j = 0; j < num_derivatives_two; ++j) {
            index1 = derivatives_one[i];
            index2 = derivatives_two[j];
            cov[(i+1)+(j+1)*(1+num_derivatives_one)] = ((point_two[index1] - point_one[index1])/lengths_sq_[index1]) *
                                                       ((point_one[index2] - point_two[index2])/lengths_sq_[index2]) * cov[0];
            if(index1 == index2){
                cov[(i+1)+(j+1)*(1+num_derivatives_one)] += cov[0]/lengths_sq_[index2];
            }
        }
    }
}

/*
  Gradient of Square Exponential (wrt ``x_1``):
  ``\pderiv{cov(x_1, x_2)}{x_{1,i}} = (x_{2,i} - x_{1,i}) / L_{i}^2 * cov(x_1, x_2)``
  the gradient of the above matrix Cov wrt to the first point x1.
*/
void SquareExponential::GradCovariance(double const * restrict point_one, int const * restrict derivatives_one, int num_derivatives_one,
                                       double const * restrict point_two, int const * restrict derivatives_two, int num_derivatives_two,
                                       double * restrict grad_cov) const noexcept {

  const double norm_val = NormSquaredWithInverseWeights(point_one, point_two, lengths_sq_.data(), dim_);
  const double cov = alpha_*std::exp(-0.5*norm_val);

  int index1 = 0;
  int index2 = 0;

  for (int i = 0; i < dim_; ++i) {
      grad_cov[i] = (point_two[i] - point_one[i])/lengths_sq_[i]*cov;
      for (int m = 0; m < num_derivatives_one; ++m){
          index1 = derivatives_one[m];
          grad_cov[i + (m+1)*dim_] = (point_two[i] - point_one[i])/lengths_sq_[i] *
                                     (point_two[index1] - point_one[index1])/lengths_sq_[index1]*cov;
          if (i == index1){
              grad_cov[i + (m+1)*dim_] -= cov/lengths_sq_[index1];
          }
      }
      for (int n =0; n < num_derivatives_two; ++n){
          index2 = derivatives_two[n];
          grad_cov[i + (n+1)*dim_*(num_derivatives_one+1)] = (point_two[i] - point_one[i])/lengths_sq_[i] *
                                                             (point_one[index2] - point_two[index2])/lengths_sq_[index2]*cov;
          if (i == index2){
              grad_cov[i + (n+1)*dim_*(num_derivatives_one+1)] += cov/lengths_sq_[index2];
          }
      }
      for (int m = 0; m < num_derivatives_one; ++m){
          index1 = derivatives_one[m];
          for (int n = 0; n < num_derivatives_two; ++n){
              index2 = derivatives_two[n];
              grad_cov[i+ (m+1)*dim_ + (n+1)*dim_*(num_derivatives_one+1)] = (point_two[index1] - point_one[index1])/lengths_sq_[index1] *
                                                                             (point_one[index2] - point_two[index2])/lengths_sq_[index2];
              if (index1 == index2){
                  grad_cov[i+ (m+1)*dim_ + (n+1)*dim_*(num_derivatives_one+1)] += 1/lengths_sq_[index1];
              }
              grad_cov[i+ (m+1)*dim_ + (n+1)*dim_*(num_derivatives_one+1)] *= (point_two[i] - point_one[i])/lengths_sq_[i];
              if (index1 == i){
                  grad_cov[i+ (m+1)*dim_ + (n+1)*dim_*(num_derivatives_one+1)] -= (point_one[index2] - point_two[index2])/
                                                                                  (lengths_sq_[index1] * lengths_sq_[index2]);
              }
              if (index2 == i){
                  grad_cov[i+ (m+1)*dim_ + (n+1)*dim_*(num_derivatives_one+1)] += (point_two[index1] - point_one[index1])/
                                                                                  (lengths_sq_[index1] * lengths_sq_[index2]);
              }
              grad_cov[i+ (m+1)*dim_ + (n+1)*dim_*(num_derivatives_one+1)] *= cov;
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

/*void SquareExponential::HyperparameterHessianCovariance(double const * restrict point_one, double const * restrict point_two,
                                                        double * restrict hessian_hyperparameter_cov) const noexcept {
  const double cov = Covariance(point_one, point_two);
  const int num_hyperparameters = GetNumberOfHyperparameters();
  std::vector<double> grad_hyper_cov(num_hyperparameters);
  std::vector<double> componentwise_scaled_distance(num_hyperparameters);
  HyperparameterGradCovariance(point_one, point_two, grad_hyper_cov.data());

  for (int i = 0; i < dim_; ++i) {
    // note the indexing is shifted by one here so that we can iterate over componentwise_scaled_distance with the
    // same index that is used for hyperparameter indexing
    componentwise_scaled_distance[i+1] = Square((point_one[i] - point_two[i])/lengths_[i])/lengths_[i];  // (x1_i - x2_i)^2/l_i^3
  }

  double const * restrict hessian_hyperparameter_cov_row = hessian_hyperparameter_cov + 1;  // used to index through rows
  // first column of hessian, derivatives of pderiv{cov}{\theta_0} with respect to \theta_i
  // \theta_0 is alpha, the scaling factor; its derivatives are fundamentally different in form than those wrt length scales;
  // hence they are split out of the loop
  hessian_hyperparameter_cov[0] = 0.0;
  for (int i = 1; i < num_hyperparameters; ++i) {
    // this is simply pderiv{cov}{\theta_i}/alpha, i = 1..dim+1
    hessian_hyperparameter_cov[i] = grad_hyper_cov[0]*componentwise_scaled_distance[i];
  }
  hessian_hyperparameter_cov += num_hyperparameters;

  // remaining columns of the hessian: derivatives with respect to pderiv{cov}{\theta_j} for j = 1..dim+1 (length scales)
  // terms come from straightforward differentiation of HyperparameterGradCovariance() expressions; no simplification needed
  for (int j = 1; j < num_hyperparameters; ++j) {
    // copy j-th column from j-th row, which has already been computed
    for (int i = 0; i < j; ++i) {
      hessian_hyperparameter_cov[i] = hessian_hyperparameter_cov_row[0];
      hessian_hyperparameter_cov_row += num_hyperparameters;
    }
    hessian_hyperparameter_cov_row -= j*num_hyperparameters;  // reset row for next iteration

    // on diagonal component has extra terms since normally dx_i/dx_j = 0 except for i == j
    // the RHS terms are only read from already-computed or copied components
    hessian_hyperparameter_cov[j] = cov*(Square(componentwise_scaled_distance[j]) - 3.0*componentwise_scaled_distance[j]/lengths_[j-1]);
    // remaining off-digaonal terms
    for (int i = j+1; i < num_hyperparameters; ++i) {
      hessian_hyperparameter_cov[i] = cov*componentwise_scaled_distance[i]*componentwise_scaled_distance[j];
    }

    hessian_hyperparameter_cov += num_hyperparameters;
    hessian_hyperparameter_cov_row += 1;
  }
}*/

CovarianceInterface * SquareExponential::Clone() const {
  return new SquareExponential(*this);
}

namespace {

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
OL_NONNULL_POINTERS void InitializeDeepKernel(int dim, double& alpha, const std::vector<double>& hypers_in,
                                              double * restrict w_0, double * restrict b_0,
                                              double * restrict w_1, double * restrict b_1,
                                              double * restrict w_2, double * restrict b_2,
                                              double * restrict w_3, double * restrict b_3,
                                              double * restrict lengths, double * restrict lengths_sq) {
  // validate inputs
  if (dim < 0) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "Negative spatial dimension.", dim, 0);
  }

  int num_hypers = dim + 1 +
                   50*dim +
                   dim + 50*50 + 50 +
                   50*50 + 50 +
                   50 + 1;

  if (static_cast<unsigned>(num_hypers) != hypers_in.size()) {
    OL_THROW_EXCEPTION(InvalidValueException<int>, "num_hypers (truth) and hypers vector size do not match.", hypers_in.size(), num_hypers);
  }

  std::copy(hypers_in.data(), hypers_in.data() + 50*dim, w_0);
  std::copy(hypers_in.data() + 50*dim, hypers_in.data() + 50*dim + dim, b_0);
  std::copy(hypers_in.data() + 50*dim + dim, hypers_in.data() + 50*dim + dim + 50*50, w_1);
  std::copy(hypers_in.data() + 50*dim + dim + 50*50, hypers_in.data() + 50*dim + dim + 50*50 + 50, b_1);
  std::copy(hypers_in.data() + 50*dim + dim + 50*50 + 50, hypers_in.data() + 50*dim + dim + 50*50 + 50 + 50*50, w_2);
  std::copy(hypers_in.data() + 50*dim + dim + 50*50 + 50 + 50*50, hypers_in.data() + 50*dim + dim + 50*50 + 50 + 50*50 + 50, b_2);
  std::copy(hypers_in.data() + 50*dim + dim + 50*50 + 50 + 50*50 + 50, hypers_in.data() + 50*dim + dim + 50*50 + 50 + 50*50 + 50 + 50, w_3);
  std::copy(hypers_in.data() + 50*dim + dim + 50*50 + 50 + 50*50 + 50 + 50, hypers_in.data() + 50*dim + dim + 50*50 + 50 + 50*50 + 50 + 50 + 1, b_3);
  alpha = hypers_in[50*dim + dim + 50*50 + 50 + 50*50 + 50 + 50 + 1];
  std::copy(hypers_in.data() + 50*dim + dim + 50*50 + 50 + 50*50 + 50 + 50 + 1 + 1,
            hypers_in.data() + 50*dim + dim + 50*50 + 50 + 50*50 + 50 + 50 + 1 + 1 + dim, lengths);

  if (alpha <= 0.0) {
    OL_THROW_EXCEPTION(LowerBoundException<double>, "Invalid hyperparameter (alpha).", alpha, std::numeric_limits<double>::min());
  }

  // fill lengths_sq array
  for (int i = 0; i < dim; ++i) {
    lengths_sq[i] = Square(lengths[i]);
    if (unlikely(lengths[i] <= 0.0)) {
      OL_THROW_EXCEPTION(LowerBoundException<double>, "Invalid hyperparameter (length).", lengths[i], std::numeric_limits<double>::min());
    }
  }
}

} // end unnamed namespace

void DeepKernel::Initialize(std::vector<double> hypers) {
  InitializeDeepKernel(dim_, alpha_, hypers,
                       w_0_.data(), b_0_.data(),
                       w_1_.data(), b_1_.data(),
                       w_2_.data(), b_2_.data(),
                       w_3_.data(), b_3_.data(),
                       lengths_.data(), lengths_sq_.data());
}

DeepKernel::DeepKernel(int dim, std::vector<double> hypers)
    : dim_(dim), alpha_(0.0), lengths_(dim), lengths_sq_(dim),
      w_0_(dim_*50), b_0_(dim_),
      w_1_(50*50), b_1_(50),
      w_2_(50*50), b_2_(50),
      w_3_(50), b_3_(1) {
  Initialize(hypers);
}

DeepKernel::DeepKernel(int dim, double const * restrict hypers)
    : DeepKernel(dim, std::vector<double>(hypers, hypers + dim + 1 + 50*dim_ + dim_ +
                                                         50*50 + 50 +
                                                         50*50 + 50 +
                                                         50 + 1)) {
}

DeepKernel::DeepKernel(const DeepKernel& OL_UNUSED(source)) = default;

void DeepKernel::NeuralNetwork(double const * restrict point_one, double * restrict projection) const noexcept {
  std::vector<double> layer1(b_0_);
  std::vector<double> layer2(b_1_);
  std::vector<double> layer3(b_2_);

  GeneralMatrixVectorMultiply(w_0_.data(), 'T', point_one, 1.0, 1.0, dim_, 50, dim_, layer1.data());
  for (int i=0; i<50; ++i){
    layer1[i] = tanh(layer1[i]);
  }

  GeneralMatrixVectorMultiply(w_1_.data(), 'T', layer1.data(), 1.0, 1.0, 50, 50, 50, layer2.data());
  for (int i=0; i<50; ++i){
    layer2[i] = tanh(layer2[i]);
  }

  GeneralMatrixVectorMultiply(w_2_.data(), 'T', layer2.data(), 1.0, 1.0, 50, 50, 50, layer3.data());
  for (int i=0; i<50; ++i){
    layer3[i] = tanh(layer3[i]);
  }

  projection[0] = b_3_[0];
  GeneralMatrixVectorMultiply(w_3_.data(), 'T', layer3.data(), 1.0, 1.0, 1, 50, 1, projection);
  projection[0] = tanh(projection[0]);
}

/*
  Deep Kernel: ``cov(x_1, x_2) = \alpha * \exp(-1/2 * ((g(x_1) - g(x_2))^T * L^{-1} * (g(x_1) - g(x_2))) )``
*/
void DeepKernel::Covariance(double const * restrict point_one, int const * restrict derivatives_one, int num_derivatives_one,
                            double const * restrict point_two, int const * restrict derivatives_two, int num_derivatives_two,
                            double * restrict cov) const noexcept {
  std::vector<double> projection_one(1);
  NeuralNetwork(point_one, projection_one.data());
  std::vector<double> projection_two(1);
  NeuralNetwork(point_two, projection_two.data());
  const double norm_val = NormSquaredWithInverseWeights(projection_one.data(), projection_two.data(), lengths_sq_.data(), dim_);
  cov[0] = alpha_*std::exp(-0.5*norm_val);
}

/*
  Gradient of Deep Kernel (wrt ``x_1``):
  ``\pderiv{cov(x_1, x_2)}{x_{1,i}} = (x_{2,i} - x_{1,i}) / L_{i}^2 * cov(x_1, x_2)``
*/
void DeepKernel::GradCovariance(double const * restrict point_one, int const * restrict derivatives_one, int num_derivatives_one,
                                double const * restrict point_two, int const * restrict derivatives_two, int num_derivatives_two,
                                double * restrict grad_cov) const noexcept {
  std::vector<double> layer1(b_0_);
  std::vector<double> layer2(b_1_);
  std::vector<double> layer3(b_2_);

  GeneralMatrixVectorMultiply(w_0_.data(), 'T', point_one, 1.0, 1.0, dim_, 50, dim_, layer1.data());
  for (int i=0; i<50; ++i){
    layer1[i] = tanh(layer1[i]);
  }

  GeneralMatrixVectorMultiply(w_1_.data(), 'T', layer1.data(), 1.0, 1.0, 50, 50, 50, layer2.data());
  for (int i=0; i<50; ++i){
    layer2[i] = tanh(layer2[i]);
  }

  GeneralMatrixVectorMultiply(w_2_.data(), 'T', layer2.data(), 1.0, 1.0, 50, 50, 50, layer3.data());
  for (int i=0; i<50; ++i){
    layer3[i] = tanh(layer3[i]);
  }

  std::vector<double> projection_one(1);
  projection_one[0] = b_3_[0];
  GeneralMatrixVectorMultiply(w_3_.data(), 'N', layer3.data(), 1.0, 1.0, 1, 50, 1, projection_one.data());
  projection_one[0] = tanh(projection_one[0]);

  std::vector<double> projection_two(1);
  NeuralNetwork(point_two, projection_two.data());
  const double norm_val = NormSquaredWithInverseWeights(projection_one.data(), projection_two.data(), lengths_sq_.data(), dim_);

  const double cov = alpha_*std::exp(-0.5*norm_val);
  std::vector<double> top(1);
  top[0] = (projection_one[0]-projection_two[0])/lengths_sq_[0]*cov;
  top[0] = top[0]*(1-Square(projection_one[0]));

  std::vector<double> layer_dervi3(50);
  GeneralMatrixVectorMultiply(w_3_.data(), 'N', top.data(), 1.0, 0.0, 50, 1, 50, layer_dervi3.data());
  for (int i=0; i<50; ++i){
    layer_dervi3[i] *= (1-Square(layer3[i]));
  }

  std::vector<double> layer_dervi2(50);
  GeneralMatrixVectorMultiply(w_2_.data(), 'N', layer_dervi3.data(), 1.0, 0.0, 50, 50, 50, layer_dervi2.data());
  for (int i=0; i<50; ++i){
    layer_dervi2[i] *= (1-Square(layer2[i]));
  }

  std::vector<double> layer_dervi1(50);
  GeneralMatrixVectorMultiply(w_1_.data(), 'N', layer_dervi2.data(), 1.0, 0.0, 50, 50, 50, layer_dervi1.data());
  for (int i=0; i<50; ++i){
    layer_dervi1[i] *= (1-Square(layer1[i]));
  }

  GeneralMatrixVectorMultiply(w_0_.data(), 'N', layer_dervi1.data(), 1.0, 0.0, dim_, 50, 50, grad_cov);
}

CovarianceInterface * DeepKernel::Clone() const {
  return new DeepKernel(*this);
}

/*namespace {

// computes ||p1 - p2||_2 if all entries of L == 1
OL_PURE_FUNCTION OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT double
NormSquaredWithConstInverseWeights(double const * restrict point_one, double const * restrict point_two,
                                   double weight, int size) noexcept {
  // calculates the one norm of two vectors (point_one and point_two of size size)
  double norm = 0.0;

  for (int i = 0; i < size; ++i) {
    norm += Square(point_one[i] - point_two[i]);
  }
  return norm/weight;
}

}  // end unnamed namespace

SquareExponentialSingleLength::SquareExponentialSingleLength(int dim, double alpha, double length)
    : dim_(dim), alpha_(alpha), length_(length), length_sq_(length*length) {
}

SquareExponentialSingleLength::SquareExponentialSingleLength(int dim, double alpha, double const * restrict length)
    : SquareExponentialSingleLength(dim, alpha, length[0]) {
}

SquareExponentialSingleLength::SquareExponentialSingleLength(int dim, double alpha, std::vector<double> length)
    : SquareExponentialSingleLength(dim, alpha, length[0]) {
}

SquareExponentialSingleLength::SquareExponentialSingleLength(const SquareExponentialSingleLength& OL_UNUSED(source)) = default;

double SquareExponentialSingleLength::Covariance(double const * restrict point_one,
                                                 double const * restrict point_two) const noexcept {
  const double norm_val = NormSquaredWithConstInverseWeights(point_one, point_two, length_sq_, dim_);
  return alpha_*std::exp(-0.5*norm_val);
}

void SquareExponentialSingleLength::GradCovariance(double const * restrict point_one, double const * restrict point_two,
                                                   double * restrict grad_cov) const noexcept {
  const double cov = Covariance(point_one, point_two)/length_sq_;

  for (int i = 0; i < dim_; ++i) {
    grad_cov[i] = (point_two[i] - point_one[i])*cov;
  }
}

void SquareExponentialSingleLength::HyperparameterGradCovariance(double const * restrict point_one,
                                                                 double const * restrict point_two,
                                                                 double * restrict grad_hyperparameter_cov) const noexcept {
  const double cov = Covariance(point_one, point_two);
  const double norm_val = NormSquaredWithConstInverseWeights(point_one, point_two, length_sq_, dim_);

  grad_hyperparameter_cov[0] = cov/alpha_;
  grad_hyperparameter_cov[1] = cov*norm_val/length_;
}

void SquareExponentialSingleLength::HyperparameterHessianCovariance(double const * restrict point_one,
                                                                    double const * restrict point_two,
                                                                    double * restrict hessian_hyperparameter_cov) const noexcept {
  const double cov = Covariance(point_one, point_two);
  const double scaled_norm_val = NormSquaredWithConstInverseWeights(point_one, point_two, length_sq_, dim_)/length_;

  hessian_hyperparameter_cov[0] = 0.0;
  hessian_hyperparameter_cov[1] = cov/alpha_*scaled_norm_val;
  hessian_hyperparameter_cov[2] = hessian_hyperparameter_cov[1];
  hessian_hyperparameter_cov[3] = cov*(Square(scaled_norm_val) - 3.0*scaled_norm_val/length_);
}

CovarianceInterface * SquareExponentialSingleLength::Clone() const {
  return new SquareExponentialSingleLength(*this);
}

void MaternNu1p5::Initialize() {
  InitializeCovariance(dim_, alpha_, lengths_, lengths_sq_.data());
}

MaternNu1p5::MaternNu1p5(int dim, double alpha, std::vector<double> lengths)
    : dim_(dim), alpha_(alpha), lengths_(lengths), lengths_sq_(dim) {
  Initialize();
}

MaternNu1p5::MaternNu1p5(int dim, double alpha, double const * restrict lengths)
    : MaternNu1p5(dim, alpha, std::vector<double>(lengths, lengths + dim)) {
}

MaternNu1p5::MaternNu1p5(int dim, double alpha, double length)
    : MaternNu1p5(dim, alpha, std::vector<double>(dim, length)) {
}

MaternNu1p5::MaternNu1p5(const MaternNu1p5& OL_UNUSED(source)) = default;

double MaternNu1p5::Covariance(double const * restrict point_one, double const * restrict point_two) const noexcept {
  const double norm_val = NormSquaredWithInverseWeights(point_one, point_two, lengths_sq_.data(), dim_);
  const double matern_arg = kSqrt3 * std::sqrt(norm_val);

  return alpha_*(1.0 + matern_arg)*std::exp(-matern_arg);
}

void MaternNu1p5::GradCovariance(double const * restrict point_one, double const * restrict point_two,
                                 double * restrict grad_cov) const noexcept {
  const double norm_val = NormSquaredWithInverseWeights(point_one, point_two, lengths_sq_.data(), dim_);
  if (norm_val == 0.0) {
    std::fill(grad_cov, grad_cov + dim_, 0.0);
    return;
  }
  const double matern_arg = kSqrt3 * std::sqrt(norm_val);
  const double exp_part = std::exp(-matern_arg);

  // terms from differentiating Covariance wrt spatial dimensions; since exp(x) is the derivative's identity, some cancellation of
  // analytic 0s is possible (and desired since it reduces compute-time and is more accurate)
  for (int i = 0; i < dim_; ++i) {
    const double dr_dxi = (point_one[i] - point_two[i])/std::sqrt(norm_val)/lengths_sq_[i];
    grad_cov[i] = -3.0*alpha_*dr_dxi*exp_part*std::sqrt(norm_val);
  }
}

void MaternNu1p5::HyperparameterGradCovariance(double const * restrict point_one, double const * restrict point_two,
                                               double * restrict grad_hyperparameter_cov) const noexcept {
  const double norm_val = NormSquaredWithInverseWeights(point_one, point_two, lengths_sq_.data(), dim_);
  if (norm_val == 0.0) {
    grad_hyperparameter_cov[0] = 1.0;
    std::fill(grad_hyperparameter_cov+1, grad_hyperparameter_cov + 1+dim_, 0.0);
    return;
  }
  const double matern_arg = kSqrt3 * std::sqrt(norm_val);
  const double exp_part = std::exp(-matern_arg);

  // deriv wrt alpha does not have the same form as the length terms, special case it
  grad_hyperparameter_cov[0] = (1.0 + matern_arg) * exp_part;
  // terms from differentiating Covariance wrt spatial dimensions; since exp(x) is the derivative's identity, some cancellation of
  // analytic 0s is possible (and desired since it reduces compute-time and is more accurate)
  for (int i = 0; i < dim_; ++i) {
    const double dr_dleni = -Square((point_one[i] - point_two[i])/lengths_[i])/std::sqrt(norm_val)/lengths_[i];
    grad_hyperparameter_cov[i+1] = -3.0*alpha_*dr_dleni*exp_part*std::sqrt(norm_val);
  }
}

void MaternNu1p5::HyperparameterHessianCovariance(double const * restrict point_one, double const * restrict point_two,
                                                  double * restrict hessian_hyperparameter_cov) const noexcept {
  const int num_hyperparameters = GetNumberOfHyperparameters();
  const double norm_val = NormSquaredWithInverseWeights(point_one, point_two, lengths_sq_.data(), dim_);
  if (norm_val == 0.0) {
    std::fill(hessian_hyperparameter_cov, hessian_hyperparameter_cov + Square(num_hyperparameters), 0.0);
    return;
  }

  const double sqrt_norm_val = std::sqrt(norm_val);
  const double matern_arg = kSqrt3 * sqrt_norm_val;
  const double exp_part = std::exp(-matern_arg);

  // precompute deriv of r wrt length hyperparameters, where r is NormSquaredWithInverseWeights
  std::vector<double> dr_dlen(num_hyperparameters);
  for (int i = 0; i < dim_; ++i) {
    // note the indexing is shifted by one here so that we can iterate over componentwise_scaled_distance with the
    // same index that is used for hyperparameter indexing
    dr_dlen[i+1] = -Square((point_one[i] - point_two[i])/lengths_[i])/lengths_[i]/sqrt_norm_val;  // -1/r*(x1_i - x2_i)^2/l_i^3
  }

  double const * restrict hessian_hyperparameter_cov_row = hessian_hyperparameter_cov + 1;  // used to index through rows
  // first column of hessian, derivatives of pderiv{cov}{\theta_0} with respect to \theta_i
  // \theta_0 is alpha, the scaling factor; its derivatives are fundamentally different than those wrt length scales; hence
  // they are split out of the loop
  // deriv wrt alpha does not have the same form as the length terms, special case it
  hessian_hyperparameter_cov[0] = 0.0;
  for (int i = 1; i < num_hyperparameters; ++i) {
    // this is simply pderiv{cov}{\theta_i}/alpha, i = 1..dim+1
    hessian_hyperparameter_cov[i] = -3.0*dr_dlen[i]*exp_part*sqrt_norm_val;
  }
  hessian_hyperparameter_cov += num_hyperparameters;

  // remaining columns of the hessian: derivatives with respect to pderiv{cov}{\theta_j} for j = 1..dim+1 (length scales)
  // terms from differentiating HyperparameterGradCovariance wrt spatial dimensions; since exp(x) is the derivative's identity,
  // some cancellation of analytic 0s is possible (and desired since it reduces compute-time and is more accurate)
  for (int j = 1; j < num_hyperparameters; ++j) {
    // copy j-th column from j-th row, which has already been computed
    for (int i = 0; i < j; ++i) {
      hessian_hyperparameter_cov[i] = hessian_hyperparameter_cov_row[0];
      hessian_hyperparameter_cov_row += num_hyperparameters;
    }
    hessian_hyperparameter_cov_row -= j*num_hyperparameters;  // reset row for next iteration

    // on diagonal component has extra terms since normally dx_i/dx_j = 0 except for i == j
    // the RHS terms are only read from already-computed or copied components
    hessian_hyperparameter_cov[j] = -kSqrt3*alpha_*hessian_hyperparameter_cov[0]*dr_dlen[j] +
        alpha_*(-3.0/lengths_[j-1])*hessian_hyperparameter_cov[0];
    // remaining off-digaonal terms
    for (int i = j+1; i < num_hyperparameters; ++i) {
      hessian_hyperparameter_cov[i] = -kSqrt3*alpha_*hessian_hyperparameter_cov[0]*dr_dlen[i];
    }

    hessian_hyperparameter_cov += num_hyperparameters;
    hessian_hyperparameter_cov_row += 1;
  }
}

CovarianceInterface * MaternNu1p5::Clone() const {
  return new MaternNu1p5(*this);
}

void MaternNu2p5::Initialize() {
  InitializeCovariance(dim_, alpha_, lengths_, lengths_sq_.data());
}

MaternNu2p5::MaternNu2p5(int dim, double alpha, std::vector<double> lengths)
    : dim_(dim), alpha_(alpha), lengths_(lengths), lengths_sq_(dim) {
  Initialize();
}

MaternNu2p5::MaternNu2p5(int dim, double alpha, double const * restrict lengths)
    : MaternNu2p5(dim, alpha, std::vector<double>(lengths, lengths + dim)) {
}

MaternNu2p5::MaternNu2p5(int dim, double alpha, double length)
    : MaternNu2p5(dim, alpha, std::vector<double>(dim, length)) {
}

MaternNu2p5::MaternNu2p5(const MaternNu2p5& OL_UNUSED(source)) = default;

double MaternNu2p5::Covariance(double const * restrict point_one, double const * restrict point_two) const noexcept {
  const double norm_val = NormSquaredWithInverseWeights(point_one, point_two, lengths_sq_.data(), dim_);
  const double matern_arg = kSqrt5 * std::sqrt(norm_val);

  return alpha_*(1.0 + matern_arg + 5.0/3.0*norm_val)*std::exp(-matern_arg);
}

void MaternNu2p5::GradCovariance(double const * restrict point_one, double const * restrict point_two,
                                 double * restrict grad_cov) const noexcept {
  const double norm_val = NormSquaredWithInverseWeights(point_one, point_two, lengths_sq_.data(), dim_);
  if (norm_val == 0.0) {
    std::fill(grad_cov, grad_cov + dim_, 0.0);
    return;
  }
  const double matern_arg = kSqrt5 * std::sqrt(norm_val);
  const double poly_part = matern_arg + 5.0/3.0*norm_val;
  const double exp_part = std::exp(-matern_arg);

  for (int i = 0; i < dim_; ++i) {
    const double dr2_dxi = 2.0*(point_one[i] - point_two[i])/lengths_sq_[i];
    const double dr_dxi = 0.5*dr2_dxi/std::sqrt(norm_val);
    grad_cov[i] = alpha_*exp_part*(5.0/3.0*dr2_dxi - poly_part*kSqrt5*dr_dxi);
  }
}

void MaternNu2p5::HyperparameterGradCovariance(double const * restrict point_one, double const * restrict point_two,
                                               double * restrict grad_hyperparameter_cov) const noexcept {
  const double norm_val = NormSquaredWithInverseWeights(point_one, point_two, lengths_sq_.data(), dim_);
  if (norm_val == 0.0) {
    grad_hyperparameter_cov[0] = 1.0;
    std::fill(grad_hyperparameter_cov+1, grad_hyperparameter_cov + 1+dim_, 0.0);
    return;
  }
  const double matern_arg = kSqrt5 * std::sqrt(norm_val);
  const double poly_part = matern_arg + 5.0/3.0*norm_val;
  const double exp_part = std::exp(-matern_arg);

  // deriv wrt alpha does not have the same form as the length terms, special case it
  grad_hyperparameter_cov[0] = (1.0 + poly_part) * exp_part;
  // terms from differentiating Covariance wrt spatial dimensions; since exp(x) is the derivative's identity, some cancellation of
  // analytic 0s is possible (and desired since it reduces compute-time and is more accurate)
  for (int i = 0; i < dim_; ++i) {
    const double dr2_dleni = -2.0*Square((point_one[i] - point_two[i])/lengths_[i])/lengths_[i];
    const double dr_dleni = 0.5*dr2_dleni/std::sqrt(norm_val);
    grad_hyperparameter_cov[i+1] = alpha_*exp_part*(5.0/3.0*dr2_dleni - poly_part*kSqrt5*dr_dleni);
  }
}

void MaternNu2p5::HyperparameterHessianCovariance(double const * restrict point_one, double const * restrict point_two,
                                                  double * restrict hessian_hyperparameter_cov) const noexcept {
  const int num_hyperparameters = GetNumberOfHyperparameters();
  const double norm_val = NormSquaredWithInverseWeights(point_one, point_two, lengths_sq_.data(), dim_);
  if (norm_val == 0.0) {
    std::fill(hessian_hyperparameter_cov, hessian_hyperparameter_cov + Square(num_hyperparameters), 0.0);
    return;
  }

  const double sqrt_norm_val = std::sqrt(norm_val);
  const double matern_arg = kSqrt5 * sqrt_norm_val;
  const double poly_part = matern_arg + 5.0/3.0*norm_val;
  const double exp_part = std::exp(-matern_arg);

  // precompute deriv of r wrt length hyperparameters, where r is NormSquaredWithInverseWeights
  std::vector<double> dr_dlen(num_hyperparameters);
  for (int i = 0; i < dim_; ++i) {
    // note the indexing is shifted by one here so that we can iterate over componentwise_scaled_distance with the
    // same index that is used for hyperparameter indexing
    dr_dlen[i+1] = -Square((point_one[i] - point_two[i])/lengths_[i])/lengths_[i]/sqrt_norm_val;  // -1/r*(x1_i - x2_i)^2/l_i^3
  }

  double const * restrict hessian_hyperparameter_cov_row = hessian_hyperparameter_cov + 1;  // used to index through rows
  // first column of hessian, derivatives of pderiv{cov}{\theta_0} with respect to \theta_i
  // \theta_0 is alpha, the scaling factor; its derivatives are fundamentally different than those wrt length scales; hence
  // they are split out of the loop
  // deriv wrt alpha does not have the same form as the length terms, special case it
  hessian_hyperparameter_cov[0] = 0.0;
  for (int i = 1; i < num_hyperparameters; ++i) {
    // this is simply pderiv{cov}{\theta_i}/alpha, i = 1..dim+1
    const double dr2_dlenj = 2.0*sqrt_norm_val*dr_dlen[i];
    hessian_hyperparameter_cov[i] = exp_part*(5.0/3.0*dr2_dlenj - poly_part*kSqrt5*dr_dlen[i]);
  }
  hessian_hyperparameter_cov += num_hyperparameters;

  // remaining columns of the hessian: derivatives with respect to pderiv{cov}{\theta_j} for j = 1..dim+1 (length scales)
  // terms from differentiating HyperparameterGradCovariance wrt spatial dimensions; since exp(x) is the derivative's identity,
  // some cancellation of analytic 0s is possible (and desired since it reduces compute-time and is more accurate)
  for (int j = 1; j < num_hyperparameters; ++j) {
    // copy j-th column from j-th row, which has already been computed
    for (int i = 0; i < j; ++i) {
      hessian_hyperparameter_cov[i] = hessian_hyperparameter_cov_row[0];
      hessian_hyperparameter_cov_row += num_hyperparameters;
    }
    hessian_hyperparameter_cov_row -= j*num_hyperparameters;  // reset row for next iteration

    // on diagonal component has extra terms since normally dx_i/dx_j = 0 except for i == j
    // the RHS terms are only read from already-computed or copied components
    const double dr2_dlenj = 2.0*sqrt_norm_val*dr_dlen[j];
    hessian_hyperparameter_cov[j] = -alpha_*kSqrt5*dr_dlen[j]*(hessian_hyperparameter_cov[0] +
                                                               exp_part*(5.0/3.0*sqrt_norm_val*dr_dlen[j] -
                                                                         (matern_arg + 5.0/3.0*norm_val)*3.0/lengths_[j-1])) +
        alpha_*exp_part*(-5.0*dr2_dlenj/lengths_[j-1]);
    // remaining off-digaonal terms
    for (int i = j+1; i < num_hyperparameters; ++i) {
      hessian_hyperparameter_cov[i] = -alpha_*kSqrt5*dr_dlen[i]*(hessian_hyperparameter_cov[0] +
                                                                 exp_part*5.0/3.0*sqrt_norm_val*dr_dlen[j]);
    }

    hessian_hyperparameter_cov += num_hyperparameters;
    hessian_hyperparameter_cov_row += 1;
  }
}

CovarianceInterface * MaternNu2p5::Clone() const {
  return new MaternNu2p5(*this);
}*/

}  // end namespace optimal_learning
