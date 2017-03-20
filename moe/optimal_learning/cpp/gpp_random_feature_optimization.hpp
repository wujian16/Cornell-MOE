/*!
  \file gpp_random_feature_optimization.hpp
  \rst
\endrst*/


#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_RANDOM_FEATURE_OPTIMIZATION_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_RANDOM_FEATURE_OPTIMIZATION_HPP_

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
class RandomFeature final {
  public:

  RandomFeature(const GaussianProcess& gaussian_process_in,
                const int nFeature,
                NormalRNG * rng);

  int dim() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim_;
  }

  int nFeature() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return nFeature_;
  }

  void Feature_W (double * feature_W){
    std::copy(W.data(), W.data() + dim_*nFeature_, feature_W);
  }

  void Feature_b (double * feature_b){
    std::copy(b.data(), b.data() + nFeature_, feature_b);
  }

  void Theta (double * theta_mean){
    std::copy(theta_mean_.data(), theta_mean_.data() + nFeature_, theta_mean);
  }

  void Theta_Var (double * theta_var){
    std::copy(theta_var_.data(), theta_var_.data() + nFeature_*nFeature_, theta_var);
  }

  private:
    const int dim_;

    const int nFeature_;

    const std::vector<double> W;

    const std::vector<double> b;

    const std::vector<double> theta_mean_;

    const std::vector<double> theta_var_;
};
}