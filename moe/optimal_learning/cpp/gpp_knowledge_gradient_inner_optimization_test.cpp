/*!
  \file gpp_knowledge_gradient_inner_optimization_test.cpp
  \rst
  Routines to test the functions in gpp_knowledge_gradient_optimization.cpp.

  The tests verify KnowledgeGradientEvaluator, and KG optimization from gpp_knowledge_gradient_optimization.cpp.

  1. Ping testing (verifying analytic gradient computation against finite difference approximations)

     a. Following gpp_covariance_test.cpp, we define class PingKnowledgeGradient for
        evaluating those functions + their spatial gradients.

     b. Ping for derivative accuracy (PingGPComponentTest, PingEITest); these unit test the analytic derivatives.

  2. Monte-Carlo KG vs analytic KG validation: the monte-carlo versions are run to "high" accuracy and checked against
     analytic formulae when applicable
  3. Gradient Descent: using polynomials and other simple fucntions with analytically known optima
     to verify that the algorithm(s) underlying KG optimization are performing correctly.
  4. Single-threaded vs multi-threaded KG optimization validation: single and multi-threaded runs are checked to have the same
     output.
  5. End-to-end test of the KG optimization process for the analytic and monte-carlo cases.  These tests use constructed
     data for inputs but otherwise exercise the same code paths used for KG optimization in production.
\endrst*/

// #define OL_VERBOSE_PRINT
#include "gpp_knowledge_gradient_inner_optimization_test.hpp"

#include <cmath>

#include <algorithm>
#include <limits>
#include <vector>

#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_geometry.hpp"
#include "gpp_linear_algebra.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"
#include "gpp_knowledge_gradient_inner_optimization.hpp"
#include "gpp_optimizer_parameters.hpp"

namespace optimal_learning {

namespace {  // contains classes/routines for ping testing

/*!\rst
  Supports evaluating the knowledge gradient, KnowledgeGradientEvaluator::ComputeKnowledgeGradient() and
  its gradient, KnowledgeGradientEvaluator::ComputeGradKnowledgeGradient()

  The gradient is taken wrt ``points_to_sample[dim]``, so this is the ``input_matrix``, ``X_{d,i}``.
  The other inputs to KG are not differentiated against, so they are taken as input and stored by the constructor.

  The output of KG is a scalar.
\endrst*/
class PingFuturePosteriorMean final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "Posterior Mean";

  PingFuturePosteriorMean(double const * restrict lengths, double const * restrict points_sampled, double const * restrict points_sampled_value,
                          int const * restrict gradients, double const * restrict points_to_sample,
                          double alpha, int dim, int num_to_sample, int num_sampled, int num_gradients) OL_NONNULL_POINTERS
      : dim_(dim),
        num_to_sample_(num_to_sample),
        num_sampled_(num_sampled),
        num_gradients_(num_gradients),
        gradients_already_computed_(false),
        gradients_(gradients, gradients + num_gradients),
        noise_variance_(1+num_gradients, 0.1),
        points_sampled_(points_sampled, points_sampled + dim_*num_sampled_),
        points_to_sample_(points_to_sample, points_to_sample + dim_*num_to_sample_),
        points_sampled_value_(points_sampled_value, points_sampled_value + num_sampled_*(1+num_gradients_)),
        grad_PS_(dim_*num_to_sample_),
        sqexp_covariance_(dim_, alpha, lengths),
        gaussian_process_(sqexp_covariance_, points_sampled_.data(), points_sampled_value_.data(), noise_variance_.data(),
                          gradients_.data(), num_gradients_, dim_, num_sampled_),
        normals_(normals(num_to_sample, num_gradients)),
        chol_(chol()),
        train_sample_(train()),
        ps_evaluator_(gaussian_process_, normals_.data(), points_to_sample_.data(), num_to_sample_, gradients_.data(), num_gradients_,
                      chol_.data(), train_sample_.data()) {
  }

  std::vector<double> normals(int num_to_sample, int num_gradients) {
     std::vector<double> randomDiscrete(num_to_sample*(1+num_gradients));
     UniformRandomGenerator uniform_generator(318);
     boost::uniform_real<double> uniform_double(-5.0, 5.0);
     for (int i = 0; i < num_to_sample*(1+num_gradients); ++i) {
       randomDiscrete[i] = uniform_double(uniform_generator.engine);
     }
     return randomDiscrete;
  }

  std::vector<double> chol() {
     std::vector<double> chol_(Square(num_to_sample_*(1+num_gradients_)));
     PointsToSampleState points_to_sample_state(gaussian_process_, points_to_sample_.data(),
                                                num_to_sample_, gradients_.data(), num_gradients_, 0);
     gaussian_process_.ComputeVarianceOfPoints(&points_to_sample_state, gradients_.data(),
                                                num_gradients_, chol_.data());
     int leading_minor_index = ComputeCholeskyFactorL(num_to_sample_*(1+num_gradients_), chol_.data());
     ZeroUpperTriangle(num_to_sample_*(1+num_gradients_), chol_.data());
     return chol_;
  }

  std::vector<double> train() {
     std::vector<double> randomDiscrete(num_to_sample_*Square(1+num_gradients_)*num_sampled_);
     gaussian_process_.ComputeTrain(points_to_sample_.data(), num_to_sample_, gradients_.data(), num_gradients_, randomDiscrete.data());
     return randomDiscrete;
  }

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS {
    *num_rows = dim_;
    *num_cols = 1;
  }

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return dim_*GetOutputSize()*1;
  }

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return 1;
  }

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict points_to_sample, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2) {
    if (gradients_already_computed_ == true) {
      OL_WARNING_PRINTF("WARNING: grad_KG data already set.  Overwriting...\n");
    }
    gradients_already_computed_ = true;

    NormalRNG normal_rng(3141);
    bool configure_for_gradients = true;

    FuturePosteriorMeanEvaluator::StateType ps_state(ps_evaluator_, 0, points_to_sample, configure_for_gradients);

    ps_evaluator_.ComputeGradPosteriorMean(&ps_state, grad_PS_.data());


    if (gradients != nullptr) {
      std::copy(grad_PS_.begin(), grad_PS_.end(), gradients);
    }
  }

  virtual double GetAnalyticGradient(int row_index, int column_index, int OL_UNUSED(output_index)) const override OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingPosteriorMean::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    return grad_PS_[column_index*dim_ + row_index];
  }

  virtual void EvaluateFunction(double const * restrict points_to_sample, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS {
    bool configure_for_gradients = false;

    FuturePosteriorMeanEvaluator::StateType ps_state(ps_evaluator_, 0, points_to_sample, configure_for_gradients);
    *function_values = ps_evaluator_.ComputePosteriorMean(&ps_state);
  }

 private:
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  int dim_;
  //! number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
  int num_to_sample_;
  //! number of points in ``points_sampled``
  int num_sampled_;
  //! number of derivatives' observations.
  int num_gradients_;

  //! whether gradients been computed and stored--whether this class is ready for use
  bool gradients_already_computed_;
  // indices of the derivatives' observations.
  std::vector<int> gradients_;

  //! ``\sigma_n^2``, the noise variance
  std::vector<double> noise_variance_;
  //! coordinates of already-sampled points, ``X``
  std::vector<double> points_sampled_;
  std::vector<double> points_to_sample_;
  //! function values at points_sampled, ``y``
  std::vector<double> points_sampled_value_;
  //! the gradient of KG at union_of_points, wrt union_of_points[0:num_to_sample]
  std::vector<double> grad_PS_;

  //! covariance class (for computing covariance and its gradients)
  SquareExponential sqexp_covariance_;
  //! gaussian process used for computations
  GaussianProcess gaussian_process_;

  std::vector<double> normals_;
  std::vector<double> chol_;
  std::vector<double> train_sample_;

  //! expected improvement evaluator object that specifies the parameters & GP for KG evaluation
  FuturePosteriorMeanEvaluator ps_evaluator_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingFuturePosteriorMean);
};

/*!\rst
  Pings the gradients (spatial) of the KG 50 times with randomly generated test cases
  Works with MC formulae

  \param
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-KG)
    :num_being_sampled: number of points being sampled in concurrent experiments (i.e., the "p" in q,p-KG)
    :epsilon: coarse, fine ``h`` sizes to use in finite difference computation
    :tolerance_fine: desired amount of deviation from the exact rate
    :tolerance_coarse: maximum allowable abmount of deviation from the exact rate
    :input_output_ratio: for ``||analytic_gradient||/||input|| < input_output_ratio``, ping testing is not performed, see PingDerivative()
  \return
    number of ping/test failures
\endrst*/
template <typename PSEvaluator>
OL_WARN_UNUSED_RESULT int PingPSTest(int num_to_sample, double epsilon[2], double tolerance_fine, double tolerance_coarse, double input_output_ratio) {
  int total_errors = 0;
  int errors_this_iteration;
  const int dim = 3;

  int num_sampled = 7;

  int gradients [3] = {0, 1, 2};
  int num_gradients = 3;

  std::vector<double> lengths(dim);
  double alpha = 2.80723;

  MockExpectedImprovementEnvironment KG_environment;

  UniformRandomGenerator uniform_generator(2718);
  boost::uniform_real<double> uniform_double(0.5, 2.5);

  for (int i = 0; i < 50; ++i) {
    KG_environment.Initialize(dim, num_to_sample, 3, num_sampled, num_gradients);
    for (int j = 0; j < dim; ++j) {
      lengths[j] = uniform_double(uniform_generator.engine);
    }

    PSEvaluator PS_evaluator(lengths.data(), KG_environment.points_sampled(),
                             KG_environment.points_sampled_value(), gradients, KG_environment.points_being_sampled(),
                             alpha, KG_environment.dim, KG_environment.num_being_sampled, KG_environment.num_sampled, num_gradients);

    //KGEvaluator KG_evaluator(lengths.data(), KG_environment.points_sampled(), KG_environment.points_sampled_value(), alpha, KG_environment.dim, KG_environment.num_to_sample, KG_environment.num_sampled);
    PS_evaluator.EvaluateAndStoreAnalyticGradient(KG_environment.points_to_sample(), nullptr);

    errors_this_iteration = PingDerivative(PS_evaluator, KG_environment.points_to_sample(), epsilon, tolerance_fine, tolerance_coarse, input_output_ratio);

    if (errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("on iteration %d\n", i);
    }
    total_errors += errors_this_iteration;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s (%d,%d-PS) gradient pings failed with %d errors\n", PSEvaluator::kName, num_to_sample, 0, total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("%s (%d,%d-PS) gradient pings passed\n", PSEvaluator::kName, num_to_sample, 0);
  }

  return total_errors;
};

}  // end unnamed namespace

/*!\rst
  Wrapper to ping the gradients (spatial) of the inverse of the cholesky factorization with noise.

  \return
    number of ping/test failures
\endrst*/

int PingKGInnerGeneralTest() {
  double epsilon_KG[2] = {1.0e-3, 1.0e-4};

  int total_errors = PingPSTest<PingFuturePosteriorMean>(1, epsilon_KG, 9.0e-2, 3.0e-1, 1.0e-18);

  return total_errors;
}


int RunKGInnerTests() {
  int total_errors = 0;
  int current_errors = 0;

  {
    current_errors = PingKGInnerGeneralTest();
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging inner KG failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("Inner KG functions failed with %d errors\n\n", total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("Inner KG functions passed\n");
  }

  return total_errors;
}
}  // end namespace optimal_learning