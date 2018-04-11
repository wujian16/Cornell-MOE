/*!
  \file gpp_knowledge_gradient_optimization_test.cpp
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
#include "gpp_knowledge_gradient_optimization_test.hpp"
#include "gpp_two_step_expected_improvement_optimization_test.hpp"

#include <cmath>

#include <algorithm>
#include <limits>
#include <vector>

#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_domain.hpp"
#include "gpp_geometry.hpp"
#include "gpp_linear_algebra.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"
#include "gpp_knowledge_gradient_optimization.hpp"
#include "gpp_two_step_expected_improvement_optimization.hpp"
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
class PingTwoExpectedImprovement final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "Two-Step EI with MC integration";

  PingTwoExpectedImprovement(TensorProductDomain domain, GradientDescentParameters& optimizer_parameters,
                        double const * restrict lengths, double const * restrict points_being_sampled,
                        double const * restrict points_sampled, double const * restrict points_sampled_value,
                        int const * restrict gradients, double alpha, double best_so_far, int dim, int num_to_sample, int num_being_sampled,
                        int num_sampled, int num_mc_iter, int num_pts, int num_gradients) OL_NONNULL_POINTERS
      : dim_(dim),
        domain_(domain),
        gdp_(optimizer_parameters.num_multistarts, optimizer_parameters.max_num_steps,
             optimizer_parameters.max_num_restarts, optimizer_parameters.num_steps_averaged,
             optimizer_parameters.gamma, optimizer_parameters.pre_mult,
             optimizer_parameters.max_relative_change, optimizer_parameters.tolerance),
        num_to_sample_(num_to_sample),
        num_being_sampled_(num_being_sampled),
        num_sampled_(num_sampled),
        num_pts_(num_pts),
        num_gradients_(num_gradients),
        gradients_already_computed_(false),
        gradients_(gradients, gradients + num_gradients),
        noise_variance_(1+num_gradients, 0.1),
        points_sampled_(points_sampled, points_sampled + dim_*num_sampled_),
        points_sampled_value_(points_sampled_value, points_sampled_value + num_sampled_*(1+num_gradients_)),
        points_being_sampled_(points_being_sampled, points_being_sampled + num_being_sampled_*dim_),
        discrete_pts_(random_discrete(dim_, num_pts_)),
        grad_VF_(dim_*num_to_sample_),
        sqexp_covariance_(dim_, alpha, lengths),
        gaussian_process_(sqexp_covariance_, points_sampled_.data(), points_sampled_value_.data(), noise_variance_.data(),
                          gradients_.data(), num_gradients_, dim_, num_sampled_),
        twoei_evaluator_(gaussian_process_, 0, discrete_pts_.data(), num_pts, num_mc_iter, domain_, optimizer_parameters, best_so_far) {
  }

  std::vector<double> random_discrete(int dim, int num_pts){
     std::vector<double> randomDiscrete(dim*num_pts);
     UniformRandomGenerator uniform_generator(318);
     boost::uniform_real<double> uniform_double(-5.0, 5.0);
     for (int i = 0; i < dim_*num_pts_; ++i) {
       randomDiscrete[i] = uniform_double(uniform_generator.engine);
     }
     return randomDiscrete;
  }

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS {
    *num_rows = dim_;
    *num_cols = num_to_sample_;
  }

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return dim_*GetOutputSize()*num_to_sample_;
  }

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return 1;
  }

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict points_to_sample, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2) {
    if (gradients_already_computed_ == true) {
      OL_WARNING_PRINTF("WARNING: grad_two_step data already set.  Overwriting...\n");
    }
    gradients_already_computed_ = true;

    NormalRNG normal_rng(3141);
    bool configure_for_gradients = true;

    TwoStepExpectedImprovementEvaluator<TensorProductDomain>::StateType twoei_state(twoei_evaluator_, points_to_sample, points_being_sampled_.data(),
                                                                        num_to_sample_, num_being_sampled_, twoei_evaluator_.num_mc_iterations(), gradients_.data(),
                                                                        num_gradients_, configure_for_gradients, &normal_rng);

    twoei_evaluator_.ComputeGradValueFunction(&twoei_state, grad_VF_.data());

    if (gradients != nullptr) {
      std::copy(grad_VF_.begin(), grad_VF_.end(), gradients);
    }
  }

  virtual double GetAnalyticGradient(int row_index, int column_index, int OL_UNUSED(output_index)) const override OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingTwoExpectedImprovement::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    return grad_VF_[column_index*dim_ + row_index];
  }

  virtual void EvaluateFunction(double const * restrict points_to_sample, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS {
    NormalRNG normal_rng(3141);
    bool configure_for_gradients = false;

    TwoStepExpectedImprovementEvaluator<TensorProductDomain>::StateType twoei_state(twoei_evaluator_, points_to_sample, points_being_sampled_.data(),
                                                                                num_to_sample_, num_being_sampled_, twoei_evaluator_.num_mc_iterations(), gradients_.data(),
                                                                                num_gradients_, configure_for_gradients, &normal_rng);
    *function_values = twoei_evaluator_.ComputeValueFunction(&twoei_state);
  }

 private:
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  int dim_;
  //! domain
  TensorProductDomain domain_;
  //! gradient decent para
  GradientDescentParameters gdp_;
  //! number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
  int num_to_sample_;
  //! number of points being sampled concurrently (i.e., the "p" in q,p-EI)
  int num_being_sampled_;
  //! number of points in ``points_sampled``
  int num_sampled_;
  //! number of points in ``discret_pts''
  int num_pts_;
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
  //! function values at points_sampled, ``y``
  std::vector<double> points_sampled_value_;
  //! points that are being sampled in concurrently experiments
  std::vector<double> points_being_sampled_;
  //! points to approximate KG
  std::vector<double> discrete_pts_;
  //! the gradient of KG at union_of_points, wrt union_of_points[0:num_to_sample]
  std::vector<double> grad_VF_;

  //! covariance class (for computing covariance and its gradients)
  SquareExponential sqexp_covariance_;
  //! gaussian process used for computations
  GaussianProcess gaussian_process_;
  //! expected improvement evaluator object that specifies the parameters & GP for KG evaluation
  TwoStepExpectedImprovementEvaluator<TensorProductDomain> twoei_evaluator_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingTwoExpectedImprovement);
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
template <typename TwoEIEvaluator>
OL_WARN_UNUSED_RESULT int PingTwoEITest(int num_to_sample, int num_being_sampled, double epsilon[2],
                                        double tolerance_fine, double tolerance_coarse, double input_output_ratio) {
  using DomainType = TensorProductDomain;
  int total_errors = 0;
  int errors_this_iteration;
  const int dim = 3;

  int num_sampled = 7;
  int num_pts = 5;

  int * gradients = new int[3]{0, 1, 2};
  int num_gradients = 3;

  std::vector<double> lengths(dim);
  double alpha = 2.80723;
  // set best_so_far to be larger than max(points_sampled_value) (but don't make it huge or stability will be suffer)
  double best_so_far = 7.0;
  const int num_mc_iter = 16;

  MockExpectedImprovementEnvironment KG_environment;

  // gradient descent parameters
  const double gamma = 0.7;
  const double pre_mult = 1.0;
  const double max_relative_change = 0.7;
  const double tolerance = 1.0e-5;

  const int max_gradient_descent_steps = 500;
  const int max_num_restarts = 50;
  const int num_steps_averaged = 15;

  GradientDescentParameters gd_params(1, max_gradient_descent_steps, max_num_restarts,
                                      num_steps_averaged, gamma, pre_mult,
                                      max_relative_change, tolerance);
  ClosedInterval * domain_bounds = new ClosedInterval[dim];
  for (int i=0; i<dim; ++i){
    domain_bounds[i] = ClosedInterval(-5.0, 5.0);
  }
  delete [] domain_bounds;
  TensorProductDomain domain(domain_bounds, dim);
  // seed randoms
  UniformRandomGenerator uniform_generator(314);

  //UniformRandomGenerator uniform_generator(2718);
  boost::uniform_real<double> uniform_double(0.5, 2.5);

  for (int i = 0; i < 10; ++i) {
    KG_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled, num_gradients);
    //std::vector<double> noise_variance(num_sampled, 0.0003);
    for (int j = 0; j < dim; ++j) {
      lengths[j] = uniform_double(uniform_generator.engine);
    }

    TwoEIEvaluator twoei_evaluator(domain, gd_params, lengths.data(), KG_environment.points_being_sampled(), KG_environment.points_sampled(),
                             KG_environment.points_sampled_value(), gradients, alpha, best_so_far, KG_environment.dim,
                             KG_environment.num_to_sample, KG_environment.num_being_sampled, KG_environment.num_sampled,
                             num_mc_iter, num_pts, num_gradients);

    //KGEvaluator KG_evaluator(lengths.data(), KG_environment.points_sampled(), KG_environment.points_sampled_value(), alpha, KG_environment.dim, KG_environment.num_to_sample, KG_environment.num_sampled);
    twoei_evaluator.EvaluateAndStoreAnalyticGradient(KG_environment.points_to_sample(), nullptr);

    errors_this_iteration = PingDerivative(twoei_evaluator, KG_environment.points_to_sample(), epsilon, tolerance_fine, tolerance_coarse, input_output_ratio);

    if (errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("on iteration %d\n", i);
    }
    total_errors += errors_this_iteration;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s (%d,%d-two-step) gradient pings failed with %d errors\n", TwoEIEvaluator::kName, num_to_sample, num_being_sampled, total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("%s (%d,%d-two-step) gradient pings passed\n", TwoEIEvaluator::kName, num_to_sample, num_being_sampled);
  }

  delete [] gradients;
  return total_errors;
};

}  // end unnamed namespace


/*!\rst
  Wrapper to ping the gradients (spatial) of the inverse of the cholesky factorization with noise.

  \return
    number of ping/test failures
\endrst*/

int PingTwoEIGeneralTest() {
  double epsilon_KG[2] = {1.0e-4, 1.0e-5};
  int total_errors = PingTwoEITest<PingTwoExpectedImprovement>(1, 0, epsilon_KG, 9.0e-2, 3.0e-1, 1.0e-18);

  total_errors += PingTwoEITest<PingTwoExpectedImprovement>(2, 0, epsilon_KG, 9.0e-2, 3.0e-1, 1.0e-18);

  total_errors += PingTwoEITest<PingTwoExpectedImprovement>(1, 2, epsilon_KG, 9.0e-2, 3.0e-1, 1.0e-18);

  total_errors += PingTwoEITest<PingTwoExpectedImprovement>(3, 2, epsilon_KG, 9.0e-2, 3.0e-1, 1.0e-18);

  return total_errors;
}

int RunTwoEITests() {
  int total_errors = 0;
  int current_errors = 0;

  {
    current_errors = PingTwoEIGeneralTest();
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging two-step EI failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("two-step EI functions failed with %d errors\n\n", total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("two-step EI functions passed\n");
  }

  return total_errors;
}


}  // end namespace optimal_learning