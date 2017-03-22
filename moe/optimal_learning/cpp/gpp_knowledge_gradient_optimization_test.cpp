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
#include "gpp_optimizer_parameters.hpp"

namespace optimal_learning {

MockKnowledgeGradientEnvironment::MockKnowledgeGradientEnvironment()
    : dim(-1),
      num_sampled(-1),
      num_to_sample(-1),
      num_being_sampled(-1),
      num_pts(-1),
      num_derivatives(-1),
      points_sampled_(20*4),
      points_sampled_value_(20),
      points_to_sample_(4),
      points_being_sampled_(20*4),
      discrete_pts_(20*4),
      uniform_generator_(kDefaultSeed),
      uniform_double_(range_min, range_max) {
}

void MockKnowledgeGradientEnvironment::Initialize(int dim_in, int num_to_sample_in, int num_being_sampled_in,
               int num_sampled_in, int num_pts_in, int num_derivatives_in, UniformRandomGenerator * uniform_generator) {
  if (dim_in != dim || num_to_sample_in != num_to_sample || num_being_sampled_in != num_being_sampled || num_sampled_in != num_sampled || num_pts_in != num_pts
     || num_derivatives_in != num_derivatives) {
    dim = dim_in;
    num_to_sample = num_to_sample_in;
    num_being_sampled = num_being_sampled_in;
    num_sampled = num_sampled_in;
    num_pts = num_pts_in;
    num_derivatives = num_derivatives_in;

    points_sampled_.resize(num_sampled*dim);
    points_sampled_value_.resize(num_sampled*(1+num_derivatives));
    points_to_sample_.resize(num_to_sample*dim);
    points_being_sampled_.resize(num_being_sampled*dim);
    discrete_pts_.resize(num_pts*dim);
  }

  for (int i = 0; i < dim*num_sampled; ++i) {
    points_sampled_[i] = uniform_double_(uniform_generator->engine);
  }

  for (int i = 0; i < num_sampled*(1+num_derivatives); ++i) {
    points_sampled_value_[i] = uniform_double_(uniform_generator->engine);
  }

  for (int i = 0; i < dim*num_to_sample; ++i) {
    points_to_sample_[i] = uniform_double_(uniform_generator->engine);
  }

  for (int i = 0; i < dim*num_being_sampled; ++i) {
    points_being_sampled_[i] = uniform_double_(uniform_generator->engine);
  }

  for (int i = 0; i < dim*num_pts; ++i) {
    discrete_pts_[i] = uniform_double_(uniform_generator->engine);
  }
}

namespace {  // contains classes/routines for ping testing

/*!\rst
  Supports evaluating the knowledge gradient, KnowledgeGradientEvaluator::ComputeKnowledgeGradient() and
  its gradient, KnowledgeGradientEvaluator::ComputeGradKnowledgeGradient()

  The gradient is taken wrt ``points_to_sample[dim]``, so this is the ``input_matrix``, ``X_{d,i}``.
  The other inputs to KG are not differentiated against, so they are taken as input and stored by the constructor.

  The output of KG is a scalar.
\endrst*/
class PingKnowledgeGradient final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "KG with MC integration";

  PingKnowledgeGradient(TensorProductDomain domain, GradientDescentParameters& optimizer_parameters,
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
        grad_KG_(dim_*num_to_sample_),
        sqexp_covariance_(dim_, alpha, lengths),
        gaussian_process_(sqexp_covariance_, points_sampled_.data(), points_sampled_value_.data(), noise_variance_.data(),
                          gradients_.data(), num_gradients_, dim_, num_sampled_),
        kg_evaluator_(gaussian_process_, discrete_pts_.data(), num_pts, num_mc_iter, domain_, optimizer_parameters, best_so_far) {
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
      OL_WARNING_PRINTF("WARNING: grad_KG data already set.  Overwriting...\n");
    }
    gradients_already_computed_ = true;

    NormalRNG normal_rng(3141);
    bool configure_for_gradients = true;

    KnowledgeGradientEvaluator<TensorProductDomain>::StateType kg_state(kg_evaluator_, points_to_sample, points_being_sampled_.data(),
                                                                        num_to_sample_, num_being_sampled_, num_pts_, gradients_.data(),
                                                                        num_gradients_, configure_for_gradients, &normal_rng);

    kg_evaluator_.ComputeGradKnowledgeGradient(&kg_state, grad_KG_.data());

    if (gradients != nullptr) {
      std::copy(grad_KG_.begin(), grad_KG_.end(), gradients);
    }
  }

  virtual double GetAnalyticGradient(int row_index, int column_index, int OL_UNUSED(output_index)) const override OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingKnowledgeGradient::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    return grad_KG_[column_index*dim_ + row_index];
  }

  virtual void EvaluateFunction(double const * restrict points_to_sample, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS {
    NormalRNG normal_rng(3141);
    bool configure_for_gradients = false;

    KnowledgeGradientEvaluator<TensorProductDomain>::StateType kg_state(kg_evaluator_, points_to_sample, points_being_sampled_.data(),
                                                                       num_to_sample_, num_being_sampled_, num_pts_, gradients_.data(),
                                                                       num_gradients_, configure_for_gradients, &normal_rng);
    *function_values = kg_evaluator_.ComputeKnowledgeGradient(&kg_state);
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
  std::vector<double> grad_KG_;

  //! covariance class (for computing covariance and its gradients)
  SquareExponential sqexp_covariance_;
  //! gaussian process used for computations
  GaussianProcess gaussian_process_;
  //! expected improvement evaluator object that specifies the parameters & GP for KG evaluation
  KnowledgeGradientEvaluator<TensorProductDomain> kg_evaluator_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingKnowledgeGradient);
};

/*!\rst
  Supports evaluating the knowledge gradient, KnowledgeGradientEvaluator::ComputeKnowledgeGradient() and
  its gradient, KnowledgeGradientEvaluator::ComputeGradKnowledgeGradient()

  The gradient is taken wrt ``points_to_sample[dim]``, so this is the ``input_matrix``, ``X_{d,i}``.
  The other inputs to KG are not differentiated against, so they are taken as input and stored by the constructor.

  The output of KG is a scalar.
\endrst*/
class PingPosteriorMean final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "Posterior Mean";

  PingPosteriorMean(double const * restrict lengths, double const * restrict points_sampled, double const * restrict points_sampled_value,
                    int const * restrict gradients, double alpha, int dim, int num_to_sample, int num_sampled, int num_gradients) OL_NONNULL_POINTERS
      : dim_(dim),
        num_to_sample_(num_to_sample),
        num_sampled_(num_sampled),
        num_gradients_(num_gradients),
        gradients_already_computed_(false),
        gradients_(gradients, gradients + num_gradients),
        noise_variance_(1+num_gradients, 0.1),
        points_sampled_(points_sampled, points_sampled + dim_*num_sampled_),
        points_sampled_value_(points_sampled_value, points_sampled_value + num_sampled_*(1+num_gradients_)),
        grad_PS_(dim_*num_to_sample_),
        sqexp_covariance_(dim_, alpha, lengths),
        gaussian_process_(sqexp_covariance_, points_sampled_.data(), points_sampled_value_.data(), noise_variance_.data(),
                          gradients_.data(), num_gradients_, dim_, num_sampled_),
        ps_evaluator_(gaussian_process_){
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
      OL_WARNING_PRINTF("WARNING: grad_KG data already set.  Overwriting...\n");
    }
    gradients_already_computed_ = true;

    NormalRNG normal_rng(3141);
    bool configure_for_gradients = true;

    PosteriorMeanEvaluator::StateType ps_state(ps_evaluator_, points_to_sample, configure_for_gradients);

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

    PosteriorMeanEvaluator::StateType ps_state(ps_evaluator_, points_to_sample, configure_for_gradients);
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
  //! function values at points_sampled, ``y``
  std::vector<double> points_sampled_value_;
  //! the gradient of KG at union_of_points, wrt union_of_points[0:num_to_sample]
  std::vector<double> grad_PS_;

  //! covariance class (for computing covariance and its gradients)
  SquareExponential sqexp_covariance_;
  //! gaussian process used for computations
  GaussianProcess gaussian_process_;
  //! expected improvement evaluator object that specifies the parameters & GP for KG evaluation
  PosteriorMeanEvaluator ps_evaluator_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingPosteriorMean);
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
template <typename KGEvaluator>
OL_WARN_UNUSED_RESULT int PingKGTest(int num_to_sample, int num_being_sampled, double epsilon[2],
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
  const double tolerance = 1.0e-1;

  const int max_gradient_descent_steps = 250;
  const int max_num_restarts = 3;
  const int num_steps_averaged = 15;

  GradientDescentParameters gd_params(1, max_gradient_descent_steps, max_num_restarts,
                                      num_steps_averaged, gamma, pre_mult,
                                      max_relative_change, tolerance);
  ClosedInterval * domain_bounds = new ClosedInterval[dim];
  for (int i=0; i<dim; ++i){
      domain_bounds[i] = ClosedInterval(-5.0, 5.0);
  }
  TensorProductDomain domain(domain_bounds, dim);
  // seed randoms
  UniformRandomGenerator uniform_generator(314);

  //UniformRandomGenerator uniform_generator(2718);
  boost::uniform_real<double> uniform_double(0.5, 2.5);

  for (int i = 0; i < 20; ++i) {
    KG_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled, num_gradients);
    //std::vector<double> noise_variance(num_sampled, 0.0003);
    for (int j = 0; j < dim; ++j) {
      lengths[j] = uniform_double(uniform_generator.engine);
    }

    KGEvaluator KG_evaluator(domain, gd_params, lengths.data(), KG_environment.points_being_sampled(), KG_environment.points_sampled(),
                             KG_environment.points_sampled_value(), gradients, alpha, best_so_far, KG_environment.dim,
                             KG_environment.num_to_sample, KG_environment.num_being_sampled, KG_environment.num_sampled,
                             num_mc_iter, num_pts, num_gradients);

    //KGEvaluator KG_evaluator(lengths.data(), KG_environment.points_sampled(), KG_environment.points_sampled_value(), alpha, KG_environment.dim, KG_environment.num_to_sample, KG_environment.num_sampled);
    KG_evaluator.EvaluateAndStoreAnalyticGradient(KG_environment.points_to_sample(), nullptr);

    errors_this_iteration = PingDerivative(KG_evaluator, KG_environment.points_to_sample(), epsilon, tolerance_fine, tolerance_coarse, input_output_ratio);

    if (errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("on iteration %d\n", i);
    }
    total_errors += errors_this_iteration;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s (%d,%d-KG) gradient pings failed with %d errors\n", KGEvaluator::kName, num_to_sample, num_being_sampled, total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("%s (%d,%d-KG) gradient pings passed\n", KGEvaluator::kName, num_to_sample, num_being_sampled);
  }

  delete [] gradients;
  return total_errors;
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

  int * gradients = new int[3]{0, 1, 2};
  int num_gradients = 3;

  std::vector<double> lengths(dim);
  double alpha = 2.80723;

  MockExpectedImprovementEnvironment KG_environment;

  UniformRandomGenerator uniform_generator(2718);
  boost::uniform_real<double> uniform_double(0.5, 2.5);

  for (int i = 0; i < 1; ++i) {
    KG_environment.Initialize(dim, num_to_sample, 0, num_sampled, num_gradients);
    for (int j = 0; j < dim; ++j) {
      lengths[j] = uniform_double(uniform_generator.engine);
    }

    PSEvaluator PS_evaluator(lengths.data(), KG_environment.points_sampled(),
                             KG_environment.points_sampled_value(), gradients, alpha, KG_environment.dim,
                             KG_environment.num_to_sample, KG_environment.num_sampled, num_gradients);

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
  delete [] gradients;

  return total_errors;
};

}  // end unnamed namespace


/*!\rst
  Wrapper to ping the gradients (spatial) of the inverse of the cholesky factorization with noise.

  \return
    number of ping/test failures
\endrst*/

int PingKGGeneralTest() {
  double epsilon_KG[2] = {1.0e-3, 1.0e-4};
  int total_errors = PingKGTest<PingKnowledgeGradient>(2, 0, epsilon_KG, 9.0e-2, 3.0e-1, 1.0e-18);

  total_errors += PingKGTest<PingKnowledgeGradient>(1, 2, epsilon_KG, 9.0e-2, 3.0e-1, 1.0e-18);

  total_errors += PingKGTest<PingKnowledgeGradient>(3, 2, epsilon_KG, 9.0e-2, 3.0e-1, 1.0e-18);

  total_errors += PingKGTest<PingKnowledgeGradient>(10, 0, epsilon_KG, 9.0e-2, 3.0e-1, 1.0e-18);

  total_errors += PingPSTest<PingPosteriorMean>(1, epsilon_KG, 9.0e-2, 3.0e-1, 1.0e-18);

  return total_errors;
}


int RunKGTests() {
  int total_errors = 0;
  int current_errors = 0;

  {
    current_errors = PingKGGeneralTest();
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging KG failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("KG functions failed with %d errors\n\n", total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("KG functions passed\n");
  }

  return total_errors;

}


/*!\rst
  Tests that single & multithreaded KG optimization produce *the exact same* results.

  We do this by first setting up KG optimization in a single threaded scenario with 2 starting points and 2 random number generators.
  Optimization is run one from starting point 0 with RNG 0, and then again from starting point 1 with RNG 1.

  Then we run the optimization multithreaded (with 2 threads) over both starting points simultaneously.  One of the threads
  will see the winning (point, RNG) pair from the single-threaded won.  Hence one result point will match with the single threaded
  results exactly.

  Then we re-run the multithreaded optimization, swapping the position of the RNGs and starting points.  If thread 0 won in the
  previous test, thread 1 will win here (and vice versa).

  Note that it's tricky to run single-threaded optimization over both starting points simultaneously because we won't know which
  (point, RNG) pair won (which is required to ascertain the 'winner' since we are not computing KG accurately enough to avoid
  error).
\endrst*/

/*
int MultithreadedKGOptimizationTest() {
  using DomainType = TensorProductDomain;
  const int num_sampled = 17;
  static const int kDim = 3;

  // q,p-KG computation parameters
  int num_to_sample = 2;
  int num_being_sampled = 0;
  int num_pts = 1000;
  //double noise=0.01;

  int * gradients = new int[3]{0, 1, 2};
  int num_gradients = 3;

  std::vector<double> points_being_sampled(kDim*num_being_sampled);
  std::vector<double> discrete_pts(kDim*num_pts);

  // gradient descent parameters
  const double gamma = 0.7;
  const double pre_mult = 1.0;
  const double max_relative_change = 0.7;
  const double tolerance = 1.0e-1;

  const int max_gradient_descent_steps = 250;
  const int max_num_restarts = 3;
  const int num_steps_averaged = 15;
  GradientDescentParameters gd_params(0, max_gradient_descent_steps, max_num_restarts,
                                      num_steps_averaged, gamma, pre_mult,
                                      max_relative_change, tolerance);

  int max_mc_iterations = 967;

  int total_errors = 0;

  // seed randoms
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(1.0, 2.5);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(2.5, 5.5);

  //std::vector<double> noise_variance(num_sampled, 0.0003);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(kDim, 1.0, 1.0), std::vector<int>(gradients, gradients+num_gradients),
                                                        num_gradients, kDim, num_sampled, uniform_double_lower_bound,
                                                        uniform_double_upper_bound, uniform_double_hyperparameter,
                                                        &uniform_generator);

  for (int j = 0; j < num_being_sampled; ++j) {
    mock_gp_data.domain_ptr->GeneratePointInDomain(&uniform_generator, points_being_sampled.data() + j*kDim);
  }

  for (int j = 0; j < num_pts; ++j) {
    mock_gp_data.domain_ptr->GeneratePointInDomain(&uniform_generator, discrete_pts.data() + j*kDim);
  }

  const int pi_array[] = {314, 3141, 31415, 314159};
  static const int kMaxNumThreads = 2;
  std::vector<NormalRNG> normal_rng_vec(kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[j].SetExplicitSeed(pi_array[j]);
  }

  std::vector<double> starting_points(kDim*kMaxNumThreads*num_to_sample);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    for (int k = 0; k < num_to_sample; ++k) {
      mock_gp_data.domain_ptr->GeneratePointInDomain(&uniform_generator, starting_points.data() + j*kDim*num_to_sample + k*kDim);
    }
  }

  // we will optimize over the expanded region
  std::vector<ClosedInterval> domain_bounds(mock_gp_data.domain_bounds);
  ExpandDomainBounds(3.2, &domain_bounds);
  DomainType domain(domain_bounds.data(), kDim);

  // build truth data by using single threads
  bool found_flag = false;
  std::vector<double> best_next_point_single_thread(kDim*num_to_sample*kMaxNumThreads*kMaxNumThreads);
  int num_threads = 1;
  ThreadSchedule thread_schedule(num_threads, omp_sched_static);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    NormalRNG normal_rng(pi_array[j]);
    int one_multistart = 1;  // truth values come from single threaded execution
    ComputeKGOptimalPointsToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr, gd_params,
                                                               domain, thread_schedule,
                                                               starting_points.data() + j*kDim*num_to_sample,
                                                               points_being_sampled.data(), discrete_pts.data(),
                                                               one_multistart,
                                                               num_to_sample, num_being_sampled, num_pts,
                                                               mock_gp_data.best_so_far, max_mc_iterations,
                                                               &normal_rng, &found_flag,
                                                               best_next_point_single_thread.data() + j*kDim*num_to_sample);
    if (!found_flag) {
      ++total_errors;
    }

    normal_rng.SetExplicitSeed(pi_array[kMaxNumThreads - j - 1]);
    ComputeKGOptimalPointsToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr, gd_params,
                                                               domain, thread_schedule,
                                                               starting_points.data() + j*kDim*num_to_sample,
                                                               points_being_sampled.data(), discrete_pts.data(),
                                                               one_multistart,
                                                               num_to_sample, num_being_sampled, num_pts,
                                                               mock_gp_data.best_so_far, max_mc_iterations,
                                                               &normal_rng, &found_flag,
                                                               best_next_point_single_thread.data() + j*kDim*num_to_sample + kDim*kMaxNumThreads*num_to_sample);
    if (!found_flag) {
      ++total_errors;
    }
  }

  // now multithreaded to generate test data
  std::vector<double> best_next_point_multithread(kDim*num_to_sample);
  thread_schedule.max_num_threads = kMaxNumThreads;
  found_flag = false;
  ComputeKGOptimalPointsToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr, gd_params,
                                                             domain, thread_schedule, starting_points.data(),
                                                             points_being_sampled.data(), discrete_pts.data(), kMaxNumThreads,
                                                             num_to_sample, num_being_sampled, num_pts,
                                                             mock_gp_data.best_so_far,
                                                             max_mc_iterations, normal_rng_vec.data(),
                                                             &found_flag, best_next_point_multithread.data());
  if (!found_flag) {
    ++total_errors;
  }

  // best_next_point_multithread must be PRECISELY one of the points determined by single threaded runs
  double error[kMaxNumThreads*kMaxNumThreads];
  for (int i = 0; i < kMaxNumThreads; ++i) {
    for (int j = 0; j < kMaxNumThreads; ++j) {
      error[i*kMaxNumThreads + j] = 0.0;
      for (int k = 0; k < num_to_sample; ++k) {
        for (int d = 0; d < kDim; ++d) {
          error[i*kMaxNumThreads + j] += std::fabs(best_next_point_multithread[k*kDim + d] -
                                                   best_next_point_single_thread[i*kDim*kMaxNumThreads*num_to_sample +
                                                                                 j*kDim*num_to_sample + k*kDim + d]);
        }
      }
    }
  }
  // normally double precision checks like this are bad
  // but here, we want to ensure that the multithreaded & singlethreaded paths executed THE EXACT SAME CODE IN THE SAME ORDER
  // and hence their results must be identical
  bool pass = false;
  for (int i = 0; i < kMaxNumThreads*kMaxNumThreads; ++i) {
    if (error[i] == 0.0) {
      pass = true;
      break;
    }
  }
  if (pass == false) {
    OL_PARTIAL_FAILURE_PRINTF("multi & single threaded results differ 1: ");
    PrintMatrix(error, 1, Square(kMaxNumThreads));
    ++total_errors;
  }

  // reset random state & flip the points & generators so that if thread 0 won before, thread 1 wins now (or vice versa)
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[kMaxNumThreads-j-1].SetExplicitSeed(pi_array[j]);
  }

  std::vector<double> starting_points_flip(kDim*kMaxNumThreads*num_to_sample);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    for (int k = 0; k < num_to_sample; ++k) {
      for (int d = 0; d < kDim; ++d) {
        starting_points_flip[(kMaxNumThreads-j-1)*kDim*num_to_sample + k*kDim + d] = starting_points[j*kDim*num_to_sample + k*kDim + d];
      }
    }
  }

  // check multithreaded results again
  found_flag = false;
  ComputeKGOptimalPointsToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr, gd_params,
                                                             domain, thread_schedule, starting_points_flip.data(),
                                                             points_being_sampled.data(), discrete_pts.data(), kMaxNumThreads,
                                                             num_to_sample, num_being_sampled, num_pts,
                                                             mock_gp_data.best_so_far,
                                                             max_mc_iterations, normal_rng_vec.data(),
                                                             &found_flag, best_next_point_multithread.data());
  if (!found_flag) {
    ++total_errors;
  }

  for (int i = 0; i < kMaxNumThreads; ++i) {
    for (int j = 0; j < kMaxNumThreads; ++j) {
      error[i*kMaxNumThreads + j] = 0.0;
      for (int k = 0; k < num_to_sample; ++k) {
        for (int d = 0; d < kDim; ++d) {
          error[i*kMaxNumThreads + j] += std::fabs(best_next_point_multithread[k*kDim + d] -
                                                   best_next_point_single_thread[i*kDim*kMaxNumThreads*num_to_sample +
                                                                                 j*kDim*num_to_sample + k*kDim + d]);
        }
      }
    }
  }
  // normally double precision checks like this are bad
  // but here, we want to ensure that the multithreaded & singlethreaded paths executed THE EXACT SAME CODE IN THE SAME ORDER
  // and hence their results must be identical
  pass = false;
  for (int i = 0; i < kMaxNumThreads*kMaxNumThreads; ++i) {
    if (error[i] == 0.0) {
      pass = true;
      break;
    }
  }
  delete [] gradients;

  if (pass == false) {
    OL_PARTIAL_FAILURE_PRINTF("multi & single threaded results differ 2: ");
    PrintMatrix(error, 1, Square(kMaxNumThreads));
    ++total_errors;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("Single/Multithreaded KG Optimization Consistency Check failed with %d errors\n", total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("Single/Multithreaded KG Optimization Consistency Check succeeded\n");
  }

  return total_errors;
}


int EvaluateKGAtPointListTest() {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  int total_errors = 0;

  // grid search parameters
  int num_grid_search_points = 100000;

  // q,p-KG computation parameters
  int num_to_sample = 1;
  int num_being_sampled = 0;
  int max_int_steps = 10;

  int * gradients = new int[3]{0, 1, 2};
  int num_gradients = 3;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.4, 1.3);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(2.0, 3.5);

  const int64_t pi_array[] = {314, 3141, 31415, 314159, 3141592, 31415926, 314159265, 3141592653, 31415926535, 314159265359};
  static const int kMaxNumThreads = 4;
  ThreadSchedule thread_schedule(kMaxNumThreads, omp_sched_static);
  std::vector<NormalRNG> normal_rng_vec(kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[j].SetExplicitSeed(pi_array[j]);
  }

  int num_sampled = 20;  // arbitrary
  //std::vector<double> noise_variance(num_sampled, 0.002);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0), std::vector<int>(gradients, gradients+num_gradients),
                                                        num_gradients, dim, num_sampled, uniform_double_lower_bound,
                                                        uniform_double_upper_bound, uniform_double_hyperparameter,
                                                        &uniform_generator);

  // no parallel experiments
  num_being_sampled = 0;
  int num_pts = 10;
  //double noise=0.01;

  std::vector<double> points_being_sampled(dim*num_being_sampled);
  std::vector<double> discrete_pts(dim*num_pts);

  for (int j = 0; j < num_being_sampled; ++j) {
    mock_gp_data.domain_ptr->GeneratePointInDomain(&uniform_generator, points_being_sampled.data() + j*dim);
  }

  for (int j = 0; j < num_pts; ++j) {
    mock_gp_data.domain_ptr->GeneratePointInDomain(&uniform_generator, discrete_pts.data() + j*dim);
  }

  bool found_flag = false;
  std::vector<double> grid_search_best_point(dim*num_to_sample);
  std::vector<double> function_values(num_grid_search_points);
  std::vector<double> initial_guesses(dim*num_to_sample*num_grid_search_points);
  num_grid_search_points = mock_gp_data.domain_ptr->GenerateUniformPointsInDomain(num_grid_search_points, &uniform_generator, initial_guesses.data());

  EvaluateKGAtPointList(*mock_gp_data.gaussian_process_ptr, thread_schedule, initial_guesses.data(),
                        points_being_sampled.data(), discrete_pts.data(), num_grid_search_points, num_to_sample,
                        num_being_sampled, num_pts, mock_gp_data.best_so_far, max_int_steps, &found_flag,
                        normal_rng_vec.data(), function_values.data(), grid_search_best_point.data());
  if (!found_flag) {
    ++total_errors;
  }

  // find the max function_value and the index at which it occurs
  auto max_value_ptr = std::max_element(function_values.begin(), function_values.end());
  auto max_index = std::distance(function_values.begin(), max_value_ptr);

  // check that EvaluateEIAtPointList found the right point
  for (int i = 0; i < dim*num_to_sample; ++i) {
    if (!CheckDoubleWithin(grid_search_best_point[i], initial_guesses[max_index*dim + i], 0.0)) {
      ++total_errors;
    }
  }

  // now check multi-threaded & single threaded give the same result

  {
    std::vector<double> grid_search_best_point_single_thread(dim*num_to_sample);
    std::vector<double> function_values_single_thread(num_grid_search_points);
    ThreadSchedule single_thread_schedule(1, omp_sched_static);
    found_flag = false;
    EvaluateKGAtPointList(*mock_gp_data.gaussian_process_ptr, single_thread_schedule,
                          initial_guesses.data(), points_being_sampled.data(), discrete_pts.data(),
                          num_grid_search_points, num_to_sample, num_being_sampled, num_pts,
                          mock_gp_data.best_so_far, max_int_steps,
                          &found_flag, normal_rng_vec.data(),
                          function_values_single_thread.data(),
                          grid_search_best_point_single_thread.data());
    delete [] gradients;

    // check against multi-threaded result matches single
    for (int i = 0; i < dim*num_to_sample; ++i) {
      if (!CheckDoubleWithin(grid_search_best_point[i], grid_search_best_point_single_thread[i], 0.0)) {
        ++total_errors;
      }
    }

    // check all function values match too
    for (int i = 0; i < num_grid_search_points; ++i) {
      if (!CheckDoubleWithin(function_values[i], function_values_single_thread[i], 0.0)) {
        ++total_errors;
      }
    }
  }
  return total_errors;
}


namespace {  // contains tests of KG optimization
*/
/*!\rst
  Test that KG optimization works as expected for the monte-carlo evaluator types on a TensorProductDomain.

  \return
    number of test failures (invalid results, unconverged results, etc.)
\endrst*/

/*
OL_WARN_UNUSED_RESULT int KnowledgeGradientOptimizationTestCore() {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  int total_errors = 0;
  int current_errors = 0;

  // gradient descent parameters
  const double gamma = 0.5;
  const double pre_mult = 1.4;
  const double max_relative_change = 1.0;
  const double tolerance = 1.0e-7;
  const int max_gradient_descent_steps = 1000;
  const int max_num_restarts = 10;
  const int num_steps_averaged = 0;
  const int num_multistarts = 20;
  GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps,
                                      max_num_restarts, num_steps_averaged,
                                      gamma, pre_mult, max_relative_change, tolerance);

  // grid search parameters
  int num_grid_search_points = 10000;
  //const double noise = 0.1;

  int num_pts = 10;
  // 1,p-KG computation parameters
  const int num_to_sample = 1;
  int num_being_sampled = 0;
  int max_int_steps = 6000;

  int * gradients = new int[3]{0, 1, 2};
  int num_gradients = 3;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.4, 1.3);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(1.0, 2.5);

  const int64_t pi_array[] = {314, 3141, 31415, 314159, 3141592, 31415926, 314159265, 3141592653, 31415926535, 314159265359};
  static const int kMaxNumThreads = 4;
  ThreadSchedule thread_schedule(kMaxNumThreads, omp_sched_static);
  std::vector<NormalRNG> normal_rng_vec(kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[j].SetExplicitSeed(pi_array[j]);
  }

  int num_sampled = 20;

  std::vector<double> noise_variance(num_sampled, 0.002);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0), std::vector<int>(gradients, gradients+num_gradients),
                                                        num_gradients, dim, num_sampled, uniform_double_lower_bound,
                                                        uniform_double_upper_bound, uniform_double_hyperparameter,
                                                        &uniform_generator);

  // we will optimize over the expanded region
  std::vector<ClosedInterval> domain_bounds(mock_gp_data.domain_bounds);
  ExpandDomainBounds(1.5, &domain_bounds);
  DomainType domain(domain_bounds.data(), dim);

  // set up parallel experiments, if any

  // using MC integration
  num_being_sampled = 2;

  gd_params.max_num_restarts = 3;
  gd_params.max_num_steps = 250;
  gd_params.tolerance = 1.0e-5;

  std::vector<double> discrete_pts(dim*num_pts);
  for (int j = 0; j < num_pts; ++j) {
    mock_gp_data.domain_ptr->GeneratePointInDomain(&uniform_generator, discrete_pts.data() + j*dim);
  }

  std::vector<double> points_being_sampled(dim*num_being_sampled);

  // generate two non-trivial parallel samples
  // picking these randomly could place them in regions where EI is 0, which means errors in the computation would
  // likely be masked (making for a bad test)
  bool found_flag = false;
  for (int j = 0; j < num_being_sampled; ++j) {
    ComputeKGOptimalPointsToSampleWithRandomStarts(*mock_gp_data.gaussian_process_ptr, gd_params,
                                                   domain, thread_schedule, points_being_sampled.data(), discrete_pts.data(),
                                                   num_to_sample, j, num_pts,
                                                   mock_gp_data.best_so_far,
                                                   max_int_steps, &found_flag,
                                                   &uniform_generator, normal_rng_vec.data(),
                                                   points_being_sampled.data() + j*dim);
  }
  printf("setup complete, points_being_sampled:\n");
  PrintMatrixTrans(points_being_sampled.data(), num_being_sampled, dim);


  // optimize KG
  found_flag = false;
  std::vector<double> grid_search_best_point(dim*num_to_sample);
  ComputeKGOptimalPointsToSampleViaLatinHypercubeSearch(*mock_gp_data.gaussian_process_ptr, domain,
                                                        thread_schedule, points_being_sampled.data(), discrete_pts.data(),
                                                        num_grid_search_points, num_to_sample,
                                                        num_being_sampled, num_pts, mock_gp_data.best_so_far,
                                                        max_int_steps, &found_flag,
                                                        &uniform_generator, normal_rng_vec.data(),
                                                        grid_search_best_point.data());
  if (!found_flag) {
    ++total_errors;
  }

  std::vector<double> next_point(dim*num_to_sample);

  int num_multistarts_mc = 8;
  gd_params.num_multistarts = num_multistarts_mc;
  found_flag = false;
  std::vector<double> initial_guesses(num_multistarts_mc*dim);
  domain.GenerateUniformPointsInDomain(num_multistarts_mc - 1, &uniform_generator, initial_guesses.data() + dim);
  std::copy(grid_search_best_point.begin(), grid_search_best_point.end(), initial_guesses.begin());

  ComputeKGOptimalPointsToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr,
                                                             gd_params, domain, thread_schedule,
                                                             initial_guesses.data(),
                                                             points_being_sampled.data(), discrete_pts.data(),
                                                             num_multistarts_mc, num_to_sample,
                                                             num_being_sampled, num_pts,
                                                             mock_gp_data.best_so_far,
                                                             max_int_steps,
                                                             normal_rng_vec.data(), &found_flag,
                                                             next_point.data());
  if (!found_flag) {
    ++total_errors;
  }


  printf("next best point  : "); PrintMatrixTrans(next_point.data(), num_to_sample, dim);
  printf("grid search point: "); PrintMatrixTrans(grid_search_best_point.data(), num_to_sample, dim);

  // results
  double kg_optimized, kg_grid_search;
  std::vector<double> grad_kg(dim*num_to_sample);

  // set up evaluators and state to check results
  double tolerance_result = tolerance;
  bool configure_for_gradients = true;

  max_int_steps = 1000000;
  tolerance_result = 2.0e-3;  // reduce b/c we cannot achieve full accuracy in the monte-carlo case
  // while still having this test run in a reasonable amt of time
  //noise = 0.1;
  KnowledgeGradientEvaluator kg_evaluator(*mock_gp_data.gaussian_process_ptr, discrete_pts.data(), num_pts,
                                          max_int_steps, mock_gp_data.best_so_far);
  KnowledgeGradientEvaluator::StateType kg_state(kg_evaluator, next_point.data(),
                                                 points_being_sampled.data(), num_to_sample,
                                                 num_being_sampled, num_pts, gradients, num_gradients, configure_for_gradients,
                                                 normal_rng_vec.data());

  kg_optimized = kg_evaluator.ComputeKnowledgeGradient(&kg_state);
  kg_evaluator.ComputeGradKnowledgeGradient(&kg_state, grad_kg.data());

  kg_state.SetCurrentPoint(kg_evaluator, grid_search_best_point.data());
  kg_grid_search = kg_evaluator.ComputeKnowledgeGradient(&kg_state);

  delete [] gradients;
  printf("optimized KG: %.18E, grid_search_KG: %.18E\n", kg_optimized, kg_grid_search);
  printf("grad_KG: "); PrintMatrixTrans(grad_kg.data(), num_to_sample, dim);

  if (kg_optimized < kg_grid_search) {
    ++total_errors;
  }

  current_errors = 0;
  for (const auto& entry : grad_kg) {
    if (!CheckDoubleWithinRelative(entry, 0.0, tolerance_result)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  return total_errors;
}
*/
/*!\rst
  Test that KG optimization works as expected for the monte-carlo evaluator types on a SimplexIntersectTensorProductDomain.

  \return
    number of test failures (invalid results, unconverged results, etc.)
\endrst*/

/*
OL_WARN_UNUSED_RESULT int KnowledgeGradientOptimizationSimplexTestCore() {
  using DomainType = SimplexIntersectTensorProductDomain;
  const int dim = 3;

  int total_errors = 0;
  int current_errors = 0;

  // gradient descent parameters
  const double gamma = 0.8;
  const double pre_mult = 0.02;
  const double max_relative_change = 0.99;
  const double tolerance = 1.0e-7;
  const int max_gradient_descent_steps = 1000;
  const int max_num_restarts = 10;
  const int num_steps_averaged = 0;
  const int num_multistarts = 20;
  GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps,
                                      max_num_restarts, num_steps_averaged, gamma, pre_mult,
                                      max_relative_change, tolerance);

  // grid search parameters
  int num_grid_search_points = 10000;
  const double noise = 0.1;

  int num_pts = 10;

  // 1,p-KG computation parameters
  const int num_to_sample = 1;
  int num_being_sampled = 0;
  int max_int_steps = 6000;

  int * gradients = new int[3]{0, 1, 2};
  int num_gradients = 3;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.05, 0.1);
  boost::uniform_real<double> uniform_double_lower_bound(0.11, 0.15);
  boost::uniform_real<double> uniform_double_upper_bound(0.3, 0.35);

  const int64_t pi_array[] = {314, 3141, 31415, 314159, 3141592, 31415926, 314159265, 3141592653, 31415926535, 314159265359};
  static const int kMaxNumThreads = 4;
  ThreadSchedule thread_schedule(kMaxNumThreads, omp_sched_static);
  std::vector<NormalRNG> normal_rng_vec(kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[j].SetExplicitSeed(pi_array[j]);
  }

  int num_sampled = 20;

  //std::vector<double> noise_variance(num_sampled, 0.002);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0), std::vector<int>(gradients, gradients+num_gradients),
                                                        num_gradients, dim, num_sampled, uniform_double_lower_bound,
                                                        uniform_double_upper_bound, uniform_double_hyperparameter,
                                                        &uniform_generator);

  // we will optimize over the expanded region
  std::vector<ClosedInterval> domain_bounds(mock_gp_data.domain_bounds);
  ExpandDomainBounds(2.2, &domain_bounds);
  // intersect domain with bounding box of unit simplex
  for (auto& interval : domain_bounds) {
    interval.min = std::fmax(interval.min, 0.0);
    interval.max = std::fmin(interval.max, 1.0);
  }
  DomainType domain(domain_bounds.data(), dim);

  // using MC integration
  num_being_sampled = 2;

  gd_params.max_num_restarts = 4;
  gd_params.max_num_steps = 250;
  gd_params.tolerance = 1.0e-4;

  std::vector<double> discrete_pts(dim*num_pts);
  for (int j = 0; j < num_pts; ++j) {
    mock_gp_data.domain_ptr->GeneratePointInDomain(&uniform_generator, discrete_pts.data() + j*dim);
  }

  std::vector<double> points_being_sampled(dim*num_being_sampled);

  // generate two non-trivial parallel samples
  // picking these randomly could place them in regions where KG is 0, which means errors in the computation would
  // likely be masked (making for a bad test)
  bool found_flag = false;
  for (int j = 0; j < num_being_sampled; ++j) {
    ComputeKGOptimalPointsToSampleWithRandomStarts(*mock_gp_data.gaussian_process_ptr, gd_params,
                                                   domain, thread_schedule, points_being_sampled.data(), discrete_pts.data(),
                                                   num_to_sample, j, num_pts,
                                                   mock_gp_data.best_so_far,
                                                   max_int_steps, &found_flag,
                                                   &uniform_generator, normal_rng_vec.data(),
                                                   points_being_sampled.data() + j*dim);
  }
  printf("setup complete, points_being_sampled:\n");
  PrintMatrixTrans(points_being_sampled.data(), num_being_sampled, dim);

  // optimize KG
  found_flag = false;
  std::vector<double> grid_search_best_point(dim*num_to_sample);
  ComputeKGOptimalPointsToSampleViaLatinHypercubeSearch(*mock_gp_data.gaussian_process_ptr, domain,
                                                        thread_schedule, points_being_sampled.data(), discrete_pts.data(),
                                                        num_grid_search_points, num_to_sample,
                                                        num_being_sampled, num_pts, mock_gp_data.best_so_far,
                                                        max_int_steps, &found_flag,
                                                        &uniform_generator, normal_rng_vec.data(),
                                                        grid_search_best_point.data());
  if (!found_flag) {
    ++total_errors;
  }

  std::vector<double> next_point(dim*num_to_sample);

  int num_multistarts_mc = 8;
  gd_params.num_multistarts = num_multistarts_mc;
  found_flag = false;
  std::vector<double> initial_guesses(num_multistarts_mc*dim);
  int num_points_actual = domain.GenerateUniformPointsInDomain(num_multistarts_mc, &uniform_generator, initial_guesses.data());
  if (num_points_actual != num_multistarts_mc) {
    ++total_errors;
  }
  std::copy(grid_search_best_point.begin(), grid_search_best_point.end(), initial_guesses.begin());

  ComputeKGOptimalPointsToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr,
                                                             gd_params, domain, thread_schedule,
                                                             initial_guesses.data(),
                                                             points_being_sampled.data(), discrete_pts.data(),
                                                             num_multistarts_mc, num_to_sample,
                                                             num_being_sampled, num_pts,
                                                             mock_gp_data.best_so_far,
                                                             max_int_steps,
                                                             normal_rng_vec.data(), &found_flag,
                                                             next_point.data());
  if (!found_flag) {
    ++total_errors;
  }

  printf("next best point  : "); PrintMatrixTrans(next_point.data(), num_to_sample, dim);
  printf("grid search point: "); PrintMatrixTrans(grid_search_best_point.data(), num_to_sample, dim);

  // results
  double kg_optimized, kg_grid_search;
  std::vector<double> grad_kg(dim*num_to_sample);

  // set up evaluators and state to check results
  double tolerance_result = tolerance;
  bool configure_for_gradients = true;

  max_int_steps = 1000000;
  tolerance_result = 3.8e-3;  // reduce b/c we cannot achieve full accuracy in the monte-carlo case
  // while still having this test run in a reasonable amt of time
  //noise = 0.1;
  KnowledgeGradientEvaluator kg_evaluator(*mock_gp_data.gaussian_process_ptr, discrete_pts.data(), num_pts,
                                          max_int_steps, mock_gp_data.best_so_far);
  KnowledgeGradientEvaluator::StateType kg_state(kg_evaluator, next_point.data(),
                                                 points_being_sampled.data(), num_to_sample,
                                                 num_being_sampled, num_pts, gradients, num_gradients, configure_for_gradients,
                                                 normal_rng_vec.data());
  kg_optimized = kg_evaluator.ComputeKnowledgeGradient(&kg_state);
  kg_evaluator.ComputeGradKnowledgeGradient(&kg_state, grad_kg.data());

  kg_state.SetCurrentPoint(kg_evaluator, grid_search_best_point.data());
  kg_grid_search = kg_evaluator.ComputeKnowledgeGradient(&kg_state);

  delete [] gradients;
  printf("optimized KG: %.18E, grid_search_KG: %.18E\n", kg_optimized, kg_grid_search);
  printf("grad_KG: "); PrintMatrixTrans(grad_kg.data(), num_to_sample, dim);

  if (kg_optimized < kg_grid_search) {
    ++total_errors;
  }

  current_errors = 0;
  for (const auto& entry : grad_kg) {
    if (!CheckDoubleWithinRelative(entry, 0.0, tolerance_result)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  return total_errors;
}

}  // end unnamed namespace

int KnowledgeGradientOptimizationTest(DomainTypes domain_type) {
  switch (domain_type) {
    case DomainTypes::kTensorProduct: {
      return KnowledgeGradientOptimizationTestCore();
    }  // end case kTensorProduct
    case DomainTypes::kSimplex: {
      return KnowledgeGradientOptimizationSimplexTestCore();
    }  // end case kSimplex
    default: {
      OL_ERROR_PRINTF("%s: INVALID domain_type choice: %d\n", OL_CURRENT_FUNCTION_NAME, domain_type);
      return 1;
    }
  }  // end switch over domain_type
}
*/
/*!\rst
  At the moment, this test is very bare-bones.  It checks:

  1. method succeeds
  2. points returned are all inside the specified domain
  3. points returned are not within epsilon of each other (i.e., distinct)
  4. result of gradient-descent optimization is *no worse* than result of a random search
  5. final grad EI is sufficiently small

  The test sets up a toy problem by repeatedly drawing from a GP with made-up hyperparameters.
  Then it runs KG optimization, attempting to sample 3 points simultaneously.
\endrst*/

/*
int KnowledgeGradientOptimizationMultipleSamplesTest() {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  int total_errors = 0;
  int current_errors = 0;

  // gradient descent parameters
  const double gamma = 0.5;
  const double pre_mult = 1.5;
  const double max_relative_change = 1.0;
  const double tolerance = 1.0e-5;
  const int max_gradient_descent_steps = 250;
  const int max_num_restarts = 3;
  const int num_steps_averaged = 0;
  const int num_multistarts = 20;
  GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps,
                                      max_num_restarts, num_steps_averaged, gamma, pre_mult,
                                      max_relative_change, tolerance);

  // grid search parameters
  int num_grid_search_points = 1000;
  const double noise = 0.1;

  int num_pts = 10;

  // q,p-KG computation parameters
  const int num_to_sample = 3;
  const int num_being_sampled = 0;

  int * gradients = new int[3]{0, 1, 2};
  int num_gradients = 3;

  std::vector<double> points_being_sampled(dim*num_being_sampled);

  int max_int_steps = 6000;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.4, 1.3);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(1.0, 2.5);

  const int64_t pi_array[] = {314, 3141, 31415, 314159, 3141592, 31415926, 314159265, 3141592653, 31415926535, 314159265359};
  static const int kMaxNumThreads = 4;
  ThreadSchedule thread_schedule(kMaxNumThreads, omp_sched_static);
  std::vector<NormalRNG> normal_rng_vec(kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[j].SetExplicitSeed(pi_array[j]);
  }

  const int num_sampled = 20;
  //std::vector<double> noise_variance(num_sampled, 0.002);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0), std::vector<int>(gradients, gradients+num_gradients),
                                                        num_gradients, dim, num_sampled, uniform_double_lower_bound,
                                                        uniform_double_upper_bound, uniform_double_hyperparameter,
                                                        &uniform_generator);

  // we will optimize over the expanded region
  std::vector<ClosedInterval> domain_bounds(mock_gp_data.domain_bounds);
  ExpandDomainBounds(1.5, &domain_bounds);
  DomainType domain(domain_bounds.data(), dim);

  std::vector<double> discrete_pts(dim*num_pts);
  int num_points_actual = domain.GenerateUniformPointsInDomain(num_pts, &uniform_generator, discrete_pts.data());

  // optimize KG using grid search to set the baseline
  bool found_flag = false;
  std::vector<double> grid_search_best_point_set(dim*num_to_sample);
  ComputeKGOptimalPointsToSampleViaLatinHypercubeSearch(*mock_gp_data.gaussian_process_ptr, domain,
                                                        thread_schedule, points_being_sampled.data(), discrete_pts.data(),
                                                        num_grid_search_points, num_to_sample,
                                                        num_being_sampled, num_pts, mock_gp_data.best_so_far,
                                                        max_int_steps, &found_flag,
                                                        &uniform_generator, normal_rng_vec.data(),
                                                        grid_search_best_point_set.data());
  if (!found_flag) {
    ++total_errors;
  }

  // optimize KG using gradient descent
  found_flag = false;
  bool lhc_search_only = false;
  std::vector<double> best_points_to_sample(dim*num_to_sample);
  ComputeKGOptimalPointsToSample(*mock_gp_data.gaussian_process_ptr, gd_params, domain,
                                 thread_schedule, points_being_sampled.data(), discrete_pts.data(),
                                 num_to_sample, num_being_sampled, num_pts, mock_gp_data.best_so_far,
                                 max_int_steps, lhc_search_only,
                                 num_grid_search_points, &found_flag, &uniform_generator,
                                 normal_rng_vec.data(), best_points_to_sample.data());
  if (!found_flag) {
    ++total_errors;
  }

  // check points are in domain
  RepeatedDomain<DomainType> repeated_domain(domain, num_to_sample);
  if (!repeated_domain.CheckPointInside(best_points_to_sample.data())) {
    ++current_errors;
  }
#ifdef OL_ERROR_PRINT
  if (current_errors != 0) {
    OL_ERROR_PRINTF("ERROR: points were not in domain!  points:\n");
    PrintMatrixTrans(best_points_to_sample.data(), num_to_sample, dim);
    OL_ERROR_PRINTF("domain:\n");
    PrintDomainBounds(domain_bounds.data(), dim);
  }
#endif
  total_errors += current_errors;

  // check points are distinct; points within tolerance are considered non-distinct
  const double distinct_point_tolerance = 1.0e-5;
  current_errors = CheckPointsAreDistinct(best_points_to_sample.data(), num_to_sample, dim, distinct_point_tolerance);
#ifdef OL_ERROR_PRINT
  if (current_errors != 0) {
    OL_ERROR_PRINTF("ERROR: points were not distinct!  points:\n");
    PrintMatrixTrans(best_points_to_sample.data(), num_to_sample, dim);
  }
#endif
  total_errors += current_errors;

  // results
  double kg_optimized, kg_grid_search;
  std::vector<double> grad_kg(dim*num_to_sample);

  // set up evaluators and state to check results
  double tolerance_result = tolerance;
  {
    max_int_steps = 1000000;  // evaluate the final results with high accuracy
    tolerance_result = 2.0e-3;  // reduce b/c we cannot achieve full accuracy in the monte-carlo case
    // while still having this test run in a reasonable amt of time
    bool configure_for_gradients = true;
    KnowledgeGradientEvaluator kg_evaluator(*mock_gp_data.gaussian_process_ptr, discrete_pts.data(), num_pts,
                                            max_int_steps, mock_gp_data.best_so_far);
    KnowledgeGradientEvaluator::StateType kg_state(kg_evaluator, best_points_to_sample.data(),
                                                   points_being_sampled.data(), num_to_sample,
                                                   num_being_sampled, num_pts, gradients, num_gradients, configure_for_gradients,
                                                   normal_rng_vec.data());
    kg_optimized = kg_evaluator.ComputeKnowledgeGradient(&kg_state);
    kg_evaluator.ComputeGradKnowledgeGradient(&kg_state, grad_kg.data());

    KnowledgeGradientEvaluator::StateType kg_state_grid_search(kg_evaluator,
                                                               grid_search_best_point_set.data(),
                                                               points_being_sampled.data(), num_to_sample,
                                                               num_being_sampled, num_pts, gradients, num_gradients, configure_for_gradients,
                                                               normal_rng_vec.data());
    kg_grid_search = kg_evaluator.ComputeKnowledgeGradient(&kg_state_grid_search);
  }

  printf("optimized KG: %.18E, grid_search_KG: %.18E\n", kg_optimized, kg_grid_search);
  printf("grad_KG: "); PrintMatrixTrans(grad_kg.data(), num_to_sample, dim);
  delete [] gradients;

  if (kg_optimized < kg_grid_search) {
    ++total_errors;
  }

  current_errors = 0;
  for (const auto& entry : grad_kg) {
    if (!CheckDoubleWithinRelative(entry, 0.0, tolerance_result)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  return total_errors;
}
*/

}  // end namespace optimal_learning
