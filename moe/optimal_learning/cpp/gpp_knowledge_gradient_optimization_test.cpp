/*!
  \file gpp_knowledge_gradient_optimization_test.cpp
  \rst
  Routines to test the functions in gpp_knowledge_gradient_optimization.cpp.

  The tests verify KnowledgeGradientEvaluator, and KG optimization from gpp_knowledge_gradient_optimization.cpp.

  1. Ping testing (verifying analytic gradient computation against finite difference approximations)

     a. Following gpp_covariance_test.cpp, we define class PingKnowledgeGradient for
        evaluating those functions + their spatial gradients.

     b. Ping for derivative accuracy (PingGPComponentTest, PingEITest); these unit test the analytic derivatives.

  2. Monte-Carlo EI vs analytic EI validation: the monte-carlo versions are run to "high" accuracy and checked against
     analytic formulae when applicable
  3. Gradient Descent: using polynomials and other simple fucntions with analytically known optima
     to verify that the algorithm(s) underlying EI optimization are performing correctly.
  4. Single-threaded vs multi-threaded EI optimization validation: single and multi-threaded runs are checked to have the same
     output.
  5. End-to-end test of the EI optimization process for the analytic and monte-carlo cases.  These tests use constructed
     data for inputs but otherwise exercise the same code paths used for EI optimization in production.
\endrst*/

// #define OL_VERBOSE_PRINT

#include "gpp_knowledge_gradient_optimization_test.hpp"

#include <cmath>
#include <cstdio>

#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)
#include <omp.h>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_domain.hpp"
#include "gpp_exception.hpp"
#include "gpp_geometry.hpp"
#include "gpp_linear_algebra.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_mock_optimization_objective_functions.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimizer_parameters.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"
#include "gpp_knowledge_gradient_optimization.hpp"

namespace optimal_learning {

MockKnowledgeGradientEnvironment::MockKnowledgeGradientEnvironment()
    : dim(-1),
      num_sampled(-1),
      num_to_sample(-1),
      num_being_sampled(-1),
      num_pts(-1),
      noise(0.0),
      points_sampled_(20*4),
      points_sampled_value_(20),
      points_to_sample_(4),
      points_being_sampled_(20*4),
      discrete_pts_(20*4),
      uniform_generator_(kDefaultSeed),
      uniform_double_(range_min, range_max) {
}

void MockKnowledgeGradientEnvironment::Initialize(int dim_in, int num_to_sample_in, int num_being_sampled_in,
                                                    int num_sampled_in, int num_pts_in, double noise_in, UniformRandomGenerator * uniform_generator) {
  if (dim_in != dim || num_to_sample_in != num_to_sample || num_being_sampled_in != num_being_sampled || num_sampled_in != num_sampled || num_pts_in != num_pts
     || noise_in != noise) {
    dim = dim_in;
    num_to_sample = num_to_sample_in;
    num_being_sampled = num_being_sampled_in;
    num_sampled = num_sampled_in;
    num_pts = num_pts_in;

    points_sampled_.resize(num_sampled*dim);
    points_sampled_value_.resize(num_sampled);
    points_to_sample_.resize(num_to_sample*dim);
    points_being_sampled_.resize(num_being_sampled*dim);
    discrete_pts_.resize(num_pts*dim);

    noise = noise_in;
  }

  for (int i = 0; i < dim*num_sampled; ++i) {
    points_sampled_[i] = uniform_double_(uniform_generator->engine);
  }

  for (int i = 0; i < num_sampled; ++i) {
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

/*!\rst
  Supports evaluating the knowledge gradient, KnowledgeGradientEvaluator::ComputeKnowledgeGradient() and
  its gradient, KnowledgeGradientEvaluator::ComputeGradKnowledgeGradient()

  The gradient is taken wrt ``points_to_sample[dim]``, so this is the ``input_matrix``, ``X_{d,i}``.
  The other inputs to EI are not differentiated against, so they are taken as input and stored by the constructor.

  The output of KG is a scalar.
\endrst*/
class PingKnowledgeGradient final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "KG with MC integration";

  PingKnowledgeGradient(double const * restrict lengths, double const * restrict points_being_sampled,
                        double const * restrict points_sampled, double const * restrict points_sampled_value,
                        double alpha, double best_so_far, int dim, int num_to_sample, int num_being_sampled,
                        int num_sampled, int num_mc_iter, double * discrete_pts, int num_pts, double noise) OL_NONNULL_POINTERS
      : dim_(dim),
        num_to_sample_(num_to_sample),
        num_being_sampled_(num_being_sampled),
        num_sampled_(num_sampled),
        num_pts_(num_pts),
        gradients_already_computed_(false),
        noise_variance_(num_sampled_, 0.0),
        points_sampled_(points_sampled, points_sampled + dim_*num_sampled_),
        points_sampled_value_(points_sampled_value, points_sampled_value + num_sampled_),
        points_being_sampled_(points_being_sampled, points_being_sampled + num_being_sampled_*dim_),
        discrete_pts_(discrete_pts, discrete_pts + num_pts *dim_),
        grad_KG_(num_to_sample_*dim_),
        sqexp_covariance_(dim_, alpha, lengths),
        gaussian_process_(sqexp_covariance_, points_sampled_.data(), points_sampled_value_.data(), noise_variance_.data(), dim_, num_sampled_),
        kg_evaluator_(gaussian_process_, discrete_pts, num_pts, num_mc_iter, noise, best_so_far) {
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

    KnowledgeGradientEvaluator::StateType kg_state(kg_evaluator_, points_to_sample, points_being_sampled_.data(),
                                                   num_to_sample_, num_being_sampled_, num_pts_,
                                                   configure_for_gradients, &normal_rng);

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

    KnowledgeGradientEvaluator::StateType kg_state(kg_evaluator_, points_to_sample, points_being_sampled_.data(),
                                                   num_to_sample_, num_being_sampled_, num_pts_,
                                                   configure_for_gradients, &normal_rng);

    *function_values = kg_evaluator_.ComputeKnowledgeGradient(&kg_state);
  }

 private:
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  int dim_;
  //! number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
  int num_to_sample_;
  //! number of points being sampled concurrently (i.e., the "p" in q,p-EI)
  int num_being_sampled_;
  //! number of points in ``points_sampled``
  int num_sampled_;
  //! number of points in ``discret_pts''
  int num_pts_;
  //! whether gradients been computed and stored--whether this class is ready for use
  bool gradients_already_computed_;

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
  //! expected improvement evaluator object that specifies the parameters & GP for EI evaluation
  KnowledgeGradientEvaluator kg_evaluator_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingKnowledgeGradient);
};


/*!\rst
  Pings the gradients (spatial) of the EI 50 times with randomly generated test cases
  Works with various EI evaluators (e.g., MC, analytic formulae)

  \param
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled in concurrent experiments (i.e., the "p" in q,p-EI)
    :epsilon: coarse, fine ``h`` sizes to use in finite difference computation
    :tolerance_fine: desired amount of deviation from the exact rate
    :tolerance_coarse: maximum allowable abmount of deviation from the exact rate
    :input_output_ratio: for ``||analytic_gradient||/||input|| < input_output_ratio``, ping testing is not performed, see PingDerivative()
  \return
    number of ping/test failures
\endrst*/
template <typename KGEvaluator>
OL_WARN_UNUSED_RESULT int PingKGTest(int num_to_sample, int num_being_sampled, double epsilon[2], double tolerance_fine, double tolerance_coarse, double input_output_ratio) {
  int total_errors = 0;
  int errors_this_iteration;
  const int dim = 3;

  int num_sampled = 7;

  int num_pts = 10;

  double noise = 0.1;

  std::vector<double> lengths(dim);
  double alpha = 2.80723;
  // set best_so_far to be larger than max(points_sampled_value) (but don't make it huge or stability will be suffer)
  double best_so_far = 7.0;
  const int num_mc_iter = 16;

  MockKnowledgeGradientEnvironment KG_environment;

  UniformRandomGenerator uniform_generator(2718);
  boost::uniform_real<double> uniform_double(0.5, 2.5);

  for (int i = 0; i < 50; ++i) {
    KG_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled, num_pts, noise);
    for (int j = 0; j < dim; ++j) {
      lengths[j] = uniform_double(uniform_generator.engine);
    }
    KGEvaluator KG_evaluator(lengths.data(), KG_environment.points_being_sampled(), KG_environment.points_sampled(),
                             KG_environment.points_sampled_value(), alpha, best_so_far, KG_environment.dim,
                             KG_environment.num_to_sample, KG_environment.num_being_sampled, KG_environment.num_sampled,
                             num_mc_iter, KG_environment.discrete_pts(), KG_environment.num_pts, KG_environment.noise);
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

  return total_errors;
}

/*!\rst
  Pings the gradients (spatial) of the EI 50 times with randomly generated test cases

  \return
    number of ping/test failures
\endrst*/
int PingKGGeneralTest() {
  double epsilon_KG[2] = {1.0e-2, 1.0e-3};
  int total_errors = PingKGTest<PingKnowledgeGradient>(1, 0, epsilon_KG, 2.0e-3, 9.0e-2, 1.0e-18);
/*
  total_errors += PingKGTest<PingKnowledgeGradient>(1, 5, epsilon_KG, 2.0e-3, 9.0e-2, 1.0e-18);

  total_errors += PingKGTest<PingKnowledgeGradient>(3, 2, epsilon_KG, 2.0e-3, 9.0e-2, 1.0e-18);

  total_errors += PingKGTest<PingKnowledgeGradient>(4, 0, epsilon_KG, 2.0e-3, 9.0e-2, 1.0e-18);
*/
  return total_errors;
}

}  // end namespace optimal_learning
