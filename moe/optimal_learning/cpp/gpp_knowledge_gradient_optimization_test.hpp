/*!
  \file gpp_knowledge_gradient_optimization_test.hpp
  \rst
  Functions for testing KG functionality.

  Tests are broken into two main groups:

  * ping (unit) tests for KG
  * unit + integration tests for optimization methods

  The ping tests are set up the same way as the ping tests in gpp_covariance_test; using the function evaluator and ping
  framework defined in gpp_test_utils.

  Finally, we have tests for KG optimization.  These include multithreading tests (verifying that each core
  does what is expected) as well as integration tests for EI optimization.  Unit tests for optimizers live in
  gpp_optimization_test.hpp/cpp.  These integration tests use constructed data but exercise all the
  same code paths used for hyperparameter optimization in production.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_KNOWLEDGE_GRADIENT_TEST_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_KNOWLEDGE_GRADIENT_TEST_HPP_

#include <memory>
#include <vector>

#include "gpp_common.hpp"
#include "gpp_domain.hpp"

namespace optimal_learning {

/*!\rst
  Checks that the gradients (spatial) of Knowledge Gradient are computed correctly.

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int PingKGGeneralTest();


/*!\rst
  Checks that the gradients (spatial) of Knowledge Gradient are computed correctly.

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int RunKGTests();


/*!\rst
  Checks that multithreaded KG optimization behaves the same way that single threaded does.

  \return
    number of test failures: 0 if KG multi/single threaded optimization are consistent
\endrst*/
 //OL_WARN_UNUSED_RESULT int MultithreadedKGOptimizationTest();

/*!\rst
  Checks that KG optimization is working on tensor product or simplex domain using
  monte-carlo KG evaluation.

  \param
    :domain_type: type of the domain to test on (e.g., tensor product, simplex)
  \return
    number of test failures: 0 if KG optimization is working properly
\endrst*/
 //OL_WARN_UNUSED_RESULT int KnowledgeGradientOptimizationTest(DomainTypes domain_type);

/*!\rst
  Checks that ComputeKGOptimalPointsToSample works on a tensor product domain.
  This test exercises the the code tested in:
  KnowledgeGradientOptimizationTest(kTensorProduct)
  for ``ei_mode = {kAnalytic, kMonteCarlo}``.

  This test checks the generation of multiple, simultaneous experimental points to sample.

  \return
    number of test failures: 0 if KG optimization is working properly
\endrst*/
 //OL_WARN_UNUSED_RESULT int KnowledgeGradientOptimizationMultipleSamplesTest();

/*!\rst
  Tests EvaluateKGAtPointList (computes KG at a specified list of points, multithreaded).
  Checks that the returned best point is in fact the best.
  Verifies multithreaded consistency.

  \return
    number of test failures: 0 if function evaluation is working properly
\endrst*/
 //OL_WARN_UNUSED_RESULT int EvaluateKGAtPointListTest();

/*!\rst
  Class to conveniently hold and generate random data that are commonly needed for testing functions in gpp_math.cpp.  In
  particular, this mock is used for testing GP mean, GP variance, and expected improvement (and their gradients).

  This class holds arrays: ``points_sampled``, ``points_sampled_value``, ``points_to_sample``, and ``points_being_sampled``
  which are sized according to the parameters specified in Initialize(), and filled with random numbers.

  TODO(GH-125): we currently generate the point sets by repeated calls to rand().  This is generally
  unwise since the distribution of points is not particularly random.  Additionally, our current covariance
  functions are all stationary, so we would rather generate a random base point ``x``, and then a random
  (direction, radius) pair so that ``y = x + direction*radius``. We better cover the different behavioral
  regimes of our code in this case, since it's the radius value that actually correlates to results.
\endrst*/
class MockKnowledgeGradientEnvironment {
 public:
  using EngineType = UniformRandomGenerator::EngineType;

  //! default seed for repeatability in testing
  static constexpr EngineType::result_type kDefaultSeed = 314;
  //! minimum coordinate value
  static constexpr double range_min = -5.0;
  //! maximum coordinate value
  static constexpr double range_max = 5.0;

  /*!\rst
    Construct a MockExpectedImprovementEnvironment and set invalid values for all size parameters
    (so that Initialize must be called to do anything useful) and pre-allocate some space.
  \endrst*/
  MockKnowledgeGradientEnvironment();

  /*!\rst
    (Re-)initializes the data data in this function: this includes space allocation and random number generation.

    If any of the size parameters are changed from their current values, space will be realloc'd.
    Then it re-draws another set of uniform random points (in [-5, 5]) for the member arrays
    points_sampled, points_sampled_value, points_to_sample, and points_being_sampled.

    \param
      :dim: the spatial dimension of a point (i.e., number of independent params in experiment)
      :num_to_sample: number of points to be sampled in future experiments
      :num_being_sampled: number of points being sampled concurrently
      :num_sampled: number of already-sampled points
  \endrst*/
  void Initialize(int dim_in, int num_to_sample_in, int num_being_sampled_in, int num_sampled_in, int num_pts_in, int num_derivatives_in) {
    Initialize(dim_in, num_to_sample_in, num_being_sampled_in, num_sampled_in, num_pts_in, num_derivatives_in, &uniform_generator_);
  }

  void Initialize(int dim_in, int num_to_sample_in, int num_being_sampled_in, int num_sampled_in,
                  int num_pts_in, int num_derivatives_in, UniformRandomGenerator * uniform_generator);

  //! spatial dimension (e.g., entries per point of points_sampled)
  int dim;
  //! number of points in points_sampled (history)
  int num_sampled;
  //! number of points to be sampled in future experiments (i.e., the q in q,p-EI)
  int num_to_sample;
  //! number of points currently being sampled (i.e., the p in q,p-EI)
  int num_being_sampled;
  //! number of derivatives observations
  int num_derivatives;
  //! number of the points in the discretization.
  int num_pts;

  double * points_sampled() {
    return points_sampled_.data();
  }

  double * points_sampled_value() {
    return points_sampled_value_.data();
  }

  double * points_to_sample() {
    return points_to_sample_.data();
  }

  double * points_being_sampled() {
    return points_being_sampled_.data();
  }

  double * discrete_pts() {
    return discrete_pts_.data();
  }

  OL_DISALLOW_COPY_AND_ASSIGN(MockKnowledgeGradientEnvironment);

 private:
  //! coordinates of already-sampled points, ``X``
  std::vector<double> points_sampled_;
  //! function values at points_sampled, ``y``
  std::vector<double> points_sampled_value_;
  //! points to be sampled in experiments (i.e., the q in q,p-EI)
  std::vector<double> points_to_sample_;
  //! points being sampled in concurrent experiments (i.e., the p in q,p-EI)
  std::vector<double> points_being_sampled_;
  // the points in the discretization
  std::vector<double> discrete_pts_;

  //! uniform random number generator for generating coordinates
  UniformRandomGenerator uniform_generator_;
  //! distribution over ``[min, max]`` that coordinate values lie in
  boost::uniform_real<double> uniform_double_;
};

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_KNOWLEDGE_GRADIENT_TEST_HPP_