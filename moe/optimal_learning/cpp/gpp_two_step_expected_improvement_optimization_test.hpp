/*!
  \file gpp_two_step_expected_improvement_optimization_test.hpp
  \rst
  Functions for testing 2-EI functionality.

  Tests are broken into two main groups:

  * ping (unit) tests for 2-EI
  * unit + integration tests for optimization methods

  The ping tests are set up the same way as the ping tests in gpp_covariance_test; using the function evaluator and ping
  framework defined in gpp_test_utils.

  Finally, we have tests for 2-EI optimization.  These include multithreading tests (verifying that each core
  does what is expected) as well as integration tests for EI optimization.  Unit tests for optimizers live in
  gpp_optimization_test.hpp/cpp.  These integration tests use constructed data but exercise all the
  same code paths used for hyperparameter optimization in production.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_TWO_STEP_EXPECTED_IMPROVEMENT_OPTIMIZATION_TEST_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_TWO_STEP_EXPECTED_IMPROVEMENT_OPTIMIZATION_TEST_HPP_

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
OL_WARN_UNUSED_RESULT int PingTwoEIGeneralTest();

/*!\rst
  Checks that the gradients (spatial) of Knowledge Gradient are computed correctly.

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int RunTwoEITests();

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
}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_TWO_STEP_EXPECTED_IMPROVEMENT_OPTIMIZATION_TEST_HPP_