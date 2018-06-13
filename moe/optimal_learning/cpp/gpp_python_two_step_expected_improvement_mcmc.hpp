/*!
  \file gpp_python_two_step_expected_improvement_mcmc.hpp
  \rst
  This file registers the translation layer for invoking KnowledgeGradientMCMC functions
  (e.g., computing/optimizing KGMCMC; see gpp_knowledge_gradient_optimization_mcmc.hpp) from Python.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_TWO_STEP_EXPECTED_IMPROVEMENT_MCMC_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_TWO_STEP_EXPECTED_IMPROVEMENT_MCMC_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*!\rst
  Exports functions (with docstrings) for knowledge gradient operations:

  1. knowledge gradient (and its gradient) evaluation (uesful for testing)
  2. multistart knowledge gradient optimization (main entry-point)
  3. knowledge gradient evaluation at a list of points (useful for testing, plotting)

  These functions choose between monte-carlo and analytic KG evaluation automatically.
\endrst*/
void ExportTwoStepExpectedImprovementMCMCFunctions();

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_TWO_STEP_EXPECTED_IMPROVEMENT_MCMC_HPP_