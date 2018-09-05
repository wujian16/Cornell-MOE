/*!
  \file gpp_python_knowledge_gradient.hpp
  \rst
  This file registers the translation layer for invoking KnowledgeGradient functions
  (e.g., computing/optimizing KG; see gpp_knowledge_gradient_optimization.hpp) from Python.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_LOWER_CONFIDENCE_BOUND_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_LOWER_CONFIDENCE_BOUND_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*!\rst
  Exports functions (with docstrings) for knowledge gradient operations:

  1. knowledge gradient (and its gradient) evaluation (uesful for testing)
  2. multistart knowledge gradient optimization (main entry-point)
  3. knowledge gradient evaluation at a list of points (useful for testing, plotting)

  These functions choose between monte-carlo and analytic KG evaluation automatically.
\endrst*/
void ExportLowerConfidenceBoundFunctions();

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_PYTHON_LOWER_CONFIDENCE_BOUND_HPP_