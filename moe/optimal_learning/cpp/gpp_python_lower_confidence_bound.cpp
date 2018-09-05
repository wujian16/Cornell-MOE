/*!
  \file gpp_python_knowledge_gradient.cpp
  \rst
  This file has the logic to invoke C++ functions pertaining to knowledge gradient from Python.
  The data flow follows the basic 4 step from gpp_python_common.hpp.

  .. NoteL: several internal functions of this source file are only called from ``Export*()`` functions,
  so their description, inputs, outputs, etc. comments have been moved. These comments exist in
  ``Export*()`` as Python docstrings, so we saw no need to repeat ourselves.
\endrst*/
// This include violates the Google Style Guide by placing an "other" system header ahead of C and C++ system headers.  However,
// it needs to be at the top, otherwise compilation fails on some systems with some versions of python: OS X, python 2.7.3.
// Putting this include first prevents pyport from doing something illegal in C++; reference: http://bugs.python.org/issue10910
#include "Python.h"  // NOLINT(build/include)

#include "gpp_python_lower_confidence_bound.hpp"

// NOLINT-ing the C, C++ header includes as well; otherwise cpplint gets confused
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)

#include <boost/python/bases.hpp>  // NOLINT(build/include_order)
#include <boost/python/class.hpp>  // NOLINT(build/include_order)
#include <boost/python/def.hpp>  // NOLINT(build/include_order)
#include <boost/python/dict.hpp>  // NOLINT(build/include_order)
#include <boost/python/extract.hpp>  // NOLINT(build/include_order)
#include <boost/python/list.hpp>  // NOLINT(build/include_order)
#include <boost/python/object.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_exception.hpp"
#include "gpp_knowledge_gradient_optimization.hpp"
#include "gpp_geometry.hpp"
#include "gpp_math.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimizer_parameters.hpp"
#include "gpp_python_common.hpp"

namespace optimal_learning {

namespace {

double ComputeLowerConfidenceBoundWrapper(const GaussianProcess& gaussian_process,
                                   const int num_fidelity,
                                   const boost::python::list& points_to_sample) {
  int num_derivatives_input = 0;
  const boost::python::list gradients;

  PythonInterfaceInputContainer input_container(points_to_sample, gradients, gaussian_process.dim()-num_fidelity, 1, num_derivatives_input);

  bool configure_for_gradients = false;
  PosteriorMeanEvaluator ps_evaluator(gaussian_process);
  PosteriorMeanEvaluator::StateType ps_state(ps_evaluator, num_fidelity, input_container.points_to_sample.data(), configure_for_gradients);

  return ps_evaluator.ComputePosteriorMean(&ps_state);
}

boost::python::list ComputeGradLowerConfidenceBoundWrapper(const GaussianProcess& gaussian_process,
                                                    const int num_fidelity,
                                                    const boost::python::list& points_to_sample) {
  int num_derivatives_input = 0;
  const boost::python::list gradients;

  PythonInterfaceInputContainer input_container(points_to_sample, gradients, gaussian_process.dim()-num_fidelity, 1, num_derivatives_input);

  std::vector<double> grad_PS(input_container.dim);
  bool configure_for_gradients = true;

  PosteriorMeanEvaluator ps_evaluator(gaussian_process);
  PosteriorMeanEvaluator::StateType ps_state(ps_evaluator, num_fidelity, input_container.points_to_sample.data(), configure_for_gradients);

  ps_evaluator.ComputeGradPosteriorMean(&ps_state, grad_PS.data());
  return VectorToPylist(grad_PS);
}

}  // end unnamed namespace

void ExportLowerConfidenceBoundFunctions() {
  boost::python::def("compute_lower_confidence_bound", ComputePosteriorMeanWrapper, R"%%(
    Compute knowledge gradient.
    If ``num_to_sample == 1`` and ``num_being_sampled == 0`` AND ``force_monte_carlo is false``, this will
    use (fast/accurate) analytic evaluation.
    Otherwise monte carlo-based KG computation is used.

    :param gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    :type gaussian_process: GPP.GaussianProcess (boost::python ctor wrapper around optimal_learning::GaussianProcess)
    :param points_to_sample: initial points to load into state (must be a valid point for the problem);
      i.e., points at which to evaluate EI and/or its gradient
    :type points_to_sample: list of float64 with shape (num_to_sample, dim)
    :param points_being_sampled: points that are being sampled in concurrently experiments
    :type points_being_sampled: list of float64 with shape (num_being_sampled, dim)
    :param num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :type num_to_sample: int > 0
    :param num_being_sampled: number of points being sampled concurrently (i.e., the p in q,p-EI)
    :type num_being_sampled: int >= 0
    :param max_int_steps: number of MC integration points in EI
    :type max_int_steps: int >= 0
    :param best_so_far: best known value of objective so far
    :type best_so_far: float64
    :param force_monte_carlo: true to force monte carlo evaluation of EI
    :type force_monte_carlo: bool
    :param randomness_source: object containing randomness sources; only thread 0's source is used
    :type randomness_source: GPP.RandomnessSourceContainer
    :return: computed EI
    :rtype: float64 >= 0.0
    )%%");

  boost::python::def("compute_grad_lower_confidence_bound", ComputeGradPosteriorMeanWrapper, R"%%(
    Compute the gradient of knowledge gradient evaluated at points_to_sample.
    If num_to_sample = 1 and num_being_sampled = 0 AND force_monte_carlo is false, this will
    use (fast/accurate) analytic evaluation.
    Otherwise monte carlo-based KG computation is used.

    :param gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    :type gaussian_process: GPP.GaussianProcess (boost::python ctor wrapper around optimal_learning::GaussianProcess)
    :param points_to_sample: initial points to load into state (must be a valid point for the problem);
      i.e., points at which to evaluate EI and/or its gradient
    :type points_to_sample: list of float64 with shape (num_to_sample, dim)
    :param points_being_sampled: points that are being sampled in concurrently experiments
    :type points_being_sampled: list of float64 with shape (num_being_sampled, dim)
    :param num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :type num_to_sample: int > 0
    :param num_being_sampled: number of points being sampled concurrently (i.e., the p in q,p-EI)
    :type num_being_sampled: int >= 0
    :param max_int_steps: number of MC integration points in EI
    :type max_int_steps: int >= 0
    :param best_so_far: best known value of objective so far
    :type best_so_far: float64
    :param force_monte_carlo: true to force monte carlo evaluation of EI
    :type force_monte_carlo: bool
    :param randomness_source: object containing randomness sources; only thread 0's source is used
    :type randomness_source: GPP.RandomnessSourceContainer
    :return: gradient of EI (computed at points_to_sample + points_being_sampled, wrt points_to_sample)
    :rtype: list of float64 with shape (num_to_sample, dim)
    )%%");
}

}  // end namespace optimal_learning