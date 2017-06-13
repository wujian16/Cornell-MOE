/*!
  \file gpp_python_model_selection.cpp
  \rst
  This file has the logic to invoke C++ functions pertaining to model selection from Python.
  The data flow follows the basic 4 step from gpp_python_common.hpp.

  Note: several internal functions of this source file are only called from Export*() functions,
  so their description, inputs, outputs, etc. comments have been moved. These comments exist in
  Export*() as Python docstrings, so we saw no need to repeat ourselves.
\endrst*/
// This include violates the Google Style Guide by placing an "other" system header ahead of C and C++ system headers.  However,
// it needs to be at the top, otherwise compilation fails on some systems with some versions of python: OS X, python 2.7.3.
// Putting this include first prevents pyport from doing something illegal in C++; reference: http://bugs.python.org/issue10910
#include "Python.h"  // NOLINT(build/include)

#include "gpp_python_model_selection.hpp"

// NOLINT-ing the C, C++ header includes as well; otherwise cpplint gets confused
#include <algorithm>  // NOLINT(build/include_order)
#include <limits>  // NOLINT(build/include_order)
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)

#include <boost/python/def.hpp>  // NOLINT(build/include_order)
#include <boost/python/dict.hpp>  // NOLINT(build/include_order)
#include <boost/python/extract.hpp>  // NOLINT(build/include_order)
#include <boost/python/list.hpp>  // NOLINT(build/include_order)
#include <boost/python/object.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_domain.hpp"
#include "gpp_exception.hpp"
#include "gpp_geometry.hpp"
#include "gpp_model_selection.hpp"
#include "gpp_optimizer_parameters.hpp"
#include "gpp_python_common.hpp"

namespace optimal_learning {

namespace {

double ComputeLogLikelihoodWrapper(const boost::python::list& points_sampled,
                                   const boost::python::list& points_sampled_value,
                                   int dim, int num_sampled,
                                   LogLikelihoodTypes objective_type,
                                   const boost::python::list& hyperparameters,
                                   const boost::python::list& derivatives,
                                   int num_derivatives,
                                   const boost::python::list& noise_variance) {
  const int num_to_sample = 0;
  const boost::python::list points_to_sample_dummy;
  PythonInterfaceInputContainer input_container(points_sampled, points_sampled_value, noise_variance,
                                                points_to_sample_dummy, derivatives, num_derivatives, dim, num_sampled, num_to_sample);

  const int num_hypers = Square(dim) + 3*dim;
  std::vector<double> hypers(num_hypers);
  CopyPylistToVector(hyperparameters, num_hypers, hypers);

  AdditiveKernel adk(dim, hypers);

  switch (objective_type) {
    case LogLikelihoodTypes::kLogMarginalLikelihood: {
      LogMarginalLikelihoodEvaluator log_marginal_eval(input_container.points_sampled.data(),
                                                       input_container.points_sampled_value.data(),
                                                       input_container.derivatives.data(), input_container.num_derivatives,
                                                       input_container.dim, input_container.num_sampled);
      LogMarginalLikelihoodState log_marginal_state(log_marginal_eval, adk, input_container.noise_variance);
      double log_likelihood = log_marginal_eval.ComputeLogLikelihood(log_marginal_state);
      return log_likelihood;
    }  // end case LogLikelihoodTypes::kLogMarginalLikelihood
/*    case LogLikelihoodTypes::kLeaveOneOutLogLikelihood: {
      LeaveOneOutLogLikelihoodEvaluator leave_one_out_eval(input_container.points_sampled.data(),
                                                           input_container.points_sampled_value.data(),
                                                           input_container.noise_variance.data(),
                                                           input_container.dim,
                                                           input_container.num_sampled);
      LeaveOneOutLogLikelihoodState leave_one_out_state(leave_one_out_eval, matern_25);

      double loo_likelihood = leave_one_out_eval.ComputeLogLikelihood(leave_one_out_state);
      return loo_likelihood;
    }*/
    default: {
      double log_likelihood = -std::numeric_limits<double>::max();
      OL_THROW_EXCEPTION(OptimalLearningException, "ERROR: invalid objective mode choice. Setting log likelihood to -DBL_MAX.");
      return log_likelihood;
    }
  }  // end switch over objective_type
}

boost::python::list ComputeHyperparameterGradLogLikelihoodWrapper(const boost::python::list& points_sampled,
                                                                  const boost::python::list& points_sampled_value,
                                                                  int dim, int num_sampled,
                                                                  LogLikelihoodTypes objective_type,
                                                                  const boost::python::list& hyperparameters,
                                                                  const boost::python::list& derivatives,
                                                                  int num_derivatives,
                                                                  const boost::python::list& noise_variance) {
  const int num_to_sample = 0;
  const boost::python::list points_to_sample_dummy;
  PythonInterfaceInputContainer input_container(points_sampled, points_sampled_value, noise_variance,
                                                points_to_sample_dummy, derivatives, num_derivatives, dim, num_sampled, num_to_sample);

  const int num_hypers = Square(dim) + 3*dim;

  std::vector<double> hypers(num_hypers);
  CopyPylistToVector(hyperparameters, num_hypers, hypers);

  AdditiveKernel adk(dim, hypers);

  std::vector<double> grad_log_likelihood(adk.GetNumberOfHyperparameters() + 1 + num_derivatives);
  switch (objective_type) {
    case LogLikelihoodTypes::kLogMarginalLikelihood: {
      LogMarginalLikelihoodEvaluator log_marginal_eval(input_container.points_sampled.data(),
                                                       input_container.points_sampled_value.data(),
                                                       input_container.derivatives.data(), input_container.num_derivatives,
                                                       input_container.dim, input_container.num_sampled);
      LogMarginalLikelihoodState log_marginal_state(log_marginal_eval, adk, input_container.noise_variance);

      log_marginal_eval.ComputeGradLogLikelihood(&log_marginal_state, grad_log_likelihood.data());
      break;
    }  // end case LogLikelihoodTypes::kLogMarginalLikelihood
/*    case LogLikelihoodTypes::kLeaveOneOutLogLikelihood: {
      LeaveOneOutLogLikelihoodEvaluator leave_one_out_eval(input_container.points_sampled.data(),
                                                           input_container.points_sampled_value.data(),
                                                           input_container.noise_variance.data(),
                                                           input_container.dim,
                                                           input_container.num_sampled);
      LeaveOneOutLogLikelihoodState leave_one_out_state(leave_one_out_eval, matern_25);

      leave_one_out_eval.ComputeGradLogLikelihood(&leave_one_out_state, grad_log_likelihood.data());
      break;
    }*/
    default: {
      std::fill(grad_log_likelihood.begin(), grad_log_likelihood.end(), std::numeric_limits<double>::max());
      OL_THROW_EXCEPTION(OptimalLearningException, "ERROR: invalid objective mode choice. Setting all gradients to DBL_MAX.");
      break;
    }
  }  // end switch over objective_type
  return VectorToPylist(grad_log_likelihood);
}
}  // end unnamed namespace

void ExportModelSelectionFunctions() {
  boost::python::def("compute_log_likelihood", ComputeLogLikelihoodWrapper, R"%%(
    Computes the specified log likelihood measure of model fit using the given
    hyperparameters.

    :param points_sampled: points that have already been sampled
    :type points_sampled: list of float64 with shape (num_sampled, dim)
    :param points_sampled_value: values of the already-sampled points
    :type points_sampled_value: list of float64 with shape (num_sampled, )
    :param dim: the spatial dimension of a point (i.e., number of independent params in experiment)
    :type dim: int > 0
    :param num_sampled: number of already-sampled points
    :type num_sampled: int > 0
    :param objective_mode: describes which log likelihood measure to compute (e.g., kLogMarginalLikelihood)
    :type objective_mode: GPP.LogLikelihoodTypes (enum)
    :param hyperparameters: covariance hyperparameters; see "Details on ..." section at the top of ``BOOST_PYTHON_MODULE``
    :type hyperparameters: list of len 2; index 0 is a float64 ``\alpha`` (signal variance) and index 1 is the length scales (list of floa64 of length ``dim``)
    :param noise_variance: the ``\sigma_n^2`` (noise variance) associated w/observation, points_sampled_value
    :type noise_variance: list of float64 with shape (num_sampled, )
    :return: computed log marginal likelihood of prior
    :rtype: float64
    )%%");

  boost::python::def("compute_hyperparameter_grad_log_likelihood", ComputeHyperparameterGradLogLikelihoodWrapper, R"%%(
    Computes the gradient of the specified log likelihood measure of model fit using the given
    hyperparameters. Gradient computed wrt the given hyperparameters.

    ``n_hyper`` denotes the number of hyperparameters.

    :param points_sampled: points that have already been sampled
    :type points_sampled: list of float64 with shape (num_sampled, dim)
    :param points_sampled_value: values of the already-sampled points
    :type points_sampled_value: list of float64 with shape (num_sampled, )
    :param dim: the spatial dimension of a point (i.e., number of independent params in experiment)
    :type dim: int > 0
    :param num_sampled: number of already-sampled points
    :type num_sampled: int > 0
    :param objective_mode: describes which log likelihood measure to compute (e.g., kLogMarginalLikelihood)
    :type objective_mode: GPP.LogLikelihoodTypes (enum)
    :param hyperparameters: covariance hyperparameters; see "Details on ..." section at the top of ``BOOST_PYTHON_MODULE``
    :type hyperparameters: list of len 2; index 0 is a float64 ``\alpha`` (signal variance) and index 1 is the length scales (list of floa64 of length ``dim``)
    :param noise_variance: the ``\sigma_n^2`` (noise variance) associated w/observation, points_sampled_value
    :type noise_variance: list of float64 with shape (num_sampled, )
    :return: gradients of log marginal likelihood wrt hyperparameters
    :rtype: list of float64 with shape (num_hyperparameters, )
    )%%");
}

}  // end namespace optimal_learning
