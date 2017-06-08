/*!
  \file gpp_python_knowledge_gradient_mcmc.cpp
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

#include "gpp_python_knowledge_gradient_mcmc.hpp"

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
#include <boost/python/make_constructor.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_domain.hpp"
#include "gpp_exception.hpp"
#include "gpp_knowledge_gradient_mcmc_optimization.hpp"
#include "gpp_geometry.hpp"
#include "gpp_math.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimizer_parameters.hpp"
#include "gpp_python_common.hpp"

namespace optimal_learning {

namespace {
/*!\rst
  Surrogate "constructor" for GaussianProcess intended only for use by boost::python.  This aliases the normal C++ constructor,
  replacing ``double const * restrict`` arguments with ``const boost::python::list&`` arguments.
\endrst*/
GaussianProcessMCMC * make_gaussian_process_mcmc(const boost::python::list& hyperparameters_list,
                                                 const boost::python::list& noise_variance_list,
                                                 const boost::python::list& points_sampled,
                                                 const boost::python::list& points_sampled_value,
                                                 const boost::python::list& derivatives,
                                                 int num_mcmc, int num_derivatives, int dim, int num_sampled) {
  const int num_hypers = 2*dim;
  std::vector<double> hyperparameters_list_vector(num_mcmc*num_hypers);
  CopyPylistToVector(hyperparameters_list, num_mcmc*num_hypers, hyperparameters_list_vector);

  std::vector<double> noise_variance_list_vector(num_mcmc*(1+num_derivatives));
  CopyPylistToVector(noise_variance_list, num_mcmc*(1+num_derivatives), noise_variance_list_vector);

  std::vector<double> points_sampled_vector(dim*num_sampled);
  CopyPylistToVector(points_sampled, dim*num_sampled, points_sampled_vector);

  std::vector<double> points_sampled_value_vector(num_sampled*(1+num_derivatives));
  CopyPylistToVector(points_sampled_value, num_sampled*(1+num_derivatives), points_sampled_value_vector);

  std::vector<int> derivatives_vector(num_derivatives);
  CopyPylistToIntVector(derivatives, num_derivatives, derivatives_vector);

  GaussianProcessMCMC * new_gp_mcmc = new GaussianProcessMCMC(hyperparameters_list_vector.data(), noise_variance_list_vector.data(),
                                                              num_mcmc, points_sampled_vector.data(), points_sampled_value_vector.data(),
                                                              derivatives_vector.data(), num_derivatives,
                                                              dim, num_sampled);
  for (int i=0;i<num_mcmc;i++){
      new_gp_mcmc->gaussian_process_lst[i].SetRandomizedSeed(0);
  }
  return new_gp_mcmc;
}

double ComputeKnowledgeGradientMCMCWrapper(GaussianProcessMCMC& gaussian_process_mcmc,
                                           const boost::python::object& optimizer_parameters,
                                           const boost::python::list& domain_bounds,
                                           const boost::python::list& discrete_pts,
                                           const boost::python::list& points_to_sample,
                                           const boost::python::list& points_being_sampled,
                                           int num_pts, int num_to_sample, int num_being_sampled,
                                           int max_int_steps, const boost::python::list& best_so_far,
                                           RandomnessSourceContainer& randomness_source) {
  int num_derivatives_input = 0;
  const boost::python::list gradients;

  PythonInterfaceInputContainer input_container_discrete(discrete_pts, gradients, gaussian_process_mcmc.dim(),
                                                         num_pts*gaussian_process_mcmc.num_mcmc(), num_derivatives_input);
  PythonInterfaceInputContainer input_container(points_to_sample, points_being_sampled, gradients, gaussian_process_mcmc.dim(),
                                                num_to_sample, num_being_sampled, num_derivatives_input);

  bool configure_for_gradients = false;

  std::vector<ClosedInterval> domain_bounds_C(input_container.dim);
  CopyPylistToClosedIntervalVector(domain_bounds, input_container.dim, domain_bounds_C);

  std::vector<double> best_so_far_list(gaussian_process_mcmc.num_mcmc());
  CopyPylistToVector(best_so_far, gaussian_process_mcmc.num_mcmc(), best_so_far_list);

  TensorProductDomain domain(domain_bounds_C.data(), input_container.dim);
  const GradientDescentParameters& gradient_descent_parameters = boost::python::extract<GradientDescentParameters&>(optimizer_parameters.attr("optimizer_parameters"));

  std::vector<typename KnowledgeGradientState<TensorProductDomain>::EvaluatorType> evaluator_vector;
  KnowledgeGradientMCMCEvaluator<TensorProductDomain> kg_evaluator(gaussian_process_mcmc, input_container_discrete.points_to_sample.data(),
                                                                   num_pts, max_int_steps, domain, gradient_descent_parameters,
                                                                   best_so_far_list.data(), &evaluator_vector);

  std::vector<typename KnowledgeGradientEvaluator<TensorProductDomain>::StateType> state_vector;
  KnowledgeGradientMCMCEvaluator<TensorProductDomain>::StateType kg_state(kg_evaluator, input_container.points_to_sample.data(),
                                                                          input_container.points_being_sampled.data(),
                                                                          input_container.num_to_sample,
                                                                          input_container.num_being_sampled,
                                                                          num_pts, gaussian_process_mcmc.derivatives().data(),
                                                                          gaussian_process_mcmc.num_derivatives(), configure_for_gradients,
                                                                          randomness_source.normal_rng_vec.data(), &state_vector);
  return kg_evaluator.ComputeKnowledgeGradient(&kg_state);
}

boost::python::list ComputeGradKnowledgeGradientMCMCWrapper(GaussianProcessMCMC& gaussian_process_mcmc,
                                                            const boost::python::object& optimizer_parameters,
                                                            const boost::python::list& domain_bounds,
                                                            const boost::python::list& discrete_pts,
                                                            const boost::python::list& points_to_sample,
                                                            const boost::python::list& points_being_sampled,
                                                            int num_pts, int num_to_sample, int num_being_sampled,
                                                            int max_int_steps, const boost::python::list& best_so_far,
                                                            RandomnessSourceContainer& randomness_source) {
  int num_derivatives_input = 0;
  const boost::python::list gradients;

  PythonInterfaceInputContainer input_container_discrete(discrete_pts, gradients, gaussian_process_mcmc.dim(),
                                                         num_pts*gaussian_process_mcmc.num_mcmc(), num_derivatives_input);
  PythonInterfaceInputContainer input_container(points_to_sample, points_being_sampled, gradients, gaussian_process_mcmc.dim(),
                                                num_to_sample, num_being_sampled, num_derivatives_input);

  std::vector<double> grad_KG(num_to_sample*input_container.dim);
  bool configure_for_gradients = true;

  std::vector<ClosedInterval> domain_bounds_C(input_container.dim);
  CopyPylistToClosedIntervalVector(domain_bounds, input_container.dim, domain_bounds_C);

  std::vector<double> best_so_far_list(gaussian_process_mcmc.num_mcmc());
  CopyPylistToVector(best_so_far, gaussian_process_mcmc.num_mcmc(), best_so_far_list);

  TensorProductDomain domain(domain_bounds_C.data(), input_container.dim);
  const GradientDescentParameters& gradient_descent_parameters = boost::python::extract<GradientDescentParameters&>(optimizer_parameters.attr("optimizer_parameters"));

  std::vector<typename KnowledgeGradientState<TensorProductDomain>::EvaluatorType> evaluator_vector;
  KnowledgeGradientMCMCEvaluator<TensorProductDomain> kg_evaluator(gaussian_process_mcmc, input_container_discrete.points_to_sample.data(),
                                                                   num_pts, max_int_steps, domain, gradient_descent_parameters,
                                                                   best_so_far_list.data(), &evaluator_vector);

  std::vector<typename KnowledgeGradientEvaluator<TensorProductDomain>::StateType> state_vector;
  KnowledgeGradientMCMCEvaluator<TensorProductDomain>::StateType kg_state(kg_evaluator, input_container.points_to_sample.data(),
                                                                          input_container.points_being_sampled.data(),
                                                                          input_container.num_to_sample,
                                                                          input_container.num_being_sampled,
                                                                          num_pts, gaussian_process_mcmc.derivatives().data(),
                                                                          gaussian_process_mcmc.num_derivatives(), configure_for_gradients,
                                                                          randomness_source.normal_rng_vec.data(), &state_vector);
  kg_evaluator.ComputeGradKnowledgeGradient(&kg_state, grad_KG.data());

  return VectorToPylist(grad_KG);
}

/*!\rst
  Utility that dispatches KG optimization based on optimizer type and num_to_sample.
  This is just used to reduce copy-pasted code.

  \param
    :optimizer_parameters: python/cpp_wrappers/optimization._CppOptimizerParameters
      Python object containing the DomainTypes domain_type and OptimizerTypes optimzer_type to use as well as
      appropriate parameter structs e.g., NewtonParameters for type kNewton).
      See comments on the python interface for multistart_expected_improvement_optimization_wrapper
    :gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities) that describes the
      underlying GP
    :input_container: PythonInterfaceInputContainer object containing data about points_being_sampled
    :domain: object specifying the domain to optimize over (see gpp_domain.hpp)
    :optimizer_type: type of optimization to use (e.g., null, gradient descent)
    :num_to_sample: how many simultaneous experiments you would like to run (i.e., the q in q,p-KG)
    :best_so_far: value of the best sample so far (must be min(points_sampled_value))
    :max_int_steps: maximum number of MC iterations
    :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
    :randomness_source: object containing randomness sources (sufficient for multithreading) used in KG computation
    :status: pydict object; cannot be None
  \output
    :randomness_source: PRNG internal states modified
    :status: modified on exit to describe whether convergence occurred
    :best_points_to_sample[num_to_sample][dim]: next set of points to evaluate
\endrst*/

template <typename DomainType>
void DispatchKnowledgeGradientMCMCOptimization(const boost::python::object& optimizer_parameters,
                                               const boost::python::object& optimizer_parameters_inner,
                                               GaussianProcessMCMC& gaussian_process_mcmc,
                                               const PythonInterfaceInputContainer& input_container_discrete,
                                               const PythonInterfaceInputContainer& input_container,
                                               const DomainType& domain, OptimizerTypes optimizer_type,
                                               int num_pts, int num_to_sample, std::vector<double> best_so_far_list,
                                               int max_int_steps, int max_num_threads,
                                               RandomnessSourceContainer& randomness_source,
                                               boost::python::dict& status,
                                               double * restrict best_points_to_sample) {

  bool found_flag = false;

  switch (optimizer_type) {
    case OptimizerTypes::kNull: {
      ThreadSchedule thread_schedule(max_num_threads, omp_sched_static);
      // optimizer_parameters must contain an int num_random_samples field, extract it
      const GradientDescentParameters& gradient_descent_parameters_inner = boost::python::extract<GradientDescentParameters&>(optimizer_parameters_inner.attr("optimizer_parameters"));
      int num_random_samples = boost::python::extract<int>(optimizer_parameters.attr("num_random_samples"));

      ComputeKGMCMCOptimalPointsToSampleViaLatinHypercubeSearch(gaussian_process_mcmc, gradient_descent_parameters_inner, domain, thread_schedule,
                                                                input_container.points_being_sampled.data(),
                                                                input_container_discrete.points_to_sample.data(),
                                                                num_random_samples, num_to_sample,
                                                                input_container.num_being_sampled,
                                                                num_pts, best_so_far_list.data(), max_int_steps,
                                                                &found_flag, &randomness_source.uniform_generator,
                                                                randomness_source.normal_rng_vec.data(),
                                                                best_points_to_sample);
      status[std::string("lhc_") + domain.kName + "_domain_found_update"] = found_flag;
      break;
    }  // end case kNull optimizer_type
    case OptimizerTypes::kGradientDescent: {
      // optimizer_parameters must contain a optimizer_parameters field
      // of type GradientDescentParameters. extract it
      const GradientDescentParameters& gradient_descent_parameters = boost::python::extract<GradientDescentParameters&>(optimizer_parameters.attr("optimizer_parameters"));
      const GradientDescentParameters& gradient_descent_parameters_inner = boost::python::extract<GradientDescentParameters&>(optimizer_parameters_inner.attr("optimizer_parameters"));
      ThreadSchedule thread_schedule(max_num_threads, omp_sched_dynamic);
      int num_random_samples = boost::python::extract<int>(optimizer_parameters.attr("num_random_samples"));

      bool random_search_only = false;
      ComputeKGMCMCOptimalPointsToSample(gaussian_process_mcmc, gradient_descent_parameters, gradient_descent_parameters_inner, domain, thread_schedule,
                                         input_container.points_being_sampled.data(), input_container_discrete.points_to_sample.data(),
                                         num_to_sample, input_container.num_being_sampled, num_pts,
                                         best_so_far_list.data(), max_int_steps, random_search_only, num_random_samples, &found_flag,
                                         &randomness_source.uniform_generator, randomness_source.normal_rng_vec.data(), best_points_to_sample);

      status[std::string("gradient_descent_") + domain.kName + "_domain_found_update"] = found_flag;
      break;
    }  // end case kGradientDescent optimizer_type
    default: {
      std::fill(best_points_to_sample, best_points_to_sample + input_container.dim*num_to_sample, 0.0);
      OL_THROW_EXCEPTION(OptimalLearningException, "ERROR: invalid optimizer choice. Setting all coordinates to 0.0.");
      break;
    }
  }  // end switch over optimizer_type
}

boost::python::list MultistartKnowledgeGradientMCMCOptimizationWrapper(const boost::python::object& optimizer_parameters,
                                                                       const boost::python::object& optimizer_parameters_inner,
                                                                       GaussianProcessMCMC& gaussian_process_mcmc,
                                                                       const boost::python::list& domain_bounds,
                                                                       const boost::python::list& discrete_pts,
                                                                       const boost::python::list& points_being_sampled,
                                                                       int num_pts, int num_to_sample, int num_being_sampled,
                                                                       const boost::python::list& best_so_far, int max_int_steps, int max_num_threads,
                                                                       RandomnessSourceContainer& randomness_source,
                                                                       boost::python::dict& status) {
  // TODO(GH-131): make domain objects constructible from python; and pass them in through
  // the optimizer_parameters python object

  // abort if we do not have enough sources of randomness to run with max_num_threads
  if (unlikely(max_num_threads > static_cast<int>(randomness_source.normal_rng_vec.size()))) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "Fewer randomness_sources than max_num_threads.", randomness_source.normal_rng_vec.size(), max_num_threads);
  }

  int num_to_sample_input = 0;  // No points to sample; we are generating these via KG optimization
  const boost::python::list points_to_sample_dummy;

  int num_derivatives_input = 0;
  const boost::python::list gradients;

  PythonInterfaceInputContainer input_container_discrete(discrete_pts, gradients, gaussian_process_mcmc.dim(),
                                                         num_pts*gaussian_process_mcmc.num_mcmc(), num_derivatives_input);
  PythonInterfaceInputContainer input_container(points_to_sample_dummy, points_being_sampled, gradients, gaussian_process_mcmc.dim(),
                                                num_to_sample_input, num_being_sampled, num_derivatives_input);

  std::vector<ClosedInterval> domain_bounds_C(input_container.dim);
  CopyPylistToClosedIntervalVector(domain_bounds, input_container.dim, domain_bounds_C);

  std::vector<double> best_so_far_list(gaussian_process_mcmc.num_mcmc());
  CopyPylistToVector(best_so_far, gaussian_process_mcmc.num_mcmc(), best_so_far_list);

  std::vector<double> best_points_to_sample_C(input_container.dim*num_to_sample);

  DomainTypes domain_type = boost::python::extract<DomainTypes>(optimizer_parameters.attr("domain_type"));
  OptimizerTypes optimizer_type = boost::python::extract<OptimizerTypes>(optimizer_parameters.attr("optimizer_type"));
  switch (domain_type) {
    case DomainTypes::kTensorProduct: {
      TensorProductDomain domain(domain_bounds_C.data(), input_container.dim);

      DispatchKnowledgeGradientMCMCOptimization(optimizer_parameters, optimizer_parameters_inner, gaussian_process_mcmc, input_container_discrete,
                                                input_container, domain, optimizer_type, num_pts, num_to_sample, best_so_far_list,
                                                max_int_steps, max_num_threads, randomness_source, status, best_points_to_sample_C.data());
      break;
    }  // end case OptimizerTypes::kTensorProduct
    case DomainTypes::kSimplex: {
      SimplexIntersectTensorProductDomain domain(domain_bounds_C.data(), input_container.dim);

      DispatchKnowledgeGradientMCMCOptimization(optimizer_parameters, optimizer_parameters_inner, gaussian_process_mcmc, input_container_discrete,
                                                input_container, domain, optimizer_type, num_pts, num_to_sample, best_so_far_list,
                                                max_int_steps, max_num_threads, randomness_source, status, best_points_to_sample_C.data());
      break;
    }  // end case OptimizerTypes::kSimplex
    default: {
      std::fill(best_points_to_sample_C.begin(), best_points_to_sample_C.end(), 0.0);
      OL_THROW_EXCEPTION(OptimalLearningException, "ERROR: invalid domain choice. Setting all coordinates to 0.0.");
      break;
    }
  }  // end switch over domain_type

  return VectorToPylist(best_points_to_sample_C);
}

boost::python::list EvaluateKGMCMCAtPointListWrapper(GaussianProcessMCMC& gaussian_process_mcmc,
                                                     const boost::python::object& optimizer_parameters,
                                                     const boost::python::list& domain_bounds,
                                                     const boost::python::list& discrete_pts,
                                                     const boost::python::list& initial_guesses,
                                                     const boost::python::list& points_being_sampled,
                                                     int num_multistarts, int num_pts, int num_to_sample,
                                                     int num_being_sampled, const boost::python::list& best_so_far,
                                                     int max_int_steps, int max_num_threads,
                                                     RandomnessSourceContainer& randomness_source,
                                                     boost::python::dict& status) {
  // abort if we do not have enough sources of randomness to run with max_num_threads
  if (unlikely(max_num_threads > static_cast<int>(randomness_source.normal_rng_vec.size()))) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "Fewer randomness_sources than max_num_threads.", randomness_source.normal_rng_vec.size(), max_num_threads);
  }

  int num_to_sample_input = 0;  // No points to sample; we are generating these via KG optimization
  const boost::python::list points_to_sample_dummy;

  int num_derivatives_input = 0;
  const boost::python::list gradients;

  PythonInterfaceInputContainer input_container_discrete(discrete_pts, gradients, gaussian_process_mcmc.dim(),
                                                         num_pts*gaussian_process_mcmc.num_mcmc(), num_derivatives_input);

  PythonInterfaceInputContainer input_container(points_to_sample_dummy, points_being_sampled, gradients, gaussian_process_mcmc.dim(),
                                                num_to_sample_input, num_being_sampled, num_derivatives_input);


  std::vector<double> result_point_C(input_container.dim);  // not used
  std::vector<double> result_function_values_C(num_multistarts);
  std::vector<double> initial_guesses_C(input_container.dim * num_multistarts);

  CopyPylistToVector(initial_guesses, input_container.dim * num_multistarts, initial_guesses_C);

  ThreadSchedule thread_schedule(max_num_threads, omp_sched_static);
  bool found_flag = false;

  std::vector<ClosedInterval> domain_bounds_C(input_container.dim);
  CopyPylistToClosedIntervalVector(domain_bounds, input_container.dim, domain_bounds_C);

  std::vector<double> best_so_far_list(gaussian_process_mcmc.num_mcmc());
  CopyPylistToVector(best_so_far, gaussian_process_mcmc.num_mcmc(), best_so_far_list);

  TensorProductDomain domain(domain_bounds_C.data(), input_container.dim);
  const GradientDescentParameters& gradient_descent_parameters = boost::python::extract<GradientDescentParameters&>(optimizer_parameters.attr("optimizer_parameters"));

  EvaluateKGMCMCAtPointList(gaussian_process_mcmc, gradient_descent_parameters, domain, thread_schedule, initial_guesses_C.data(),
                            input_container.points_being_sampled.data(), input_container_discrete.points_to_sample.data(),
                            num_multistarts, num_to_sample, input_container.num_being_sampled,
                            num_pts, best_so_far_list.data(), max_int_steps, &found_flag, randomness_source.normal_rng_vec.data(),
                            result_function_values_C.data(), result_point_C.data());

  status["evaluate_KG_at_point_list"] = found_flag;

  return VectorToPylist(result_function_values_C);
}
}  // end unnamed namespace

void ExportKnowldegeGradientMCMCFunctions() {
  boost::python::class_<GaussianProcessMCMC, boost::noncopyable>("GaussianProcessMCMC", boost::python::no_init)
      .def("__init__", boost::python::make_constructor(&make_gaussian_process_mcmc), R"%%(
    Constructor for a ``GPP.GaussianProcess`` object.

    Seeds internal NormalRNG randomly.

    :param hyperparameters: covariance hyperparameters; see "Details on ..." section at the top of ``BOOST_PYTHON_MODULE``
    :type hyperparameters: list of len 2; index 0 is a float64 ``\alpha`` (signal variance) and index 1 is the length scales (list of floa64 of length ``dim``)
    :param points_sampled: points that have already been sampled
    :type points_sampled: list of float64 with shape (num_sampled, dim)
    :param points_sampled_value: values of the already-sampled points
    :type points_sampled_value: list of float64 with shape (num_sampled, )
    :param noise_variance: the ``\sigma_n^2`` (noise variance) associated w/observation, points_sampled_value
    :type noise_variance: list of float64 with shape (num_sampled, )
    :param dim: the spatial dimension of a point (i.e., number of independent params in experiment)
    :type param: int > 0
    :param num_sampled: number of already-sampled points
    :type num_sampled: int > 0
          )%%");

  boost::python::def("compute_knowledge_gradient_mcmc", ComputeKnowledgeGradientMCMCWrapper, R"%%(
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

  boost::python::def("compute_grad_knowledge_gradient_mcmc", ComputeGradKnowledgeGradientMCMCWrapper, R"%%(
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

  boost::python::def("multistart_knowledge_gradient_mcmc_optimization", MultistartKnowledgeGradientMCMCOptimizationWrapper, R"%%(
    Optimize expected improvement (i.e., solve q,p-EI) over the specified domain using the specified optimization method.
    Can optimize for num_to_sample new points to sample (i.e., aka "q", experiments to run) simultaneously.
    Allows the user to specify num_being_sampled (aka "p") ongoing/concurrent experiments.

    The _CppOptimizerParameters object is a python class defined in:
    python/cpp_wrappers/optimization._CppOptimizerParameters
    See that class definition for more details.

    This function expects it to have the fields:

    * domain_type (DomainTypes enum from this file)
    * optimizer_type (OptimizerTypes enum from this file)
    * num_random_samples (int, number of samples to 'dumb' search over, if 'dumb' search is being used.
      e.g., if optimizer = kNull or if to_sample > 1)
    * optimizer_parameters (*Parameters struct (gpp_optimizer_parameters.hpp) where * matches optimizer_type
      unused if optimizer_type == kNull)

    This function also has the option of using GPU to compute general q,p-EI via MC simulation. To enable it,
    make sure you have installed GPU components of MOE, otherwise, it will throw Runtime excpetion.

    .. WARNING:: this function FAILS and returns an EMPTY LIST if the number of random sources < max_num_threads

    :param optimizer_parameters: python object containing the DomainTypes domain_type and
      OptimizerTypes optimzer_type to use as well as
      appropriate parameter structs e.g., NewtonParameters for type kNewton)
    :type optimizer_parameters: _CppOptimizerParameters
    :param gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    :type gaussian_process: GPP.GaussianProcess (boost::python ctor wrapper around optimal_learning::GaussianProcess)
    :param domain: [lower, upper] bound pairs for each dimension
    :type domain: list of float64 with shape (dim, 2)
    :param points_being_sampled: points that are being sampled in concurrently experiments
    :type points_being_sampled: list of float64 with shape (num_being_sampled, dim)
    :param num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :type num_to_sample: int > 0
    :param num_being_sampled: number of points being sampled concurrently (i.e., the p in q,p-EI)
    :type num_being_sampled: int >= 0
    :param best_so_far: best known value of objective so far
    :type best_so_far: float64
    :param max_int_steps: number of MC integration points in EI
    :type max_int_steps: int >= 0
    :param max_num_threads: max number of threads to use during EI optimization
    :type max_num_threads: int >= 1
    :param use_gpu: set to 1 if user wants to use GPU for MC computation
    :type use_gpu: bool
    :param which_gpu: GPU device ID
    :type which_gpu: int >= 0
    :param randomness_source: object containing randomness sources; only thread 0's source is used
    :type randomness_source: GPP.RandomnessSourceContainer
    :param status: pydict object (cannot be None!); modified on exit to describe whether convergence occurred
    :type status: dict
    :return: next set of points to eval
    :rtype: list of float64 with shape (num_to_sample, dim)
    )%%");

  boost::python::def("evaluate_KG_mcmc_at_point_list", EvaluateKGMCMCAtPointListWrapper, R"%%(
    Evaluates the expected improvement at each point in initial_guesses; can handle q,p-EI.
    Useful for plotting.

    Equivalent to::

      result = []
      for point in initial_guesses:
          result.append(compute_expected_improvement(point, ...))

    But this method is substantially faster (loop in C++ and multithreaded).

    .. WARNING:: this function FAILS and returns an EMPTY LIST if the number of random sources < max_num_threads


    :param num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :type num_to_sample: int > 0


    :param gaussian_process: GaussianProcess object (holds points_sampled, values, noise_variance, derived quantities)
    :type gaussian_process: GPP.GaussianProcess (boost::python ctor wrapper around optimal_learning::GaussianProcess)
    :param initial_guesses: points at which to evaluate EI
    :type initial_guesses: list of flaot64 with shape (num_multistarts, num_to_sample, dim)
    :param points_being_sampled: points that are being sampled in concurrently experiments
    :type points_being_sampled: list of float64 with shape (num_being_sampled, dim)
    :param num_multistarts: number of points at which to evaluate EI
    :type num_multistarts: int > 0
    :param num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :type num_to_sample: int > 0
    :param num_being_sampled: number of points being sampled concurrently (i.e., the p in q,p-EI)
    :type num_being_sampled: int >= 0
    :param best_so_far: best known value of objective so far
    :type best_so_far: float64
    :param max_int_steps: number of MC integration points in EI
    :type max_int_steps: int >= 0
    :param max_num_threads: max number of threads to use during EI optimization
    :type max_num_threads: int >= 1
    :param randomness_source: object containing randomness sources; only thread 0's source is used
    :type randomness_source: GPP.RandomnessSourceContainer
    :param status: pydict object (cannot be None!); modified on exit to describe whether convergence occurred
    :type status: dict
    :return: EI values at each point of the initial_guesses list, in the same order
    :rtype: list of float64 with shape (num_multistarts, )
    )%%");
}
}