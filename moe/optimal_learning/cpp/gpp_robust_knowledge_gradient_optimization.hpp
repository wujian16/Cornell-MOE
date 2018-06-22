/*!
  \file gpp_robust_knowledge_gradient_optimization.hpp
  \rst
  1. OVERVIEW OF KNOWLEDGE GRADIENT WHAT ARE WE TRYING TO DO?
  2. IMPLEMENTATION NOTES
  3. CITATIONS
    **1. OVERVIEW OF KNOWLEDGE GRADIENT; WHAT ARE WE TRYING TO DO?**
    .. Note:: these comments are copied in Python: interfaces/__init__.py
    The optimization process models the objective using a Gaussian process (GP) prior
    (also called a GP predictor) based on the specified covariance and the input
    data (e.g., through member functions ComputeMeanOfPoints, ComputeVarianceOfPoints).  Using the GP,
    we can compute the knowledge gradient (KG) from sampling any particular point.  KG
    is defined relative to the best currently known value, and it represents what the
    algorithm believes is the most likely outcome from sampling a particular point in parameter
    space (aka conducting a particular experiment).
    See KnowledgeGradientEvaluator class docs for further details on computing KG.
    Both support ComputeKnowledgeGradient() and ComputeGradKnowledgeGradient().
    The behavior of the GP is controlled by its underlying
    covariance function and the data/uncertainty of prior points (experiments).
    With the ability of the compute KG, the final step is to optimize
    to find the best KG.  This is done using multistart gradient descent (MGD), in
    ComputeKGOptimalPointsToSample(). This method wraps a MGD call and falls back on random search
    if that fails. See gpp_optimization.hpp for multistart/optimization templates. This method
    can evaluate and optimize KG at serval points simultaneously; e.g., if we wanted to run 4 simultaneous
    experiments, we can use KG to select all 4 points at once.
    Additionally, there are use cases where we have existing experiments that are not yet complete but
    we have an opportunity to start some new trials. For example, maybe we are a drug company currently
    testing 2 combinations of dosage levels. We got some new funding, and can now afford to test
    3 more sets of dosage parameters. Ideally, the decision on the new experiments should depend on
    the existence of the 2 ongoing tests. We may not have any data from the ongoing experiments yet;
    e.g., they are [double]-blind trials. If nothing else, we would not want to duplicate any
    existing experiments! So we want to solve 3-KG using the knowledge of the 2 ongoing experiments.
    We call this q,p-KG, so the previous example would be 3,2-KG. So q is the number of new
    (simultaneous) experiments to select. In code, this would be the size of the output from KG
    optimization (i.e., ``best_points_to_sample``, of which there are ``q = num_to_sample points``).
    p is the number of ongoing/incomplete experiments to take into account (i.e., ``points_being_sampled``
    of which there are ``p = num_being_sampled`` points).
    Back to optimization: the idea behind gradient descent is simple.  The gradient gives us the
    direction of steepest ascent (negative gradient is steepest descent).  So each iteration, we
    compute the gradient and take a step in that direction.  The size of the step is not specified
    by GD and is left to the specific implementation.  Basically if we take steps that are
    too large, we run the risk of over-shooting the solution and even diverging.  If we
    take steps that are too small, it may take an intractably long time to reach the solution.
    Thus the magic is in choosing the step size; we do not claim that our implementation is
    perfect, but it seems to work reasonably.  See ``gpp_optimization.hpp`` for more details about
    GD as well as the template definition.
    For particularly difficult problems or problems where gradient descent's parameters are not
    well-chosen, GD can fail to converge.  If this happens, we can fall back on heuristics;
    e.g., 'dumb' search (i.e., evaluate EI at a large number of random points and take the best
    one). Naive search lives in: ComputeOptimalPointsToSampleViaLatinHypercubeSearch<>().
    **2. IMPLEMENTATION NOTES**
    a. This file has a few primary endpoints for KG optimization:
       i. ComputeKGOptimalPointsToSampleWithRandomStarts<>():
          Solves the q,p-KG problem.
          Takes in a gaussian_process describing the prior, domain, config, etc.; outputs the next best point(s) (experiment)
          to sample (run). Uses gradient descent.
       ii. ComputeKGOptimalPointsToSampleViaLatinHypercubeSearch<>():
           Estimates the q,p-KG problem.
           Takes in a gaussian_process describing the prior, domain, etc.; outputs the next best point(s) (experiment)
           to sample (run). Uses 'dumb' search.
       iii. ComputeKGOptimalPointsToSample<>() (Recommended):
            Solves the q,p-KG problem.
            Wraps the previous two items; relies on gradient descent and falls back to "dumb" search if it fails.
       .. NOTE::
           See ``gpp_knowledge_gradient_optimization.cpp``'s header comments for more detailed implementation notes.
           There are also several other functions with external linkage in this header; these
           are provided primarily to ease testing and to permit lower level access from python.
    b. See ``gpp_common.hpp`` header comments for additional implementation notes.
    **3. CITATIONS**
    a. Gaussian Processes for Machine Learning.
       Carl Edward Rasmussen and Christopher K. I. Williams. 2006.
       Massachusetts Institute of Technology.  55 Hayward St., Cambridge, MA 02142.
       http://www.gaussianprocess.org/gpml/ (free electronic copy)
    b. The Knowledge-Gradient Policy for Correlated Normal Beliefs.
       P.I. Frazier, W.B. Powell & S. Dayanik.
       INFORMS Journal on Computing, 2009.
    c. Differentiation of the Cholesky Algorithm.
       S. P. Smith. 1995.
       Journal of Computational and Graphical Statistics. Volume 4. Number 2. p134-147
    d. A Multi-points Criterion for Deterministic Parallel Global Optimization based on Gaussian Processes.
       David Ginsbourger, Rodolphe Le Riche, and Laurent Carraro.  2008.
       D´epartement 3MI. Ecole Nationale Sup´erieure des Mines. 158 cours Fauriel, Saint-Etienne, France.
       ginsbourger@emse.fr, leriche@emse.fr, carraro@emse.fr
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_ROBUST_KNOWLEDGE_GRADIENT_OPTIMIZATION_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_ROBUST_KNOWLEDGE_GRADIENT_OPTIMIZATION_HPP_

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include <stdlib.h>
#include <queue>

#include <boost/math/distributions/normal.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_exception.hpp"
#include "gpp_covariance.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimizer_parameters.hpp"
#include "gpp_random.hpp"

namespace optimal_learning {

template <typename DomainType>
struct RobustKnowledgeGradientState;
/*!\rst
  A class to encapsulate the computation of knowledge gradient and its spatial gradient. This class handles the
  general KG computation case using monte carlo integration; it can support q,p-KG optimization. It is designed to work
  with any GaussianProcess.  Additionally, this class has no state and within the context of KG optimization, it is
  meant to be accessed by const reference only.
  The random numbers needed for KG computation will be passed as parameters instead of contained as members to make
  multithreading more straightforward.
\endrst*/
template <typename DomainType>
class RobustKnowledgeGradientEvaluator final {
 public:
  using StateType = RobustKnowledgeGradientState<DomainType>;

  //! Minimum allowed variance value in the "1D" analytic EI computation.
  //! Values that are too small result in problems b/c we may compute ``std_dev/var`` (which is enormous
  //! if ``std_dev = 1.0e-150`` and ``var = 1.0e-300``) since this only arises when we fail to compute ``std_dev = var = 0.0``.
  //! Note: this is only relevant if noise = 0.0; this minimum will not affect EI computation with noise since this value
  //! is below the smallest amount of noise users can meaningfully add.
  //! This is the smallest possible value that prevents the denominator (best_so_far - mean) / sqrt(variance)
  //! from being 0. 1D analytic EI is simple and no other robustness considerations are needed.
  static constexpr double kMinimumVarianceEI = std::numeric_limits<double>::min();

  //! Minimum allowed variance value in the "1D" analytic grad EI computation.
  //! See kMinimumVarianceEI for more details.
  //! This value was chosen so its sqrt would be a little larger than GaussianProcess::kMinimumStdDev (by ~12x).
  //! The 150.0 was determined by numerical experiment with the setup in EIOnePotentialSampleEdgeCasesTest
  //! in order to find a setting that would be robust (no 0/0) while introducing minimal error.
  static constexpr double kMinimumVarianceGradEI = 150.0*Square(GaussianProcess::kMinimumStdDev);

  /*!\rst
    Constructs a KnowledgeGradientEvaluator object.  All inputs are required; no default constructor nor copy/assignment are allowed.
    \param
      :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
        that describes the underlying GP
      :discrete_pts[dim][num_pts]: the set of points to approximate the KG factor
      :num_pts: number of points in discrete_pts
      :num_mc_iterations: number of monte carlo iterations
      :best_so_far: best (minimum) objective function value (in ``points_sampled_value``)
  \endrst*/
  explicit RobustKnowledgeGradientEvaluator(const GaussianProcess& gaussian_process_in, const int num_fidelity,
                                             double const * discrete_pts,
                                             int num_pts,
                                             int num_mc_iterations,
                                             const DomainType& domain,
                                             const GradientDescentParameters& optimizer_parameters,
                                             double best_so_far,
                                             const double factor);

  RobustKnowledgeGradientEvaluator(RobustKnowledgeGradientEvaluator&& other);

  int dim() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim_;
  }

  int num_fidelity() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return num_fidelity_;
  }

  int num_mc_iterations() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return num_mc_iterations_;
  }

  double best_so_far() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return best_so_far_;
  }

  double factor() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return factor_;
  }

  GradientDescentParameters gradient_descent_params() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return GradientDescentParameters(optimizer_parameters_.num_multistarts, optimizer_parameters_.max_num_steps,
                                     optimizer_parameters_.max_num_restarts, optimizer_parameters_.num_steps_averaged,
                                     optimizer_parameters_.gamma, optimizer_parameters_.pre_mult,
                                     optimizer_parameters_.max_relative_change, optimizer_parameters_.tolerance);
  }

  DomainType domain() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return domain_;
  }

  int number_discrete_pts() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return num_pts_;
  }

  std::vector<double> discrete_pts_copy() const noexcept OL_WARN_UNUSED_RESULT {
    return discrete_pts_;
  }

  std::vector<double> discrete_points(double const * discrete_pts,
                                      int num_pts) const noexcept OL_WARN_UNUSED_RESULT {
    std::vector<double> result(num_pts*(dim_ - num_fidelity_));
    std::copy(discrete_pts, discrete_pts + num_pts*(dim_ - num_fidelity_), result.data());
    return result;
  }

  const GaussianProcess * gaussian_process() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return gaussian_process_;
  }

  /*!\rst
    Wrapper for ComputeKnowledgeGradient(); see that function for details.
  \endrst*/
  double ComputeObjectiveFunction(StateType * vf_state) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    return ComputeValueFunction(vf_state);
  }

  /*!\rst
    Wrapper for ComputeGradKnowledgeGradient(); see that function for details.
  \endrst*/
  void ComputeGradObjectiveFunction(StateType * vf_state, double * restrict grad_VF) const OL_NONNULL_POINTERS {
    ComputeGradValueFunction(vf_state, grad_VF);
  }

  /*!\rst
    Computes the knowledge gradient
    \param
      :kg_state[1]: properly configured state object
    \output
      :kg_state[1]: state with temporary storage modified; ``normal_rng`` modified
    \return
      the knowledge gradient from sampling ``points_to_sample`` with ``points_being_sampled`` concurrent experiments
  \endrst*/
  double ComputeValueFunction(StateType * vf_state) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT;

  /*!\rst
    Computes the (partial) derivatives of the knowledge gradient with respect to each point of ``points_to_sample``.
    As with ComputeKnowledgeGradient(), this computation accounts for the effect of ``points_being_sampled``
    concurrent experiments.
    ``points_to_sample`` is the "q" and ``points_being_sampled`` is the "p" in q,p-KG.
    \param
      :kg_state[1]: properly configured state object
    \output
      :kg_state[1]: state with temporary storage modified; ``normal_rng`` modified
      :grad_KG[dim][num_to_sample]: gradient of KG, ``\pderiv{KG(Xq \cup Xp)}{Xq_{d,i}}`` where ``Xq`` is ``points_to_sample``
      and ``Xp`` is ``points_being_sampled`` (grad KG from sampling ``points_to_sample`` with
      ``points_being_sampled`` concurrent experiments wrt each dimension of the points in ``points_to_sample``)
  \endrst*/
  double ComputeGradValueFunction(StateType * vf_state, double * restrict grad_KG) const OL_NONNULL_POINTERS;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(RobustKnowledgeGradientEvaluator);

 private:
  //! spatial dimension (e.g., entries per point of points_sampled)
  const int dim_;
  //! dim of the fidelity
  const int num_fidelity_;
  //! number of monte carlo iterations
  int num_mc_iterations_;

  //! best (minimum) objective function value (in points_sampled_value)
  double best_so_far_;
  //! factor to balance the immediate improvement vs. futural improvement
  double factor_;
  //! the gradient decsent parameter
  const GradientDescentParameters optimizer_parameters_;
  const DomainType domain_;

  //! normal distribution object
  const boost::math::normal_distribution<double> normal_;
  //! pointer to gaussian process used in KG computations
  const GaussianProcess * gaussian_process_;

  //! the set of points to approximate KG factor
  std::vector<double> discrete_pts_;
  //! number of points in discrete_pts
  const int num_pts_;
};

extern template class RobustKnowledgeGradientEvaluator<TensorProductDomain>;
extern template class RobustKnowledgeGradientEvaluator<SimplexIntersectTensorProductDomain>;

/*!\rst
  State object for KnowledgeGradientEvaluator.  This tracks the points being sampled in concurrent experiments
  (``points_being_sampled``) ALONG with the points currently being evaluated via knowledge gradient for future experiments
  (called ``points_to_sample``); these are the p and q of q,p-KG, respectively.  ``points_to_sample`` joined with
  ``points_being_sampled`` is stored in ``union_of_points`` in that order.
  This struct also tracks the state of the GaussianProcess that underlies the knowledge gradient computation: the GP state
  is built to handle the initial ``union_of_points``, and subsequent updates to ``points_to_sample`` in this object also update
  the GP state.
  This struct also holds a pointer to a random number generator needed for Monte Carlo integrated KG computations.
  .. WARNING::
       Users MUST guarantee that multiple state objects DO NOT point to the same RNG (in a multithreaded env).
  See general comments on State structs in ``gpp_common.hpp``'s header docs.
\endrst*/
template <typename DomainType>
struct RobustKnowledgeGradientState final {
  using EvaluatorType = RobustKnowledgeGradientEvaluator<DomainType>;

  /*!\rst
    Constructs an KnowledgeGradientState object with a specified source of randomness for the purpose of computing KG
    (and its gradient) over the specified set of points to sample.
    This establishes properly sized/initialized temporaries for KG computation, including dependent state from the
    associated Gaussian Process (which arrives as part of the kg_evaluator).
    .. WARNING:: This object is invalidated if the associated kg_evaluator is mutated.  SetupState() should be called to reset.
    .. WARNING::
         Using this object to compute gradients when ``configure_for_gradients`` := false results in UNDEFINED BEHAVIOR.
    \param
      :kg_evaluator: knowledge gradient evaluator object that specifies the parameters & GP for KG evaluation
      :points_to_sample[dim][num_to_sample]: points at which to evaluate KG and/or its gradient to check their value in future experiments (i.e., test points for GP predictions)
      :points_being_sampled[dim][num_being_sampled]: points being sampled in concurrent experiments
      :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-KG)
      :num_being_sampled: number of points being sampled in concurrent experiments (i.e., the "p" in q,p-KG)
      :configure_for_gradients: true if this object will be used to compute gradients, false otherwise
      :normal_rng[1]: pointer to a properly initialized\* NormalRNG object
    .. NOTE::
         \* The NormalRNG object must already be seeded.  If multithreaded computation is used for KG, then every state object
         must have a different NormalRNG (different seeds, not just different objects).
  \endrst*/
  explicit RobustKnowledgeGradientState(const EvaluatorType& kg_evaluator, double const * restrict points_to_sample,
                                           double const * restrict points_being_sampled, int num_to_sample_in,
                                           int num_being_sampled_in, int num_pts_in, int const * restrict gradients_in, int num_gradients_in,
                                           bool configure_for_gradients, NormalRNGInterface * normal_rng_in);

  RobustKnowledgeGradientState(RobustKnowledgeGradientState&& other);

  /*!\rst
    Create a vector with the union of points_to_sample and points_being_sampled (the latter is appended to the former).
    Note the l-value return. Assigning the return to a std::vector<double> or passing it as an argument to the ctor
    will result in copy-elision or move semantics; no copying/performance loss.
    \param:
      :points_to_sample[dim][num_to_sample]: points at which to evaluate KG and/or its gradient to check their value in future experiments (i.e., test points for GP predictions)
      :points_being_sampled[dim][num_being_sampled]: points being sampled in concurrent experiments
      :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-KG)
      :num_being_sampled: number of points being sampled in concurrent experiments (i.e., the "p" in q,p-KG)
      :dim: the number of spatial dimensions of each point array
    \return
      std::vector<double> with the union of the input arrays: points_being_sampled is *appended* to points_to_sample
  \endrst*/
  static std::vector<double> BuildUnionOfPoints(double const * restrict points_to_sample,
                                                double const * restrict points_being_sampled,
                                                int num_to_sample, int num_being_sampled,
                                                int dim) noexcept OL_WARN_UNUSED_RESULT {
    std::vector<double> union_of_points(dim*(num_to_sample + num_being_sampled));
    std::copy(points_to_sample, points_to_sample + dim*num_to_sample, union_of_points.data());
    std::copy(points_being_sampled, points_being_sampled + dim*num_being_sampled,
              union_of_points.data() + dim*num_to_sample);
    return union_of_points;
  }

  std::vector<double> SubsetData(double const * restrict union_of_points,
                                 int num_union, int num_fidelity) noexcept OL_WARN_UNUSED_RESULT {
    std::vector<double> subset_data((dim-num_fidelity)*num_union);
    for (int i=0; i<num_union; ++i){
      std::copy(union_of_points + i*dim, union_of_points+i*dim+dim-num_fidelity, subset_data.data()+i*(dim-num_fidelity));
    }
    return subset_data;
  }

  int GetProblemSize() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim*num_to_sample;
  }

  /*!\rst
    Get the ``points_to_sample``: potential future samples whose KG (and/or gradients) are being evaluated
    \output
      :points_to_sample[dim][num_to_sample]: potential future samples whose KG (and/or gradients) are being evaluated
  \endrst*/
  void GetCurrentPoint(double * restrict points_to_sample) const noexcept OL_NONNULL_POINTERS {
    std::copy(union_of_points.data(), union_of_points.data() + num_to_sample*dim, points_to_sample);
  }

  /*!\rst
    Change the potential samples whose KG (and/or gradient) are being evaluated.
    Update the state's derived quantities to be consistent with the new points.
    \param
      :kg_evaluator: expected improvement evaluator object that specifies the parameters & GP for KG evaluation
      :points_to_sample[dim][num_to_sample]: potential future samples whose KG (and/or gradients) are being evaluated
  \endrst*/
  void SetCurrentPoint(const EvaluatorType& kg_evaluator,
                       double const * restrict points_to_sample) OL_NONNULL_POINTERS;

  /*!\rst
    Configures this state object with new ``points_to_sample``, the location of the potential samples whose KG is to be evaluated.
    Ensures all state variables & temporaries are properly sized.
    Properly sets all dependent state variables (e.g., GaussianProcess's state) for KG evaluation.
    .. WARNING::
         This object's state is INVALIDATED if the ``kg_evaluator`` (including the GaussianProcess it depends on) used in
         SetupState is mutated! SetupState() should be called again in such a situation.
    \param
      :kg_evaluator: knowledge gradient evaluator object that specifies the parameters & GP for KG evaluation
      :points_to_sample[dim][num_to_sample]: potential future samples whose KG (and/or gradients) are being evaluated
  \endrst*/
  void SetupState(const EvaluatorType& kg_evaluator, double const * restrict points_to_sample);

  /*!\rst
    Pre-compute to_sample_mean_, and cholesky_to_sample_var
  \endrst*/
  void PreCompute(const EvaluatorType& kg_evaluator, double const * restrict points_to_sample);

  // size information
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  const int dim;
  //! number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-KG)
  const int num_to_sample;
  //! number of points being sampled concurrently (i.e., the "p" in q,p-KG)
  const int num_being_sampled;
  //! number of derivative terms desired (usually 0 for no derivatives or num_to_sample)
  const int num_derivatives;
  //! number of points in union_of_points: num_to_sample + num_being_sampled
  const int num_union;
  //! number of mc iterations
  const int num_iterations;

  // gradients index
  std::vector<int> gradients;
  // the number of gradients observations
  int num_gradients_to_sample;

  //! points currently being sampled; this is the union of the points represented by "q" and "p" in q,p-KG
  //! ``points_to_sample`` is stored first in memory, immediately followed by ``points_being_sampled``
  std::vector<double> union_of_points;

  //! discretized set in KG computation
  std::vector<double> subset_union_of_points;
  std::vector<double> discretized_set;

  //! gaussian process state
  GaussianProcess::StateType points_to_sample_state;

  //! random number generator
  NormalRNGInterface * normal_rng;

  // temporary storage: preallocated space used by KnowledgeGradientEvaluator's member functions
  //! the cholesky (``LL^T``) factorization of the GP variance evaluated at union_of_points
  std::vector<double> cholesky_to_sample_var;
  //! the gradient of the cholesky (``LL^T``) factorization of the GP variance evaluated at union_of_points
  // wrt union_of_points[0:num_to_sample]
  // (L_{d,*,*,k}^{-1} * L_{d,*,*,k} * L_{d,*,*,k}^{-1})^T
  std::vector<double> grad_chol_decomp;
  //! the mean of the GP evaluated at discrete_pts and the union_of_points
  std::vector<double> to_sample_mean_;
  //! tracks the aggregate grad KG from all mc iterations
  std::vector<double> aggregate;
  //! normal rng draws
  std::vector<double> normals;
  //! the best point
  std::vector<double> best_point;
  //! the inverse chol cov for the best point
  std::vector<double> chol_inverse_cov;
  //! grad_chol_inverse_cov
  std::vector<double> grad_chol_inverse_cov;
  //! the standard deviation at step two
  std::vector<double> best_standard_deviation;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(RobustKnowledgeGradientState);
};

extern template struct RobustKnowledgeGradientState<TensorProductDomain>;
extern template struct RobustKnowledgeGradientState<SimplexIntersectTensorProductDomain>;

struct PosteriorCVARState;
/*!\rst
  This is a specialization of the ExpectedImprovementEvaluator class for when the number of potential samples is 1; i.e.,
  ``num_to_sample == 1`` and the number of concurrent samples is 0; i.e. ``num_being_sampled == 0``.
  In other words, this class only supports the computation of 1,0-EI.  In this case, we have analytic formulas
  for computing EI and its gradient.
  Thus this class does not perform any explicit numerical integration, nor do its EI functions require access to a
  random number generator.
  This class's methods have some parameters that are unused or redundant.  This is so that the interface matches that of
  the more general ExpectedImprovementEvaluator.
  For other details, see ExpectedImprovementEvaluator for more complete description of what EI is and the outputs of
  EI and grad EI computations.
\endrst*/
class PosteriorCVAREvaluator final {
 public:
  using StateType = PosteriorCVARState;

  //! Minimum allowed variance value in the "1D" analytic EI computation.
  //! Values that are too small result in problems b/c we may compute ``std_dev/var`` (which is enormous
  //! if ``std_dev = 1.0e-150`` and ``var = 1.0e-300``) since this only arises when we fail to compute ``std_dev = var = 0.0``.
  //! Note: this is only relevant if noise = 0.0; this minimum will not affect EI computation with noise since this value
  //! is below the smallest amount of noise users can meaningfully add.
  //! This is the smallest possible value that prevents the denominator (best_so_far - mean) / sqrt(variance)
  //! from being 0. 1D analytic EI is simple and no other robustness considerations are needed.
  static constexpr double kMinimumVarianceEI = std::numeric_limits<double>::min();

  //! Minimum allowed variance value in the "1D" analytic grad EI computation.
  //! See kMinimumVarianceEI for more details.
  //! This value was chosen so its sqrt would be a little larger than GaussianProcess::kMinimumStdDev (by ~12x).
  //! The 150.0 was determined by numerical experiment with the setup in EIOnePotentialSampleEdgeCasesTest
  //! in order to find a setting that would be robust (no 0/0) while introducing minimal error.
  static constexpr double kMinimumVarianceGradEI = 150.0*Square(GaussianProcess::kMinimumStdDev);

  /*!\rst
    Constructs a OnePotentialSampleExpectedImprovementEvaluator object.  All inputs are required; no default constructor nor copy/assignment are allowed.
    \param
      :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
        that describes the underlying GP
      :best_so_far: best (minimum) objective function value (in ``points_sampled_value``)
  \endrst*/
  PosteriorCVAREvaluator(const GaussianProcess& gaussian_process_in);

  int dim() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim_;
  }

  const GaussianProcess * gaussian_process() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return gaussian_process_;
  }

  /*!\rst
    Wrapper for ComputeExpectedImprovement(); see that function for details.
  \endrst*/
  double ComputeObjectiveFunction(StateType * ps_state) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    return ComputePosteriorCVAR(ps_state);
  }

  /*!\rst
    Wrapper for ComputeGradExpectedImprovement(); see that function for details.
  \endrst*/
  void ComputeGradObjectiveFunction(StateType * ps_state, double * restrict grad_PS) const OL_NONNULL_POINTERS {
    ComputeGradPosteriorCVAR(ps_state, grad_PS);
  }

  /*!\rst
    Computes the expected improvement ``EI(Xs) = E_n[[f^*_n(X) - min(f(Xs_1),...,f(Xs_m))]^+]``
    Uses analytic formulas to evaluate the expected improvement.
    \param
      :ei_state[1]: properly configured state object
    \output
      :ei_state[1]: state with temporary storage modified
    \return
      the expected improvement from sampling ``point_to_sample``
  \endrst*/
  double ComputePosteriorCVAR(StateType * ps_state) const;

  /*!\rst
    Computes the (partial) derivatives of the expected improvement with respect to the point to sample.
    Uses analytic formulas to evaluate the spatial gradient of the expected improvement.
    \param
      :ei_state[1]: properly configured state object
    \output
      :ei_state[1]: state with temporary storage modified
      :grad_EI[dim]: gradient of EI, ``\pderiv{EI(x)}{x_d}``, where ``x`` is ``points_to_sample``
  \endrst*/
  void ComputeGradPosteriorCVAR(StateType * ps_state, double * restrict grad_PS) const;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PosteriorCVAREvaluator);

 private:
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  const int dim_;

  //! pointer to gaussian process used in EI computations
  const GaussianProcess * gaussian_process_;
};


/*!\rst
  State object for OnePotentialSampleExpectedImprovementEvaluator.  This tracks the *ONE* ``point_to_sample``
  being evaluated via expected improvement.
  This is just a special case of ExpectedImprovementState; see those class docs for more details.
  See general comments on State structs in ``gpp_common.hpp``'s header docs.
\endrst*/
struct PosteriorCVARState final {
  using EvaluatorType = PosteriorCVAREvaluator;

  /*!\rst
    Constructs an OnePotentialSampleExpectedImprovementState object for the purpose of computing EI
    (and its gradient) over the specified point to sample.
    This establishes properly sized/initialized temporaries for EI computation, including dependent state from the
    associated Gaussian Process (which arrives as part of the ``ei_evaluator``).
    .. WARNING::
         This object is invalidated if the associated ei_evaluator is mutated.  SetupState() should be called to reset.
    .. WARNING::
         Using this object to compute gradients when ``configure_for_gradients`` := false results in UNDEFINED BEHAVIOR.
    \param
      :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
      :point_to_sample[dim]: point at which to evaluate EI and/or its gradient to check their value in future experiments (i.e., test point for GP predictions)
      :configure_for_gradients: true if this object will be used to compute gradients, false otherwise
  \endrst*/
  PosteriorCVARState(const EvaluatorType& ps_evaluator, const int num_fidelity_in,
                     double const * restrict point_to_sample_in, bool configure_for_gradients);

  PosteriorCVARState(PosteriorCVARState&& other);

  int GetProblemSize() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim - num_fidelity;
  }

  std::vector<double> BuildUnionOfPoints(double const * restrict points_to_sample) noexcept OL_WARN_UNUSED_RESULT {
    std::vector<double> union_of_points(dim);
    std::copy(points_to_sample, points_to_sample + dim - num_fidelity, union_of_points.data());
    std::fill(union_of_points.data() + dim - num_fidelity, union_of_points.data() + dim, 1.0);
    return union_of_points;
  }

  /*!\rst
    Get ``point_to_sample``: the potential future sample whose EI (and/or gradients) is being evaluated
    \output
      :point_to_sample[dim]: potential sample whose EI is being evaluted
  \endrst*/
  void GetCurrentPoint(double * restrict point_to_sample_out) const noexcept OL_NONNULL_POINTERS {
    std::copy(point_to_sample.data(), point_to_sample.data() + dim - num_fidelity, point_to_sample_out);
  }

  /*!\rst
    Change the potential sample whose EI (and/or gradient) is being evaluated.
    Update the state's derived quantities to be consistent with the new point.
    \param
      :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
      :point_to_sample[dim]: potential future sample whose EI (and/or gradients) is being evaluated
  \endrst*/
  void SetCurrentPoint(const EvaluatorType& ps_evaluator,
                       double const * restrict point_to_sample_in) OL_NONNULL_POINTERS;

  /*!\rst
    Configures this state object with a new ``point_to_sample``, the location of the potential sample whose EI is to be evaluated.
    Ensures all state variables & temporaries are properly sized.
    Properly sets all dependent state variables (e.g., GaussianProcess's state) for EI evaluation.
    .. WARNING::
         This object's state is INVALIDATED if the ei_evaluator (including the GaussianProcess it depends on) used in
         SetupState is mutated! SetupState() should be called again in such a situation.
    \param
      :ei_evaluator: expected improvement evaluator object that specifies the parameters & GP for EI evaluation
      :point_to_sample[dim]: potential future sample whose EI (and/or gradients) is being evaluated
  \endrst*/
  void SetupState(const EvaluatorType& ps_evaluator, double const * restrict point_to_sample_in) OL_NONNULL_POINTERS;

  // size information
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  const int dim;
  //! dim of the fidelity
  const int num_fidelity;
  //! number of points to sample (i.e., the "q" in q,p-EI); MUST be 1
  const int num_to_sample = 1;
  //! number of derivative terms desired (usually 0 for no derivatives or num_to_sample)
  const int num_derivatives;

  //! point at which to evaluate EI and/or its gradient (e.g., to check its value in future experiments)
  std::vector<double> point_to_sample;

  //! gaussian process state
  GaussianProcess::StateType points_to_sample_state;
  // temporary storage: preallocated space used by OnePotentialSampleExpectedImprovementEvaluator's member functions
  //! the gradient of the GP mean evaluated at point_to_sample, wrt point_to_sample
  std::vector<double> grad_mu;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PosteriorCVARState);
};

/*!\rst
  Perform multistart gradient descent (MGD) to solve the q,p-EI problem (see ComputeOptimalPointsToSample and/or
  header docs), starting from ``num_multistarts`` points selected randomly from the within th domain.
  This function is a simple wrapper around ComputeOptimalPointsToSampleViaMultistartGradientDescent(). It additionally
  generates a set of random starting points and is just here for convenience when better initial guesses are not
  available.
  See ComputeOptimalPointsToSampleViaMultistartGradientDescent() for more details.
  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :optimizer_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_dynamic), chunk_size (0).
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-EI)
    :best_so_far: value of the best sample so far (must be ``min(points_sampled_value)``)
    :max_int_steps: maximum number of MC iterations
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    :normal_rng[thread_schedule.max_num_threads]: a vector of NormalRNG objects that provide
      the (pesudo)random source for MC integration
  \output
    :found_flag[1]: true if best_next_point corresponds to a nonzero EI
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :normal_rng[thread_schedule.max_num_threads]: NormalRNG objects will have their state changed due to random draws
    :best_next_point[dim][num_to_sample]: points yielding the best EI according to MGD
\endrst*/
template <typename DomainType>
void ComputeOptimalPosteriorCVAR(const GaussianProcess& gaussian_process, const int num_fidelity,
                                 const GradientDescentParameters& optimizer_parameters,
                                 const DomainType& domain, double const * restrict initial_guess, const int num_starts,
                                 bool * restrict found_flag, double * restrict best_next_point, double * best_function_value);
// template explicit instantiation declarations, see gpp_common.hpp header comments, item 6
extern template void ComputeOptimalPosteriorCVAR(const GaussianProcess& gaussian_process, const int num_fidelity,
                                          const GradientDescentParameters& optimizer_parameters,
                                          const TensorProductDomain& domain, double const * restrict initial_guess, const int num_starts,
                                          bool * restrict found_flag, double * restrict best_next_point, double * best_function_value);
extern template void ComputeOptimalPosteriorCVAR(const GaussianProcess& gaussian_process, const int num_fidelity,
                                          const GradientDescentParameters& optimizer_parameters,
                                          const SimplexIntersectTensorProductDomain& domain, double const * restrict initial_guess, const int num_starts,
                                          bool * restrict found_flag, double * restrict best_next_point, double * best_function_value);

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_ROBUST_KNOWLEDGE_GRADIENT_OPTIMIZATION_HPP_