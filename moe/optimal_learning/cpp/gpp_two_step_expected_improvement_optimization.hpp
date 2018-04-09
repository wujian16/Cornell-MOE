/*!
  \file gpp_two_step_expected_improvement_optimization.hpp
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

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_TWO_STEP_EXPECTED_IMPROVEMENT_OPTIMIZATION_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_TWO_STEP_EXPECTED_IMPROVEMENT_OPTIMIZATION_HPP_

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
struct TwoStepExpectedImprovementState;
/*!\rst
  A class to encapsulate the computation of knowledge gradient and its spatial gradient. This class handles the
  general KG computation case using monte carlo integration; it can support q,p-KG optimization. It is designed to work
  with any GaussianProcess.  Additionally, this class has no state and within the context of KG optimization, it is
  meant to be accessed by const reference only.

  The random numbers needed for KG computation will be passed as parameters instead of contained as members to make
  multithreading more straightforward.
\endrst*/
template <typename DomainType>
class TwoStepExpectedImprovementEvaluator final {
 public:
  using StateType = TwoStepExpectedImprovementState<DomainType>;

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
  explicit TwoStepExpectedImprovementEvaluator(const GaussianProcess& gaussian_process_in, const int num_fidelity,
                                               double const * discrete_pts,
                                               int num_pts,
                                               int num_mc_iterations,
                                               const DomainType& domain,
                                               const GradientDescentParameters& optimizer_parameters,
                                               double best_so_far);

  TwoStepExpectedImprovementEvaluator(TwoStepExpectedImprovementEvaluator&& other);

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

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(TwoStepExpectedImprovementEvaluator);

 private:
  //! spatial dimension (e.g., entries per point of points_sampled)
  const int dim_;
  //! dim of the fidelity
  const int num_fidelity_;
  //! number of monte carlo iterations
  int num_mc_iterations_;

  //! best (minimum) objective function value (in points_sampled_value)
  double best_so_far_;
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

extern template class TwoStepExpectedImprovementEvaluator<TensorProductDomain>;
extern template class TwoStepExpectedImprovementEvaluator<SimplexIntersectTensorProductDomain>;

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
struct TwoStepExpectedImprovementState final {
  using EvaluatorType = TwoStepExpectedImprovementEvaluator<DomainType>;

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
  explicit TwoStepExpectedImprovementState(const EvaluatorType& kg_evaluator, double const * restrict points_to_sample,
                                           double const * restrict points_being_sampled, int num_to_sample_in,
                                           int num_being_sampled_in, int num_pts_in, int const * restrict gradients_in, int num_gradients_in,
                                           bool configure_for_gradients, NormalRNGInterface * normal_rng_in);

  TwoStepExpectedImprovementState(TwoStepExpectedImprovementState&& other);

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
  //! the gradient of the GP mean evaluated at union_of_points, wrt union_of_points[0:num_to_sample]
  std::vector<double> grad_mu;
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
  //! the mean difference at step two
  std::vector<double> best_mean_difference;
  //! the standard deviation at step two
  std::vector<double> best_standard_deviation;
  //! the gradient of the step one best point wrt the step-one sampled point
  std::vector<double> step_one_gradient;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(TwoStepExpectedImprovementState);
};

extern template struct TwoStepExpectedImprovementState<TensorProductDomain>;
extern template struct TwoStepExpectedImprovementState<SimplexIntersectTensorProductDomain>;
}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_TWO_STEP_EXPECTED_IMPROVEMENT_OPTIMIZATION_HPP_