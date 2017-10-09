/*!
  \file gpp_knowledge_gradient_optimization.hpp
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

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_KNOWLEDGE_GRADIENT_OPTIMIZATION_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_KNOWLEDGE_GRADIENT_OPTIMIZATION_HPP_

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
struct KnowledgeGradientState;
/*!\rst
  A class to encapsulate the computation of knowledge gradient and its spatial gradient. This class handles the
  general KG computation case using monte carlo integration; it can support q,p-KG optimization. It is designed to work
  with any GaussianProcess.  Additionally, this class has no state and within the context of KG optimization, it is
  meant to be accessed by const reference only.

  The random numbers needed for KG computation will be passed as parameters instead of contained as members to make
  multithreading more straightforward.
\endrst*/
template <typename DomainType>
class KnowledgeGradientEvaluator final {
 public:
  using StateType = KnowledgeGradientState<DomainType>;
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
  explicit KnowledgeGradientEvaluator(const GaussianProcess& gaussian_process_in, const int num_fidelity,
                                      double const * discrete_pts,
                                      int num_pts,
                                      int num_mc_iterations,
                                      const DomainType& domain,
                                      const GradientDescentParameters& optimizer_parameters,
                                      double best_so_far);

  KnowledgeGradientEvaluator(KnowledgeGradientEvaluator&& other);

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
  double ComputeObjectiveFunction(StateType * kg_state) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    return ComputeKnowledgeGradient(kg_state);
  }

  /*!\rst
    Wrapper for ComputeGradKnowledgeGradient(); see that function for details.
  \endrst*/
  void ComputeGradObjectiveFunction(StateType * kg_state, double * restrict grad_KG) const OL_NONNULL_POINTERS {
    ComputeGradKnowledgeGradient(kg_state, grad_KG);
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
  double ComputeKnowledgeGradient(StateType * kg_state) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT;

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
  void ComputeGradKnowledgeGradient(StateType * kg_state, double * restrict grad_KG) const OL_NONNULL_POINTERS;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(KnowledgeGradientEvaluator);

 private:
  //! spatial dimension (e.g., entries per point of points_sampled)
  const int dim_;
  //! dim of the fidelity
  const int num_fidelity_;
  //! number of monte carlo iterations
  int num_mc_iterations_;

  //! best (minimum) objective function value (in points_sampled_value)
  double best_so_far_;

  const GradientDescentParameters optimizer_parameters_;
  const DomainType domain_;

  //! pointer to gaussian process used in KG computations
  const GaussianProcess * gaussian_process_;

  //! the set of points to approximate KG factor
  std::vector<double> discrete_pts_;
  //! number of points in discrete_pts
  const int num_pts_;
};

extern template class KnowledgeGradientEvaluator<TensorProductDomain>;
extern template class KnowledgeGradientEvaluator<SimplexIntersectTensorProductDomain>;

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
struct KnowledgeGradientState final {
  using EvaluatorType = KnowledgeGradientEvaluator<DomainType>;

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
  explicit KnowledgeGradientState(const EvaluatorType& kg_evaluator, double const * restrict points_to_sample,
                                  double const * restrict points_being_sampled, int num_to_sample_in,
                                  int num_being_sampled_in, int num_pts_in, int const * restrict gradients_in, int num_gradients_in,
                                  bool configure_for_gradients, NormalRNGInterface * normal_rng_in);

  KnowledgeGradientState(KnowledgeGradientState&& other);

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
  //! track the gradient of the cost function
  std::vector<double> gradcost;
  //! normal rng draws
  std::vector<double> normals;
  //! the best point
  std::vector<double> best_point;
  //! the inverse chol cov for the best point
  std::vector<double> chol_inverse_cov;
  //! grad_chol_inverse_cov
  std::vector<double> grad_chol_inverse_cov;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(KnowledgeGradientState);
};

extern template struct KnowledgeGradientState<TensorProductDomain>;
extern template struct KnowledgeGradientState<SimplexIntersectTensorProductDomain>;

struct PosteriorMeanState;
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
class PosteriorMeanEvaluator final {
 public:
  using StateType = PosteriorMeanState;

  /*!\rst
    Constructs a OnePotentialSampleExpectedImprovementEvaluator object.  All inputs are required; no default constructor nor copy/assignment are allowed.
    \param
      :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
        that describes the underlying GP
      :best_so_far: best (minimum) objective function value (in ``points_sampled_value``)
  \endrst*/
  PosteriorMeanEvaluator(const GaussianProcess& gaussian_process_in);

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
    return ComputePosteriorMean(ps_state);
  }

  /*!\rst
    Wrapper for ComputeGradExpectedImprovement(); see that function for details.
  \endrst*/
  void ComputeGradObjectiveFunction(StateType * ps_state, double * restrict grad_PS) const OL_NONNULL_POINTERS {
    ComputeGradPosteriorMean(ps_state, grad_PS);
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
  double ComputePosteriorMean(StateType * ps_state) const;

  /*!\rst
    Computes the (partial) derivatives of the expected improvement with respect to the point to sample.
    Uses analytic formulas to evaluate the spatial gradient of the expected improvement.
    \param
      :ei_state[1]: properly configured state object
    \output
      :ei_state[1]: state with temporary storage modified
      :grad_EI[dim]: gradient of EI, ``\pderiv{EI(x)}{x_d}``, where ``x`` is ``points_to_sample``
  \endrst*/
  void ComputeGradPosteriorMean(StateType * ps_state, double * restrict grad_PS) const;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PosteriorMeanEvaluator);

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
struct PosteriorMeanState final {
  using EvaluatorType = PosteriorMeanEvaluator;

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
  PosteriorMeanState(const EvaluatorType& ps_evaluator, const int num_fidelity_in, double const * restrict point_to_sample_in, bool configure_for_gradients);

  PosteriorMeanState(PosteriorMeanState&& other);

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

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PosteriorMeanState);
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
void ComputeOptimalPosteriorMean(const GaussianProcess& gaussian_process, const int num_fidelity,
                                 const GradientDescentParameters& optimizer_parameters,
                                 const DomainType& domain, double const * restrict initial_guess,
                                 bool * restrict found_flag, double * restrict best_next_point);
// template explicit instantiation declarations, see gpp_common.hpp header comments, item 6
extern template void ComputeOptimalPosteriorMean(const GaussianProcess& gaussian_process, const int num_fidelity,
                                                 const GradientDescentParameters& optimizer_parameters,
                                                 const TensorProductDomain& domain, double const * restrict initial_guess,
                                                 bool * restrict found_flag, double * restrict best_next_point);
extern template void ComputeOptimalPosteriorMean(const GaussianProcess& gaussian_process, const int num_fidelity,
                                                 const GradientDescentParameters& optimizer_parameters,
                                                 const SimplexIntersectTensorProductDomain& domain, double const * restrict initial_guess,
                                                 bool * restrict found_flag, double * restrict best_next_point);

/*!\rst
  Set up vector of KnowledgeGradientEvaluator::StateType.

  This is a utility function just for reducing code duplication.

  \param
    :kg_evaluator: evaluator object associated w/the state objects being constructed
    :points_to_sample[dim][num_to_sample]: initial points to load into state (must be a valid point for the problem);
      i.e., points at which to evaluate KG and/or its gradient
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrently experiments
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-KG)
    :num_being_sampled: number of points being sampled concurrently (i.e., the p in q,p-KG)
    :max_num_threads: maximum number of threads for use by OpenMP (generally should be <= # cores)
    :configure_for_gradients: true if these state objects will be used to compute gradients, false otherwise
    :state_vector[arbitrary]: vector of state objects, arbitrary size (usually 0)
    :normal_rng[max_num_threads]: a vector of NormalRNG objects that provide the (pesudo)random source for MC integration
  \output
    :state_vector[max_num_threads]: vector of states containing ``max_num_threads`` properly initialized state objects
\endrst*/
template <typename DomainType>
inline OL_NONNULL_POINTERS void SetupKnowledgeGradientState(
    const KnowledgeGradientEvaluator<DomainType>& kg_evaluator,
    double const * restrict points_to_sample,
    double const * restrict points_being_sampled,
    int num_to_sample,
    int num_being_sampled,
    int const * restrict gradients, int num_gradients,
    int max_num_threads,
    bool configure_for_gradients,
    NormalRNG * normal_rng,
    std::vector<typename KnowledgeGradientEvaluator<DomainType>::StateType> * state_vector) {
  state_vector->reserve(max_num_threads);
  for (int i = 0; i < max_num_threads; ++i) {
    state_vector->emplace_back(kg_evaluator, points_to_sample, points_being_sampled, num_to_sample,
                               num_being_sampled, kg_evaluator.num_mc_iterations(), gradients,
                               num_gradients, configure_for_gradients, normal_rng + i);
  }
}

/*!\rst
  Solve the q,p-KG problem (see ComputeKGOptimalPointsToSample and/or header docs) by optimizing the knowledge gradient.
  Optimization is done using restarted Gradient Descent, via GradientDescentOptimizer<...>::Optimize() from
  ``gpp_optimization.hpp``.  Please see that file for details on gradient descent and see gpp_optimizer_parameters.hpp
  for the meanings of the GradientDescentParameters.

  This function is just a simple wrapper that sets up the Evaluator's State and calls a general template for restarted GD.

  This function does not perform multistarting or employ any other robustness-boosting heuristcs; it only
  converges if the ``initial_guess`` is close to the solution. In general,
  ComputeOptimalPointsToSample() (see below) is preferred. This function is meant for:

  1. easier testing;
  2. if you really know what you're doing.

  Solution is guaranteed to lie within the region specified by ``domain``; note that this may not be a
  true optima (i.e., the gradient may be substantially nonzero).

  \param
    :kg_evaluator: reference to object that can compute ExpectedImprovement and its spatial gradient
    :optimizer_parameters: GradientDescentParameters object that describes the parameters controlling KG optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :initial_guess[dim][num_to_sample]: initial guess for gradient descent
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-KG)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-KG)
    :normal_rng[1]: a NormalRNG object that provides the (pesudo)random source for MC integration
  \output
    :normal_rng[1]: NormalRNG object will have its state changed due to random draws
    :next_point[dim][num_to_sample]: points yielding the best KG according to gradient descent
\endrst*/
template <typename KnowledgeGradientEvaluator, typename DomainType>
void RestartedGradientDescentKGOptimization(const KnowledgeGradientEvaluator& kg_evaluator,
                                            const GradientDescentParameters& optimizer_parameters,
                                            const DomainType& domain, double const * restrict initial_guess,
                                            double const * restrict points_being_sampled, int num_to_sample,
                                            int num_being_sampled, NormalRNG * normal_rng,
                                            double * restrict next_point) {
  if (unlikely(optimizer_parameters.max_num_restarts <= 0)) {
    return;
  }
  int dim = kg_evaluator.dim();
  int num_pts = kg_evaluator.number_discrete_pts();

  int num_derivatives = kg_evaluator.gaussian_process()->num_derivatives();
  std::vector<int> derivatives(kg_evaluator.gaussian_process()->derivatives());

  OL_VERBOSE_PRINTF("Knowledge Gradient Optimization via %s:\n", OL_CURRENT_FUNCTION_NAME);

  bool configure_for_gradients = true;
  typename KnowledgeGradientEvaluator::StateType kg_state(kg_evaluator, initial_guess,
                                                          points_being_sampled, num_to_sample,
                                                          num_being_sampled,
                                                          derivatives.data(), num_derivatives,
                                                          configure_for_gradients,
                                                          normal_rng);

  using RepeatedDomain = RepeatedDomain<DomainType>;
  RepeatedDomain repeated_domain(domain, num_to_sample);
  GradientDescentOptimizer<KnowledgeGradientEvaluator, RepeatedDomain> gd_opt;
  gd_opt.Optimize(kg_evaluator, optimizer_parameters, repeated_domain, &kg_state);
  kg_state.GetCurrentPoint(next_point);
}

/*!\rst
  Perform multistart gradient descent (MGD) to solve the q,p-KG problem (see ComputeKGOptimalPointsToSample and/or
  header docs).  Starts a GD run from each point in ``start_point_set``.  The point corresponding to the
  optimal KG\* is stored in ``best_next_point``.

  \* Multistarting is heuristic for global optimization. KG is not convex so this method may not find the true optimum.

  This function wraps MultistartOptimizer<>::MultistartOptimize() (see ``gpp_optimization.hpp``), which provides the multistarting
  component. Optimization is done using restarted Gradient Descent, via GradientDescentOptimizer<...>::Optimize() from
  ``gpp_optimization.hpp``. Please see that file for details on gradient descent and see ``gpp_optimizer_parameters.hpp``
  for the meanings of the GradientDescentParameters.

  This function (or its wrappers, e.g., ComputeOptimalPointsToSampleWithRandomStarts) are the primary entry-points for
  gradient descent based KG optimization in the ``optimal_learning`` library.

  Users may prefer to call ComputeKGOptimalPointsToSample(), which applies other heuristics to improve robustness.

  Currently, during optimization, we recommend that the coordinates of the initial guesses not differ from the
  coordinates of the optima by more than about 1 order of magnitude. This is a very (VERY!) rough guideline for
  sizing the domain and num_multistarts; i.e., be wary of sets of initial guesses that cover the space too sparsely.

  Solution is guaranteed to lie within the region specified by ``domain``; note that this may not be a
  true optima (i.e., the gradient may be substantially nonzero).

  .. WARNING::
       This function fails ungracefully if NO improvement can be found!  In that case,
       ``best_next_point`` will always be the first point in ``start_point_set``.
       ``found_flag`` will indicate whether this occured.

  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :optimizer_parameters: GradientDescentParameters object that describes the parameters controlling EI optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_dynamic), chunk_size (0).
    :start_point_set[dim][num_to_sample][num_multistarts]: set of initial guesses for MGD (one block of num_to_sample points per multistart)
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :discrete_pts[dim][num_pts]: points to approximate KG
    :num_multistarts: number of points in set of initial guesses
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-KG)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-KG)
    :num_pts: number of points in discrete_pts
    :best_so_far: value of the best mean value so far in discrete_pts
    :max_int_steps: maximum number of MC iterations
    :normal_rng[thread_schedule.max_num_threads]: a vector of NormalRNG objects that provide
      the (pesudo)random source for MC integration
    :noise: variance of measurement noise
  \output
    :normal_rng[thread_schedule.max_num_threads]: NormalRNG objects will have their state changed due to random draws
    :found_flag[1]: true if ``best_next_point`` corresponds to a nonzero KG
    :best_next_point[dim][num_to_sample]: points yielding the best KG according to MGD
\endrst*/
template <typename DomainType>
OL_NONNULL_POINTERS void ComputeKGOptimalPointsToSampleViaMultistartGradientDescent(
    const GaussianProcess& gaussian_process, const int num_fidelity,
    const GradientDescentParameters& optimizer_parameters,
    const GradientDescentParameters& optimizer_parameters_inner,
    const DomainType& domain, const DomainType& inner_domain,
    const ThreadSchedule& thread_schedule,
    double const * restrict start_point_set,
    double const * restrict points_being_sampled,
    double const * discrete_pts,
    int num_multistarts,
    int num_to_sample,
    int num_being_sampled,
    int num_pts,
    double best_so_far,
    int max_int_steps,
    NormalRNG * normal_rng,
    bool * restrict found_flag,
    double * restrict best_next_point) {
  if (unlikely(num_multistarts <= 0)) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "num_multistarts must be > 1", num_multistarts, 1);
  }

  bool configure_for_gradients = true;
  KnowledgeGradientEvaluator<DomainType> kg_evaluator(gaussian_process, num_fidelity, discrete_pts, num_pts, max_int_steps,
                                                      inner_domain, optimizer_parameters_inner, best_so_far);

  int num_derivatives = kg_evaluator.gaussian_process()->num_derivatives();
  std::vector<int> derivatives(kg_evaluator.gaussian_process()->derivatives());

  std::vector<typename KnowledgeGradientEvaluator<DomainType>::StateType> kg_state_vector;
  SetupKnowledgeGradientState(kg_evaluator, start_point_set, points_being_sampled,
                              num_to_sample, num_being_sampled, derivatives.data(), num_derivatives,
                              thread_schedule.max_num_threads, configure_for_gradients, normal_rng, &kg_state_vector);

  std::vector<double> KG_starting(num_multistarts);
  for (int i=0; i<num_multistarts; ++i){
    kg_state_vector[0].SetCurrentPoint(kg_evaluator, start_point_set + i*num_to_sample*gaussian_process.dim());
    KG_starting[i] = kg_evaluator.ComputeKnowledgeGradient(&kg_state_vector[0]);
  }

  std::priority_queue<std::pair<double, int>> q;
  int k = 20; // number of indices we need
  for (int i = 0; i < KG_starting.size(); ++i) {
    if (i < k){
      q.push(std::pair<double, int>(-KG_starting[i], i));
    }
    else{
      if (q.top().first > -KG_starting[i]){
        q.pop();
        q.push(std::pair<double, int>(-KG_starting[i], i));
      }
    }
  }

  std::vector<double> top_k_starting(k*num_to_sample*gaussian_process.dim());
  for (int i = 0; i < k; ++i) {
    int ki = q.top().second;
    for (int d = 0; d<num_to_sample*gaussian_process.dim(); ++d){
      top_k_starting[i*num_to_sample*gaussian_process.dim() + d] = start_point_set[ki*num_to_sample*gaussian_process.dim() + d];
    }
    q.pop();
  }

  // init winner to be first point in set and 'force' its value to be 0.0; we cannot do worse than this
  OptimizationIOContainer io_container(kg_state_vector[0].GetProblemSize(), -INFINITY, top_k_starting.data());

  using RepeatedDomain = RepeatedDomain<DomainType>;
  RepeatedDomain repeated_domain(domain, num_to_sample);
  GradientDescentOptimizer<KnowledgeGradientEvaluator<DomainType>, RepeatedDomain> gd_opt;
  MultistartOptimizer<GradientDescentOptimizer<KnowledgeGradientEvaluator<DomainType>, RepeatedDomain> > multistart_optimizer;
  multistart_optimizer.MultistartOptimize(gd_opt, kg_evaluator, optimizer_parameters,
                                          repeated_domain, thread_schedule, top_k_starting.data(),
                                          k, kg_state_vector.data(), nullptr, &io_container);
  *found_flag = io_container.found_flag;
  std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
}

/*!\rst
  Function to evaluate Knowledge Gradient (q,p-KG) over a specified list of ``num_multistarts`` points.
  Optionally outputs the KG at each of these points.
  Outputs the point of the set obtaining the maximum KG value.

  Generally gradient descent is preferred but when they fail to converge this may be the only "robust" option.
  This function is also useful for plotting or debugging purposes (just to get a bunch of KG values).

  This function is just a wrapper that builds the required state objects and a NullOptimizer object and calls
  MultistartOptimizer<...>::MultistartOptimize(...); see gpp_optimization.hpp.

  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_static), chunk_size (0).
    :initial_guesses[dim][num_to_sample][num_multistarts]: list of points at which to compute KG
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :discrete_pts[dim][num_pts]: points to approximate KG
    :num_multistarts: number of points to check
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-KG)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-KG)
    :num_pts: number of points in discrete_pts
    :best_so_far: value of the best mean value so far in discrete_pts
    :max_int_steps: maximum number of MC iterations
    :normal_rng[thread_schedule.max_num_threads]: a vector of NormalRNG objects that provide
      the (pesudo)random source for MC integration
    :noise: variance of measurement noise
  \output
    :found_flag[1]: true if best_next_point corresponds to a nonzero KG
    :normal_rng[thread_schedule.max_num_threads]: NormalRNG objects will have their state changed due to random draws
    :function_values[num_multistarts]: KG evaluated at each point of ``initial_guesses``, in the same order as
      ``initial_guesses``; never dereferenced if nullptr
    :best_next_point[dim][num_to_sample]: points yielding the best KG according to dumb search
\endrst*/
template <typename DomainType>
void EvaluateKGAtPointList(const GaussianProcess& gaussian_process, const int num_fidelity,
                           const GradientDescentParameters& optimizer_parameters_inner,
                           const DomainType& domain, const DomainType& inner_domain, const ThreadSchedule& thread_schedule,
                           double const * restrict initial_guesses,
                           double const * restrict points_being_sampled,
                           double const * discrete_pts, int num_multistarts, int num_to_sample,
                           int num_being_sampled, int num_pts, double best_so_far,
                           int max_int_steps, bool * restrict found_flag, NormalRNG * normal_rng,
                           double * restrict function_values,
                           double * restrict best_next_point) {
  if (unlikely(num_multistarts <= 0)) {
    OL_THROW_EXCEPTION(LowerBoundException<int>, "num_multistarts must be > 1", num_multistarts, 1);
  }

  using DomainType_dummy = DummyDomain;
  DomainType_dummy dummy_domain;
  bool configure_for_gradients = false;

  KnowledgeGradientEvaluator<DomainType> kg_evaluator(gaussian_process, num_fidelity, discrete_pts, num_pts, max_int_steps,
                                                      inner_domain, optimizer_parameters_inner, best_so_far);

  int num_derivatives = kg_evaluator.gaussian_process()->num_derivatives();
  std::vector<int> derivatives(kg_evaluator.gaussian_process()->derivatives());

  std::vector<typename KnowledgeGradientEvaluator<DomainType>::StateType> kg_state_vector;
  SetupKnowledgeGradientState(kg_evaluator, initial_guesses, points_being_sampled,
                              num_to_sample, num_being_sampled, derivatives.data(), num_derivatives,
                              thread_schedule.max_num_threads, configure_for_gradients, normal_rng, &kg_state_vector);

  // init winner to be first point in set and 'force' its value to be -INFINITY; we cannot do worse than this
  OptimizationIOContainer io_container(kg_state_vector[0].GetProblemSize(), -INFINITY, initial_guesses);

  NullOptimizer<KnowledgeGradientEvaluator<DomainType>, DomainType_dummy> null_opt;
  typename NullOptimizer<KnowledgeGradientEvaluator<DomainType>, DomainType_dummy>::ParameterStruct null_parameters;
  MultistartOptimizer<NullOptimizer<KnowledgeGradientEvaluator<DomainType>, DomainType_dummy> > multistart_optimizer;
  multistart_optimizer.MultistartOptimize(null_opt, kg_evaluator, null_parameters,
                                          dummy_domain, thread_schedule, initial_guesses,
                                          num_multistarts, kg_state_vector.data(), function_values, &io_container);
  *found_flag = io_container.found_flag;
  std::copy(io_container.best_point.begin(), io_container.best_point.end(), best_next_point);
}

/*!\rst
  Perform multistart gradient descent (MGD) to solve the q,p-KG problem (see ComputeKGOptimalPointsToSample and/or
  header docs), starting from ``num_multistarts`` points selected randomly from the within the domain.

  This function is a simple wrapper around ComputeOptimalPointsToSampleViaMultistartGradientDescent(). It additionally
  generates a set of random starting points and is just here for convenience when better initial guesses are not
  available.

  See ComputeKGOptimalPointsToSampleViaMultistartGradientDescent() for more details.

  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :optimizer_parameters: GradientDescentParameters object that describes the parameters controlling KG optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_dynamic), chunk_size (0).
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :discrete_pts[dim][num_pts]: points to approximate KG
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-KG)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-KG)
    :num_pts: number of points in discrete_pts
    :best_so_far: value of the best mean value so far in discrete_pts
    :max_int_steps: maximum number of MC iterations
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    :normal_rng[thread_schedule.max_num_threads]: a vector of NormalRNG objects that provide
      the (pesudo)random source for MC integration
    :noise: variance of measurement noise
  \output
    :found_flag[1]: true if best_next_point corresponds to a nonzero KG
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :normal_rng[thread_schedule.max_num_threads]: NormalRNG objects will have their state changed due to random draws
    :best_next_point[dim][num_to_sample]: points yielding the best KG according to MGD
\endrst*/
template <typename DomainType>
void ComputeKGOptimalPointsToSampleWithRandomStarts(const GaussianProcess& gaussian_process, const int num_fidelity,
                                                    const GradientDescentParameters& optimizer_parameters,
                                                    const GradientDescentParameters& optimizer_parameters_inner,
                                                    const DomainType& domain, const DomainType& inner_domain, const ThreadSchedule& thread_schedule,
                                                    double const * restrict points_being_sampled, double const * discrete_pts,
                                                    int num_to_sample, int num_being_sampled, int num_pts,
                                                    double best_so_far, int max_int_steps, bool * restrict found_flag,
                                                    UniformRandomGenerator * uniform_generator, NormalRNG * normal_rng,
                                                    double * restrict best_next_point) {
  std::vector<double> starting_points(gaussian_process.dim()*optimizer_parameters.num_multistarts*num_to_sample);

  // GenerateUniformPointsInDomain() is allowed to return fewer than the requested number of multistarts
  RepeatedDomain<DomainType> repeated_domain(domain, num_to_sample);
  int num_multistarts = repeated_domain.GenerateUniformPointsInDomain(optimizer_parameters.num_multistarts,
                                                                      uniform_generator, starting_points.data());

  ComputeKGOptimalPointsToSampleViaMultistartGradientDescent(gaussian_process, num_fidelity, optimizer_parameters, optimizer_parameters_inner, domain,
                                                             inner_domain, thread_schedule, starting_points.data(),
                                                             points_being_sampled, discrete_pts, num_multistarts,
                                                             num_to_sample, num_being_sampled, num_pts,
                                                             best_so_far, max_int_steps,
                                                             normal_rng, found_flag, best_next_point);
#ifdef OL_WARNING_PRINT
  if (false == *found_flag) {
    OL_WARNING_PRINTF("WARNING: %s DID NOT CONVERGE\n", OL_CURRENT_FUNCTION_NAME);
    OL_WARNING_PRINTF("First multistart point was returned:\n");
    PrintMatrixTrans(starting_points.data(), num_to_sample, gaussian_process.dim());
  }
#endif
}

/*!\rst
  Perform a random, naive search to "solve" the q,p-KG problem (see ComputeKGOptimalPointsToSample and/or
  header docs).  Evaluates KG at ``num_multistarts`` points (e.g., on a latin hypercube) to find the
  point with the best KG value.

  Generally gradient descent is preferred but when they fail to converge this may be the only "robust" option.

  Solution is guaranteed to lie within the region specified by ``domain``; note that this may not be a
  true optima (i.e., the gradient may be substantially nonzero).

  Wraps EvaluateKGAtPointList(); constructs the input point list with a uniform random sampling from the given Domain object.

  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_static), chunk_size (0).
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :discrete_pts[dim][num_pts]: points to approximate KG
    :num_multistarts: number of random points to check
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-KG)
    :num_being_sampled: number of points being sampled concurrently (i.e., the "p" in q,p-KG)
    :num_pts: number of points in discrete_pts
    :best_so_far: value of the best mean value so far in discrete_pts
    :max_int_steps: maximum number of MC iterations
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    :normal_rng[thread_schedule.max_num_threads]: a vector of NormalRNG objects that provide
      the (pesudo)random source for MC integration
    :noise: variance of measurement noise
  \output
    found_flag[1]: true if best_next_point corresponds to a nonzero KG
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :normal_rng[thread_schedule.max_num_threads]: NormalRNG objects will have their state changed due to random draws
    :best_next_point[dim][num_to_sample]: points yielding the best KG according to dumb search
\endrst*/
template <typename DomainType>
void ComputeKGOptimalPointsToSampleViaLatinHypercubeSearch(const GaussianProcess& gaussian_process, const int num_fidelity,
                                                           const GradientDescentParameters& optimizer_parameters_inner,
                                                           const DomainType& domain, const DomainType& inner_domain,
                                                           const ThreadSchedule& thread_schedule,
                                                           double const * restrict points_being_sampled,
                                                           double const * discrete_pts,
                                                           int num_multistarts, int num_to_sample,
                                                           int num_being_sampled, int num_pts, double best_so_far,
                                                           int max_int_steps,
                                                           bool * restrict found_flag,
                                                           UniformRandomGenerator * uniform_generator,
                                                           NormalRNG * normal_rng,
                                                           double * restrict best_next_point) {
  std::vector<double> initial_guesses(gaussian_process.dim()*num_multistarts*num_to_sample);
  RepeatedDomain<DomainType> repeated_domain(domain, num_to_sample);
  num_multistarts = repeated_domain.GenerateUniformPointsInDomain(num_multistarts, uniform_generator,
                                                                  initial_guesses.data());

  EvaluateKGAtPointList(gaussian_process, num_fidelity, optimizer_parameters_inner, domain, inner_domain, thread_schedule, initial_guesses.data(),
                        points_being_sampled, discrete_pts, num_multistarts, num_to_sample,
                        num_being_sampled, num_pts, best_so_far, max_int_steps,
                        found_flag, normal_rng, nullptr, best_next_point);
}


/*!\rst
  Solve the q,p-KG problem (see header docs) by optimizing the knowledge gradient.
  Uses multistart gradient descent, "dumb" search, and/or other heuristics to perform the optimization.

  This is the primary entry-point for KG optimization in the optimal_learning library. It offers our best shot at
  improving robustness by combining higher accuracy methods like gradient descent with fail-safes like random/grid search.

  Returns the optimal set of q points to sample CONCURRENTLY by solving the q,p-KG problem.  That is, we may want to run 4
  experiments at the same time and maximize the KG across all 4 experiments at once while knowing of 2 ongoing experiments
  (4,2-KG). This function handles this use case. Evaluation of q,p-KG (and its gradient) for q > 1 or p > 1 is expensive
  (requires monte-carlo iteration), so this method is usually very expensive.

  Wraps ComputeKGOptimalPointsToSampleWithRandomStarts() and ComputeKGOptimalPointsToSampleViaLatinHypercubeSearch().

  Compared to ComputeHeuristicPointsToSample() (``gpp_heuristic_expected_improvement_optimization.hpp``), this function
  makes no external assumptions about the underlying objective function. Instead, it utilizes a feature of the
  GaussianProcess that allows the GP to account for ongoing/incomplete experiments.

  .. NOTE:: These comments were copied into multistart_knowledge_gradient_optimization() in cpp_wrappers/knowledge_gradient.py.

  \param
    :gaussian_process: GaussianProcess object (holds ``points_sampled``, ``values``, ``noise_variance``, derived quantities)
      that describes the underlying GP
    :optimizer_parameters: GradientDescentParameters object that describes the parameters controlling KG optimization
      (e.g., number of iterations, tolerances, learning rate)
    :domain: object specifying the domain to optimize over (see ``gpp_domain.hpp``)
    :thread_schedule: struct instructing OpenMP on how to schedule threads; i.e., (suggestions in parens)
      max_num_threads (num cpu cores), schedule type (omp_sched_dynamic), chunk_size (0).
    :points_being_sampled[dim][num_being_sampled]: points that are being sampled in concurrent experiments
    :discrete_pts[dim][num_pts]: points to approximate KG
    :num_to_sample: how many simultaneous experiments you would like to run (i.e., the q in q,p-KG)
    :num_being_sampled: number of points being sampled concurrently (i.e., the p in q,p-KG)
    :num_pts: number of points in discrete_pts
    :best_so_far: value of the best mean value so far in discrete_pts
    :max_int_steps: maximum number of MC iterations
    :lhc_search_only: whether to ONLY use latin hypercube search (and skip gradient descent EI opt)
    :num_lhc_samples: number of samples to draw if/when doing latin hypercube search
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    :normal_rng[thread_schedule.max_num_threads]: a vector of NormalRNG objects that provide
      the (pesudo)random source for MC integration
    :noise: variance of measurement noise
  \output
    :found_flag[1]: true if best_points_to_sample corresponds to a nonzero KG if sampled simultaneously
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :normal_rng[thread_schedule.max_num_threads]: NormalRNG objects will have their state changed due to random draws
    :best_points_to_sample[num_to_sample*dim]: point yielding the best KG according to MGD
\endrst*/
template <typename DomainType>
void ComputeKGOptimalPointsToSample(const GaussianProcess& gaussian_process, const int num_fidelity,
                                    const GradientDescentParameters& optimizer_parameters,
                                    const GradientDescentParameters& optimizer_parameters_inner,
                                    const DomainType& domain, const DomainType& inner_domain, const ThreadSchedule& thread_schedule,
                                    double const * restrict points_being_sampled,
                                    double const * discrete_pts,
                                    int num_to_sample, int num_being_sampled,
                                    int num_pts, double best_so_far,
                                    int max_int_steps, bool lhc_search_only,
                                    int num_lhc_samples, bool * restrict found_flag,
                                    UniformRandomGenerator * uniform_generator,
                                    NormalRNG * normal_rng, double * restrict best_points_to_sample);
// template explicit instantiation declarations, see gpp_common.hpp header comments, item 6
extern template void ComputeKGOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const int num_fidelity, const GradientDescentParameters& optimizer_parameters,
    const GradientDescentParameters& optimizer_parameters_inner,
    const TensorProductDomain& domain, const TensorProductDomain& inner_domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled, double const * discrete_pts,
    int num_to_sample, int num_being_sampled,
    int num_pts, double best_so_far, int max_int_steps, bool lhc_search_only,
    int num_lhc_samples, bool * restrict found_flag, UniformRandomGenerator * uniform_generator,
    NormalRNG * normal_rng, double * restrict best_points_to_sample);
extern template void ComputeKGOptimalPointsToSample(
    const GaussianProcess& gaussian_process, const int num_fidelity, const GradientDescentParameters& optimizer_parameters,
    const GradientDescentParameters& optimizer_parameters_inner,
    const SimplexIntersectTensorProductDomain& domain, const SimplexIntersectTensorProductDomain& inner_domain, const ThreadSchedule& thread_schedule,
    double const * restrict points_being_sampled,double const * discrete_pts,
    int num_to_sample, int num_being_sampled,
    int num_pts, double best_so_far, int max_int_steps, bool lhc_search_only, int num_lhc_samples, bool * restrict found_flag,
    UniformRandomGenerator * uniform_generator, NormalRNG * normal_rng, double * restrict best_points_to_sample);
}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_HEURISTIC_EXPECTED_IMPROVEMENT_OPTIMIZATION_HPP_