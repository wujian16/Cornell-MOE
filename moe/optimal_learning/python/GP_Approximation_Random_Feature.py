import numpy          as np
import numpy.random   as npr
import scipy.linalg   as spla
import numpy.linalg   as npla
import scipy.optimize as spo

try:
    import nlopt
except:
    nlopt_imported = False
else:
    nlopt_imported = True

def chol2inv(chol):
    return spla.cho_solve((chol, False), np.eye(chol.shape[0]))

"""
See Miguel's paper (http://arxiv.org/pdf/1406.2541v1.pdf) section 2.1 and Appendix A
Returns a function the samples from the approximation...
if testing=True, it does not return the result but instead the random cosine for testing only
We express the kernel as an expectation. But then we approximate the expectation with a weighted sum
theta are the coefficients for this weighted sum. that is why we take the dot product of theta at the end
we also need to scale at the end so that it's an average of the random features.
if use_woodbury_if_faster is False, it never uses the woodbury version
"""
def sample_gp_with_random_features(gp, nFeatures, use_woodbury_if_faster=True):

    d = gp.dim
    N_data = gp.num_sampled
    N_derivatives = gp.num_derivatives

    nu2 = gp.noise_variance

    covariance = gp.get_covariance_copy()
    hyperparameters = covariance.get_hyperparameters()

    sigma2 = hyperparameters[0]  # the kernel amplitude

    # We draw the random features
    W = npr.randn(nFeatures, d) / hyperparameters[1:]
    b = npr.uniform(low=0, high=2*np.pi, size=nFeatures)[:,None]

    randomness = npr.randn(nFeatures)

    # W has size nFeatures by d
    # tDesignMatrix has size Nfeatures by Ndata
    # woodbury has size Ndata by Ndata
    # z is a vector of length nFeatures

    if gp._points_sampled.shape[0]>0:
        tDesignMatrix = np.sqrt(2.0 * sigma2 / nFeatures) * np.cos(np.dot(W, \
                                           gp._points_sampled.T) + b) / np.sqrt(nu2[0])
        index = 1
        for i in gp.derivatives:
            temp_design = -np.sqrt(2.0 * sigma2 / nFeatures) * W[:,[i]] * np.sin(np.dot(W, \
                                           gp._points_sampled.T) + b) /np.sqrt(nu2[index])
            index += 1
            tDesignMatrix = np.concatenate((tDesignMatrix, temp_design), axis=1)

        observed_value =  (gp._points_sampled_value / np.sqrt(nu2)).flatten('F')

        if use_woodbury_if_faster and N_data*(1+N_derivatives) < nFeatures:
            # you can do things in cost N^2d instead of d^3 by doing this woodbury thing

            # We obtain the posterior on the coefficients
            woodbury = np.dot(tDesignMatrix.T, tDesignMatrix) + np.eye(N_data*(1+N_derivatives))
            chol_woodbury = spla.cholesky(woodbury)
            # inverseWoodbury = chol2inv(chol_woodbury)
            z = np.dot(tDesignMatrix, observed_value)
            # m = z - np.dot(tDesignMatrix, np.dot(inverseWoodbury, np.dot(tDesignMatrix.T, z)))
            m = z - np.dot(tDesignMatrix, spla.cho_solve((chol_woodbury, False), np.dot(tDesignMatrix.T, z)))
            # (above) alternative to original but with cho_solve

            # z = np.dot(tDesignMatrix, gp.observed_values / nu2)
            # m = np.dot(np.eye(nFeatures) - \
            # np.dot(tDesignMatrix, spla.cho_solve((chol_woodbury, False), tDesignMatrix.T)), z)

            # woodbury has size N_data by N_data
            D, U = npla.eigh(woodbury)
            # sort the eigenvalues (not sure if this matters)
            idx = D.argsort()[::-1] # in decreasing order instead of increasing
            D = D[idx]
            U = U[:,idx]
            R = 1.0 / (np.sqrt(D) * (np.sqrt(D) + 1))
            # R = 1.0 / (D + np.sqrt(D*1))

            # We sample from the posterior of the coefficients
            theta = randomness - \
                    np.dot(tDesignMatrix, np.dot(U, (R * np.dot(U.T, np.dot(tDesignMatrix.T, randomness))))) + m

        else:
            # all you are doing here is sampling from the posterior of the linear model
            # that approximates the GP
            # Sigma = matrixInverse(np.dot(tDesignMatrix, tDesignMatrix.T) / nu2 + np.eye(nFeatures))
            # m = np.dot(Sigma, np.dot(tDesignMatrix, gp.observed_values / nu2))
            # theta = m + np.dot(randomness, spla.cholesky(Sigma, lower=False)).T

            # Sigma = matrixInverse(np.dot(tDesignMatrix, tDesignMatrix.T) + nu2*np.eye(nFeatures))
            # m = np.dot(Sigma, np.dot(tDesignMatrix, gp.observed_values))
            # theta = m + np.dot(randomness, spla.cholesky(Sigma*nu2, lower=False)).T

            chol_Sigma_inverse = spla.cholesky(np.dot(tDesignMatrix, tDesignMatrix.T) + np.eye(nFeatures))
            Sigma = chol2inv(chol_Sigma_inverse)
            m = spla.cho_solve((chol_Sigma_inverse, False), np.dot(tDesignMatrix, observed_value))
            theta = m + np.dot(randomness, spla.cholesky(Sigma, lower=False)).T


    else:
        # We sample from the prior -- same for Matern
        theta = npr.randn(nFeatures)

    def wrapper(x, gradient):
        # the argument "gradient" is
        # not the usual compute_grad that computes BOTH when true
        # here it only computes the objective when true

        if x.ndim == 1:
            x = x[None,:]

        if not gradient:
            result = np.dot(theta.T, np.sqrt(2.0 * sigma2 / nFeatures) * np.cos(np.dot(W, x.T) + b))
            if result.size == 1:
                result = float(result) # if the answer is just a number, take it out of the numpy array wrapper
                # (failure to do so messed up NLopt and it only gives a cryptic error message)
            return result
        else:
            grad = np.dot(theta.T, -np.sqrt(2.0 * sigma2 / nFeatures) * np.sin(np.dot(W, x.T) + b) * W)
            return grad

    return wrapper



"""
Given some approximations to the GP sample, find its minimum
We do that by first evaluating it on a grid, taking the best, and using that to
initialize an optimization. If nothing on the grid satisfies the constraint, then
we return None wrapper_functions should be a dict with keys 'objective' and optionally 'constraints'
"""
# find MINIMUM if minimize=True, else find a maximum
def global_optimization_of_GP_approximation(funs, domain_bounds, num_dims, grid):

    assert num_dims == grid.shape[1]

    # print 'evaluating on grid'
    # First, evaluate on a grid and see what you get
    obj_evals = funs(grid, gradient=False)

    best_guess_index = np.argmin(obj_evals)
    best_guess_value = np.min(obj_evals)

    x_initial = grid[best_guess_index]

    f       = lambda x: float(funs(x, gradient=False))
    f_prime = lambda x: funs(x, gradient=True).flatten()

    bounds = [(bounds[0], bounds[1]) for bounds in domain_bounds]

    opt_x = spo.fmin_slsqp(f, x_initial.copy(), bounds=bounds, disp=0, fprime=f_prime, f_ieqcons=None, fprime_ieqcons=None)

    if f(opt_x) < best_guess_value:
        return opt_x[None]
    else:
        print('SLSQP failed when optimizing x*')
        return x_initial[None]


def sample_from_global_optima(gp, nFeatures, domain_bounds, grid, nPoints):
    points = np.zeros((nPoints, gp.dim))
    for num in range(nPoints):
        points[num, :] = global_optimization_of_GP_approximation(sample_gp_with_random_features(gp, nFeatures), domain_bounds, gp.dim, grid)
    return points




