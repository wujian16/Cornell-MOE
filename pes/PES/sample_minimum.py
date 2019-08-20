import numpy as np
import numpy.random as npr
import scipy.optimize as spo
import math
from PES.utilities import *



#This function is to approximately sample from the conditional distribution of the global minimum. which
#is the first part of the PES function. It corresponds to the Section 2.1 and Appendix A of the papaer.
#Parameters: @num_features: the number of features that we would like to use for feature mapping. It is 
#                          the "m" in the paper
#            @d: the dimensions of the objective function
#            @nObservations: observations we have, which is the X_n
#			 @value_of_nObservations: the function values of the observations
#			 @alpha: cooresponding to the same alpha in the paper
#			 @l: the lengthscale parameter of the squared exponential kernel
#			 @noise: the white noise of the function, which is obtained from simulation
#			 @initial_point: the point used as the intial point of the optimization of the approximate
#			 				 posterior distribution 
#			 @optimize_method: method used to optimize the approximate posterior distribution. User can 
#			 				   choose any method specified in the scipy.optimize.minimize     
def sample_min_with_randFeatures(num_features, d, nObservations, value_of_nObservations, alpha, l, noise, initial_point, optimize_method = 'L-BFGS-B', maximize = False, bnds = None):
    
    l = np.array([l, ]*num_features)
    W = np.divide(npr.randn(num_features, d), l)
    b = 2*np.pi*npr.uniform(0,1,num_features)
    num_of_nObservations = len(nObservations)
    b_for_nObservations = np.array([b,]*num_of_nObservations).T

    
    
    #Phi with dimensions mxn
    phi_vector_inverse = math.sqrt(2*alpha/num_features)*np.cos(np.dot(W, nObservations.T) + b_for_nObservations)

 
    A = np.divide(np.dot(phi_vector_inverse, phi_vector_inverse.T), noise) + np.eye(num_features)
    A_inverse = compute_inverse(A)
    mean_of_post_theta = np.divide(np.dot(np.dot(A_inverse, phi_vector_inverse), value_of_nObservations), noise)
    mean_of_post_theta = np.squeeze(np.asarray(mean_of_post_theta))
    variance_of_post_theta = A_inverse

    sample_of_theta = npr.multivariate_normal(mean_of_post_theta, variance_of_post_theta)
    
    
    def function_to_optimize(x):
        phi_x = math.sqrt(2*alpha/num_features)*np.cos(np.dot(W, x.T) + b)
        approx_function = np.dot(phi_x.T, sample_of_theta)
        if maximize:
            approx_function = -approx_function
        return approx_function
    
    
    if optimize_method not in ('CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact', 'trust-constr'):
        result = spo.minimize(function_to_optimize, initial_point, method= optimize_method, options={'gtol': 10**(-50), 'maxiter':2000}, bounds = bnds)
    else:
    	#We pass the gradient to the spo.minimize to make the optimization faster
        def gradient_to_help(x):
            temp = np.sin(np.dot(W, x.T) + b)
            temp_multiple = np.array([temp, ]*d).T
            gradient_of_phi_x = -np.sqrt(2*alpha/num_features) * np.multiply(temp_multiple, W)
            gradient_function = np.dot(gradient_of_phi_x.T, sample_of_theta)
            if maximize:
                gradient_function = -gradient_function

            return gradient_function
        
        result = spo.minimize(function_to_optimize, initial_point, method= optimize_method, jac = gradient_to_help, bounds = bnds, options={'gtol': 10**(-50), 'maxiter':2000})
    
    return result