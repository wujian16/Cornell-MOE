import numpy as np
from PES.compute_covariance import *






#Function to compute the mean of the posterior distribution of f(x) given past observations and new sampled point, marginalized with 
#respect to the posterior distribution over the hyperparameters of the kernel. Here we do not include the noise. Therefore it is not y_n.
#Parameters: @xPrime: the input vector
#            @X_nObservations: observations we have. Here it is the set of the past observations X_n plus the new sampled point.
#            @value_of_nObservations: the function values of X_nObservations
#            @K_plus_I_inverse: (K_n + I)^(-1)
#            @l: the lengthscale parameter of the squared exponential kernel
#            @sigma: the variance parameter of the squared exponential kernel
#            @num_of_hyperSets: the number of the samples of the hyperparameters of the squared exponential kernel. It is the M defined
#                               in the paper. 
def posterior_mean_given_nObservations(xPrime, X_nObservations, value_of_nObservations, K_plus_I_inverse, l, sigma, num_of_hyperSets):
    f = 0
    xPrime = np.array([xPrime])
    for i in range(num_of_hyperSets):
        #the cross-variance between x and n observations, dimension should be 1xn
        k_n_x = cov_xPrime_nObservations(xPrime, X_nObservations, sigma[i], l[i])
        #value_of_nObservations should have dimension nx1, the dimension of f should be 1x1
        temp_mean = np.dot(np.dot(k_n_x, K_plus_I_inverse[i]), value_of_nObservations)
        f = f + temp_mean[0,0]
        
    f = f/num_of_hyperSets
    
    return f





#Function to compute the gradient of mean of the posterior distribution of f(x) given past observations and new sampled point, marginalized 
#with respect to the posterior distribution over the hyperparameters of the kernel. 
#Parameters: @xPrime: the input vector
#            @X_nObservations: observations we have. Here it is the set of the past observations X_n plus the new sampled point.
#            @value_of_nObservations: the function values of X_nObservations
#            @K_plus_I_inverse: (K_n + I)^(-1)
#            @l: the lengthscale parameter of the squared exponential kernel
#            @sigma: the variance parameter of the squared exponential kernel
#            @num_of_hyperSets: the number of the samples of the hyperparameters of the squared exponential kernel. It is the M defined
#                               in the paper. 
#            @the dimensions of the objective function
def posterior_gradient_given_nObservations(xPrime, X_nObservations, value_of_nObservations, K_plus_I_inverse, l, sigma, num_of_hyperSets, d):
    
    gradient = np.zeros((d))
    num_of_obser = len(X_nObservations)
    
    for i in range(num_of_hyperSets):
        #the cross-variance between x and n observations, dimension should be 1xn
        k_n_x = cov_xPrime_nObservations(np.array([xPrime]), X_nObservations, sigma[i], l[i])
        k_n_x = k_n_x[0]
        #n copies of l, the dimension should be nxd, the dimension of l should be 1xd
        n_l = np.array([l[i,:], ]*num_of_obser)
        #n copies of x, the dimension should be nxd, the dimension of x should be 1xd
        n_x = np.array([xPrime, ]*num_of_obser)
        #d copies of cross-variance between x and n observations, dimension should be nxd
        k_n_x_d_copies = np.array([k_n_x, ]*d).T
        #the derivative of the cross-variance between x and n observations, dimension should be nxd
        dx_k_n_x = -np.multiply(np.multiply(n_l, n_x - X_nObservations), k_n_x_d_copies)
        
        #dimension should be dx1
        new_grad = np.dot(np.dot(dx_k_n_x.T, K_plus_I_inverse[i]), value_of_nObservations)
        new_grad = new_grad.T
        new_grad = new_grad[0]
        gradient = gradient + new_grad
    
    gradient = gradient/num_of_hyperSets
    
    return gradient

