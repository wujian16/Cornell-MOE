import numpy as np
from PES.utilities import *
from PES.compute_covariance import *




#This python file is the third(last) part of the Predictive Entropy Search function, which corresponds to the Appendix B.3 of
#the paper. 





#This function is the PES acquisition function marginalized with respect to posterior distribution over the hyperparameters of 
#the kernel. It corresponds to the equation (10) of the paper. This function calls the PES_aquisition_function, whih is the 
#unmarginalized version of the PES. 
#Parameters: @xPrime: the point we want to evaluate the acquisition function at 
#            @Xsamples: observations we have, which is the X_n
#            @x_minimum: the minimum we sampled from the approximate posterior distribution, which is the output of the 
#                        sample_min_with_randFeatures function
#            @l_vec: the lengthscale parameter of the squared exponential kernel
#            @sigma: the variance parameter of the squared exponential kernel
#            @noise: the white noise of the function, which is obtained from simulation
#            @K: the same matrix defined in the Appendix section of the paper, which is the kernel matrix containing the 
#                covariance evaluated on the stacked vector [z; c]. Here, it is one of the output of the Expectation_Propagation
#                function
#            @K_star_min: K_star_min is the cross-covariance column evaluated between f(x_min) and [z; c]
#            @K_plus_W_tilde_inverse: (K + W_tilde)^(-1), the same expression defined in the Appendix of the paper
#            @m_f_minimum: one of the two elements in the m_f vector defined in the paper. Note here, we compute the m_f vector 
#                          elementwise. it is one of the output of the Expectation_Propagation function.
#            @v_f_minimum: one of the elements in the V_f vector defined in the paper, which corresponds to the variance of the 
#                          posterior distribution of f(x_min). It is one of the output of the Expectation_Propagation function.
#            @c_and_m: the stack vector of c and m_tilde, [c; m_tilde]. It is one of the output of the Expectation_Propagation function.
#            @num_of_hyperSets: the number of the samples of the hyperparameters of the squared exponential kernel. It is the M defined
#            					in the paper. 
def PES_aquisition_function_multi(xPrime, Xsamples, x_minimum, l_vec, sigma, noise, K, K_star_min,K_plus_W_tilde_inverse, m_f_minimum, v_f_minimum, c_and_m, num_of_hyperSets):
    
    objective = 0
    count = 0
    
    for i in range(num_of_hyperSets):
        try: 
            new_objective, scalar_count = PES_aquisition_function(xPrime, Xsamples, x_minimum[i], l_vec[i], sigma[i], noise[i], \
                                                K[i], K_star_min[i], K_plus_W_tilde_inverse[i], m_f_minimum[i], \
                                                v_f_minimum[i], c_and_m[i])
            if scalar_count < 10**(5):
                objective = objective + new_objective
                count = count + 1
        except:
            pass

    if count == 0:
        objective = 10**(200)
    else:
        objective = objective/count
    
    return objective





#This function is the unmarginalized version of the PES.
#Parameters: @xPrime: the point we want to evaluate the acquisition function at 
#            @Xsamples: observations we have, which is the X_n
#            @x_minimum: the minimum we sampled from the approximate posterior distribution, which is the output of the 
#                        sample_min_with_randFeatures function
#            @l_vec: the lengthscale parameter of the squared exponential kernel
#            @sigma: the variance parameter of the squared exponential kernel
#            @noise: the white noise of the function, which is obtained from simulation
#            @K: the same matrix defined in the Appendix section of the paper, which is the kernel matrix containing the 
#                covariance evaluated on the stacked vector [z; c]. Here, it is one of the output of the Expectation_Propagation
#                function
#            @K_star_min: K_star_min is the cross-covariance column evaluated between f(x_min) and [z; c]
#            @K_plus_W_tilde_inverse: (K + W_tilde)^(-1), the same expression defined in the Appendix of the paper
#            @m_f_minimum: one of the two elements in the m_f vector defined in the paper. Note here, we compute the m_f vector 
#                          elementwise. it is one of the output of the Expectation_Propagation function.
#            @v_f_minimum: one of the elements in the V_f vector defined in the paper, which corresponds to the variance of the 
#                          posterior distribution of f(x_min). It is one of the output of the Expectation_Propagation function.
#            @c_and_m: the stack vector of c and m_tilde, [c; m_tilde]. It is one of the output of the Expectation_Propagation function.
def PES_aquisition_function(xPrime, Xsamples, x_minimum, l_vec, sigma, noise, K, K_star_min,K_plus_W_tilde_inverse, m_f_minimum, v_f_minimum, c_and_m):
    
    num_of_obser = len(Xsamples)
    xPrime = np.array([xPrime])
    #K_star is the cross-covariance column evaluated between f(x) and [c;z], its dimension is 1x(n+d+d*(d-1)/2 +d+1)
    #with f(x_min) being the last element
    K_star = compute_cov_xPrime_cz(xPrime, Xsamples, x_minimum, num_of_obser, sigma, noise, l_vec)

    
    #m_f_evaluated is the other element of the m_f vector
    m_f_evaluated = np.dot(np.dot(K_star, K_plus_W_tilde_inverse), c_and_m)
    #v_f_evaluated is one of the element of the V_f vector, which corresponds to the variance of the 
    #posterior distribution of f(x)
    v_f_evaluated = sigma - np.dot(np.dot(K_star, K_plus_W_tilde_inverse), K_star.T)


    # Compute the covariance between f(x) and f(x_min), its dimension is 1x1
    cov_evaluated_minimum = K_star[-1]

    #v_f_evaluated_minimum is one of the element of the V_f vector, which corresponds to the posterior covariance between the 
    #f(x) and f(x_min)
    v_f_evaluated_minimum = cov_evaluated_minimum - np.dot(np.dot(K_star, K_plus_W_tilde_inverse), K_star_min.T)
    

    #This prevents v from being close to zero. 
    scalar = 1 - 10**(-4)
    v_too_small = True

    scalar_count = 0
    
    if (v_f_evaluated + v_f_minimum) < 10**(-10):
        v = v_f_evaluated + v_f_minimum
        
    else:
        while (v_too_small) and (scalar_count < (10**(9) + 5)):
            v_too_small = False
            scaled_v = v_f_evaluated - 2*scalar*v_f_evaluated_minimum + v_f_minimum
            if scaled_v < 10**(-10):
                scalar = scalar**2
                v_too_small = True
            scalar_count = scalar_count + 1

        v = v_f_evaluated - 2*scalar*v_f_evaluated_minimum + v_f_minimum

    alpha = np.divide(m_f_evaluated - m_f_minimum, np.sqrt(v))

    beta = (1/np.sqrt(2*np.pi))*np.exp(-0.5*alpha**2 - log_Phi(alpha))


    Vf_ccT_Vf = np.multiply(v_f_evaluated - v_f_evaluated_minimum, v_f_evaluated - v_f_evaluated_minimum)
    beta_over_v = np.divide(beta, v)

    v_n_x_xmin = v_f_evaluated - np.multiply(np.multiply(beta_over_v, alpha + beta), Vf_ccT_Vf) + noise      
    
    
    
    #covariance matrix between n observations and n observations, dimension should be nxn
    K_n = K[:len(Xsamples), :len(Xsamples)]

    if len(xPrime) > 0:
        #vector of covariance between x_to_be_evaluated and n observations, dimension should be 1xn
        k_n_x = K_star[:len(Xsamples)]
        v_n_x = noise + sigma*(1 + 10**(-10)) - np.dot(np.dot(k_n_x, compute_inverse(K_n + noise*np.eye(len(Xsamples)))), k_n_x.T)
    else:
        v_n_x = noise + sigma*(1 + 10**(-10))



    objective = 0.5*np.log(2*np.pi*np.exp(1)*(v_n_x + noise)) - 0.5*np.log(2*np.pi*np.exp(1)*(v_n_x_xmin + noise))
    
    #since we use the scipy minimize later, here we put a negtive sign at the beginning
    objective = -objective
    
    return objective, scalar_count

    
    