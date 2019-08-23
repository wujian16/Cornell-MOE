import numpy as np
import scipy.linalg as spla
from PES.utilities import *
from PES.compute_covariance import *



#Function to implement the expectation propagation algorithm, which is the second part of the PES. This function corresponds to the 
#B.1 and B.2 of the Appendix of the paper.
#Parameters: @Xsamples: observations we have, which is the X_n
#            @value_of_nObservations: the function values of the observations
#            @num_of_obser: the number of the observations we have
#            @x_minimum: the minimum we sampled from the approximate posterior distribution, which is the output of the 
#                        sample_min_with_randFeatures function
#            @d: the dimensions of the objective function
#            @l_vec: the lengthscale parameter of the squared exponential kernel
#            @sigma: the variance parameter of the squared exponential kernel
#            @noise: the white noise of the function, which is obtained from simulation
#            @hess_at_min: the approximation of the hessian values at the x_minimum. Note here, in the paper, we are required to use 
#                          the actual hessians at the x_minimum. However, considering the computational cost, we use approximation 
#                          of the hessians here. The author uses the same idea in his original code. For the original code, please visit 
#                          https://bitbucket.org/jmh233/codepesnips2014/src/master/sourceFiles/. 
def Expectation_Propagation(Xsamples, value_of_nObservations, num_of_obser, x_minimum, d, l_vec, sigma, noise, hess_at_min):
    

    #K_z, dimension is (d+1)x(d+1)
    K_z = compute_K_z(x_minimum, sigma, l_vec, noise, d)

    #K_c, dimension is (n+d+d*(d-1)/2)x(n+d+d*(d-1)/2)
    K_c = compute_K_c(Xsamples, x_minimum, num_of_obser, sigma, noise, l_vec)

    #K_zc, dimension is (d+1)x(n+d+d*(d-1)/2)
    K_cz = compute_K_cz(Xsamples, x_minimum, num_of_obser, sigma, noise, l_vec)
    K_zc = K_cz.T

    #Covariance matrix between z and c, the dimension is ( (n+d+d*(d-1)/2) + (d+1))^2
    K = compute_K(K_z, K_c, K_cz)

    n = num_of_obser


    zero_gradient = np.zeros((d))

    off_dia_hess  =  get_off_diagonal_element(hess_at_min) 
    
    #c, dimension is 1x(n+d+d*(d-1)/2)
    c = np.concatenate((value_of_nObservations, zero_gradient, off_dia_hess))


    K_c_inverse = compute_inverse(K_c)


    #m_0 is a (d+1)x1 matrix
    m_0 = np.dot(np.dot(K_zc, K_c_inverse), c.T)

    #v_0 is a (d+1)x(d+1) matrix
    v_0 = K_z - np.dot(np.dot(K_zc, K_c_inverse), K_cz)

    min_of_nObservations = np.amin(value_of_nObservations)

    v_0_inverse = compute_inverse(v_0)


    #m is a (d+1)x1 matrix, v is a (d+1)x1 matrix
    m = m_0

    v_inverse = np.reciprocal(np.diag(v_0))

    #m_tilde and v_tilde are both (d+1)x1 matrix
    m_tilde = np.zeros((d+1))
    v_tilde_inverse = np.zeros((d+1))

    damping = 1
    convergence = False
    count = 0
    
    
    
    while ((not convergence)):
        
        #the m vector in the paper before EP
        m_old = m.astype(float) 

        #the V^(-1) vector before EP. We use inverse to avoid dividing by zero in the later computations. 
        v_old_inverse = v_inverse.astype(float)

        #the m_tilde vector before EP
        m_tilde_old = m_tilde.astype(float)
        #the V_tilde^(-1) vector before EP
        v_tilde_old_inverse = v_tilde_inverse.astype(float)

        #the v_bar vector before EP
        v_bar_old = np.reciprocal(v_old_inverse - v_tilde_old_inverse)
        #the m_bar vector before EP
        m_bar_old = np.multiply(v_bar_old, np.multiply(m_old, v_old_inverse) - np.multiply(m_tilde_old, v_tilde_old_inverse))


        #Note since we are doing the minimization instead of maximization(in the paper), we should be careful for the sign 
        #of each quantity while conducting EP. 

        #First we do EP for the factors corresponding to the constraints on the diagonal Hessian
        m_bar_old_dia_hess = m_bar_old[:d]
        v_bar_old_dia_hess = v_bar_old[:d]
        
     
        alpha = np.divide(m_bar_old_dia_hess, np.sqrt(v_bar_old_dia_hess))

        phi_alpha_over_Phi_alpha = (1/np.sqrt(2*np.pi))*np.exp(-0.5*alpha**2 - log_Phi(alpha))

        beta = np.divide(np.multiply(phi_alpha_over_Phi_alpha, phi_alpha_over_Phi_alpha + alpha), v_bar_old_dia_hess)
        k = np.divide(phi_alpha_over_Phi_alpha + alpha, np.sqrt(v_bar_old_dia_hess))

        #Update m_tilde and V_tilde^(-1) vectors 
        m_tilde_new_d = m_bar_old_dia_hess + np.reciprocal(k)
        v_tilde_new_d_inverse = np.divide(beta, np.ones((len(beta))) - np.multiply(beta, v_bar_old_dia_hess))


        #EP for the soft maximum constraint
        m_bar_old_max_cons = min_of_nObservations - m_bar_old[-1]
        v_bar_old_max_cons = v_bar_old[-1] + noise


        alpha = np.divide(m_bar_old_max_cons, np.sqrt(v_bar_old_max_cons))

        phi_alpha_over_Phi_alpha = (1/np.sqrt(2*np.pi))*np.exp(-0.5*alpha**2 - log_Phi(alpha))

        beta = np.divide(np.multiply(phi_alpha_over_Phi_alpha, phi_alpha_over_Phi_alpha + alpha), v_bar_old_max_cons)
        k = np.divide(phi_alpha_over_Phi_alpha + alpha, np.sqrt(v_bar_old_max_cons))
        k = -k

        #Update m_tilde and V_tilde^(-1) vectors 
        m_tilde_new_last = m_bar_old_max_cons + np.reciprocal(k)
        v_tilde_new_last_inverse = np.divide(beta, 1.0 - np.multiply(beta, v_bar_old_max_cons))

        #Put the updated vectors together
        m_tilde_new = np.concatenate((m_tilde_new_d, np.array([m_tilde_new_last])))
        v_tilde_new_inverse = np.concatenate((v_tilde_new_d_inverse, np.array([v_tilde_new_last_inverse])))

        #For computational stability and edge cases, we use the same methods as the author used in the original code. For
        #original code, please visit https://bitbucket.org/jmh233/codepesnips2014/src/master/sourceFiles/.
        v_tilde_new_inverse[abs(v_tilde_new_inverse) < 10**(-300)] = 10**(-300)
        m_tilde_new[v_old_inverse < 0] = m_tilde_old[v_old_inverse < 0]
        v_tilde_new_inverse[v_old_inverse < 0] = v_tilde_old_inverse[v_old_inverse < 0]
        
        
        
        EP_sucess = False
        EP_count = 0
        
        #Here we do damping while updating the m_tilde and V_tilde^(-1) vectors. This is not mentioned in the paer, 
        #but it is in the author's code. Here, we use the same idea. For the original code, please visit 
        #https://bitbucket.org/jmh233/codepesnips2014/src/master/sourceFiles/.
        while ((not EP_sucess) and (EP_count < 20)):

            m_tilde_new_temp = m_tilde_new * damping + m_tilde_old * (1 - damping)
            v_tilde_new_inverse_temp = v_tilde_new_inverse * damping + v_tilde_old_inverse * (1 - damping)
            
            w, eig_vec = spla.eig(np.diag(v_tilde_new_inverse_temp) + v_0_inverse)
            eigenvalues = w.real
            

            for i in range(len(eigenvalues)):
                if (1/eigenvalues[i]) <= 10**(-10):
                    EP_sucess = True

            if not EP_sucess:
                damping = damping*0.5
            EP_count = EP_count + 1

        #Update m_tilde and V_tilde^(-1) vectors
        m_tilde = m_tilde_new_temp
        v_tilde_inverse = v_tilde_new_inverse_temp

        #Update m and V^(-1) vectors
        v_new = compute_inverse(np.diag(v_tilde_inverse) + v_0_inverse)        
        m = np.dot(v_new, np.dot(np.diag(v_tilde_inverse), m_tilde) + np.dot(v_0_inverse, m_0))
        v_inverse = np.reciprocal(get_diagonal_element(v_new))


        mean_difference = np.abs(m - m_old)
        var_difference = np.abs(np.reciprocal(v_inverse) - np.reciprocal(v_old_inverse))
        max_difference =  max(np.amax(mean_difference), np.amax(var_difference)) 
        if  max_difference < 10**(-20):
            convergence = True
 



        damping = damping*0.99

        count = count + 1
        

    v_tilde = np.reciprocal(v_tilde_inverse)
    #[K + W_tilde], dimension is ((d+1) + (n+d+d*(d-1)/2))^2
    K_plus_W_tilde_inverse = compute_inverse(np.diag(np.concatenate((np.zeros(int(n+d+d*(d-1)/2)), v_tilde))) + K)

    #[c; m_tilde], dimension is ((d+1) + (n+d+d*(d-1)/2))x1, mAux
    c_and_m = np.array(np.concatenate((c, m_tilde)))

    x_minimum_vec = np.array([x_minimum])

    #K_star_min is the cross-covariance column evaluated between f(x_min) and [z; c], its dimension is 1x((n+d+d*(d-1)/2)+(d+1))
    K_star_min = compute_cov_xPrime_cz(x_minimum_vec, Xsamples, x_minimum, num_of_obser, sigma, noise, l_vec)
    




    #m_f_minimum, dimension 1x1, one of the two elements in the m_f vector defined in the paper. Note here, we compute the m_f vector 
    #elementwise.
    m_f_minimum = np.dot(np.dot(K_star_min, K_plus_W_tilde_inverse), c_and_m)

    #v_f_minimum, dimension 1x1, one of the elements in the V_f vector defined in the paper, which corresponds to the variance of the 
    #posterior distribution of f(x_min). Since cov(f(x_min), f(x_min)) is the 0 distance squared exponential, so it equals to sigma.
    v_f_minimum = sigma - np.dot(np.dot(K_star_min, K_plus_W_tilde_inverse), K_star_min.T)

    return K, K_star_min, K_plus_W_tilde_inverse, m_f_minimum, v_f_minimum, c_and_m