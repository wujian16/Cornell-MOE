import GPy
import numpy as np


#This function includes functions to draw samples from the posterior distribution of the hyperparameters. Here we use Gpy packakge 
#developed by the by the Sheffield machine learning group. For the package info, please visit https://sheffieldml.github.io/GPy/. 



#Function to sample the posterior distribution of the hyperparameters using mcmc. 
#Parameters: @X_nObservations: observations we have, which is the X_n
#            @value_of_observations: the function values of the observations
#            @d: the dimensions of the objective function
#            @initial_value_of_l: the initial value of l to begin with simulations
#            @num_of_set_hyper: the number of the samples of the hyperparameters of the kernel we want to draw. It is the M defined
#                               in the paper. 
#            @number_burn: number of burnins
def sample_hypers_mcmc(X_nObservations, value_of_observations, d, initial_value_of_l = None, num_of_set_hyper = 100, number_burn = 50, seed = None):

    if seed is not None:
        np.random.seed(seed)
    
    if initial_value_of_l is None:
        lengthscale = 0.3 * np.ones((d))
    else:
        lengthscale = initial_value_of_l * np.ones((d))


    k = GPy.kern.RBF(input_dim = d, lengthscale = lengthscale, ARD = True)
    m = GPy.models.GPRegression(X_nObservations, value_of_observations, k)

    # Give some general prior distributions for model parameters
    m.kern.lengthscale.set_prior(GPy.priors.Gamma(2.5, 5.0))
    m.kern.variance.set_prior(GPy.priors.Gamma(1.0,1.0))
    m.likelihood.variance.set_prior(GPy.priors.Gamma(1.0,1.0))   
    

    m_noise_M = []
    m_kern_lengthscale_M = []
    m_kern_variance_M = []
    

    for i in range(num_of_set_hyper):
        mcmc = GPy.inference.mcmc.Metropolis_Hastings(m)
        num_hyper = num_of_set_hyper
        burn = number_burn
        total = (burn + num_hyper)*3
        mcmc.sample(Ntotal= total, Nburn=burn*3)
        
        m_noise_M.append(mcmc.model.Gaussian_noise.variance[0])
        lengthscale_vec = []
        for j in range(d):
            lengthscale_vec.append(mcmc.model.kern.lengthscale[j])
        m_kern_lengthscale_M.append(np.array(lengthscale_vec))
        m_kern_variance_M.append(mcmc.model.kern.variance[0])
    
    m_noise_M = np.array(m_noise_M)
    m_kern_lengthscale_M = np.array(m_kern_lengthscale_M)
    m_kern_variance_M = np.array(m_kern_variance_M)
    
    
    return m_noise_M, m_kern_lengthscale_M, m_kern_variance_M





#Function to sample the posterior distribution of the hyperparameters using hybrid monte carlo. 
#Parameters: @X_nObservations: observations we have, which is the X_n
#            @value_of_observations: the function values of the observations
#            @d: the dimensions of the objective function
#            @initial_value_of_l: the initial value of l to begin with simulations
#            @num_of_set_hyper: the number of the samples of the hyperparameters of the kernel we want to draw. It is the M defined
#                               in the paper. 
#            @number_burn: number of burnins
def sample_hypers_hmc(X_nObservations, value_of_observations, d, initial_value_of_l = None, num_of_set_hyper = 100, number_burn = 50, seed = None):


    if seed is not None:
        np.random.seed(seed)

        
    if initial_value_of_l is None:
        lengthscale = 0.3 * np.ones((d))
    else:
        lengthscale = initial_value_of_l * np.ones((d))


    k = GPy.kern.RBF(input_dim = d, lengthscale = lengthscale, ARD = True)
    m = GPy.models.GPRegression(X_nObservations, value_of_observations, k)

    # Give some general prior distributions for model parameters
    m.kern.lengthscale.set_prior(GPy.priors.Gamma(2.5, 5.0))
    m.kern.variance.set_prior(GPy.priors.Gamma(1.0,1.0))
    m.likelihood.variance.set_prior(GPy.priors.Gamma(1.0,1.0))


    
    hmc = GPy.inference.mcmc.HMC(m,stepsize= 5e-2)
    s = hmc.sample(num_samples = number_burn*3) 
    s = hmc.sample(num_samples = num_of_set_hyper*3)
    
    
    samples = s[number_burn*3:] 
    
    #Store num_of_set_hyper number of hyper parameters
    m_kern_variance_M = samples[-num_of_set_hyper: ,0]
    m_kern_lengthscale_M = samples[-num_of_set_hyper: ,1:-1]
    m_kern_noise_M = samples[-num_of_set_hyper: ,-1]
    
    
    
    return m_kern_noise_M, m_kern_lengthscale_M, m_kern_variance_M


#The wrapper of the two sampling function.
#Parameters: @X_nObservations: observations we have, which is the X_n
#            @value_of_observations: the function values of the observations
#            @d: the dimensions of the objective function
#            @initial_value_of_l: the initial value of l to begin with simulations
#            @num_of_set_hyper: the number of the samples of the hyperparameters of the kernel we want to draw. It is the M defined
#                               in the paper. 
#            @number_burn: number of burnins
#            @sample_method: the method used to sample the posterior distribution of the hyperparameters. User can choose 'mcmc' or 'hmc'.
def sample_hypers(X_nObservations, value_of_observations, d, initial_value_of_l = None, num_of_set_hyper = 100, number_burn = 50, sample_method = 'mcmc', seed = None):
    if sample_method is 'mcmc':
        noise, l, sigma = sample_hypers_mcmc(X_nObservations, value_of_observations, d, initial_value_of_l, num_of_set_hyper, number_burn)
        return noise, l, sigma
    elif sample_method is 'hmc':
        print('enter hmc')
        noise, l, sigma = sample_hypers_hmc(X_nObservations, value_of_observations, d, initial_value_of_l, num_of_set_hyper, number_burn)
        return noise, l, sigma
    else:
        print('Sample method is not available. Please choose between mcmc and hmc.')
        return None