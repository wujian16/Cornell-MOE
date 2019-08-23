import functools
import warnings
import numpy as np
import numpy.linalg as npla
import sys, os
import time
from PES.compute_covariance import *
from PES.initial_sample import *
from PES.hyper_samples import *
from PES.utilities import *
from PES.sample_minimum import *
from PES.PES import *
from PES.compute_posterior import *
from PES.EP import *
from PES.global_optimization import *
from PES.target_function import *





#The function to run PES to minimize the target function.
#Parameters: @target_function: the obejective function we want to minimize
#            @x_minimum: the lower bounds for each dimension
#            @x_maximum: the upper bounds for each dimension
#            @dimension: the dimensions of the objective function
#            @number_of_hyperparameter_sets: the number of the samples of the hyperparameters of the kernel we want to draw. 
#                                            It is the M defined in the paper.
#            @number_of_burnin: number of burnins
#            @sampling_method: the method used to sample the posterior distribution of the hyperparameters. User can choose 
#                              'mcmc' or 'hmc'.
#            @number_of_initial_points: the number of samples we want to use as initial observations
#            @number_of_experiments: number of experiments we want to run. For each experiment, we use different randomizations 
#                                    for starting points.
#            @number_of_iterations: number of iterations we want to run for each experiment
#            @number_of_features: the number of features that we would like to use for feature mapping. It is the "m" in the paper.
#            @optimization_method: optimization method used when calling global_optimization function. User can choose any method 
#                                  specified in the scipy.optimize.minimize 
#            @seed: seed specified for randomization
def run_PES(target_function, x_minimum, x_maximum, dimension, number_of_hyperparameter_sets = 100, number_of_burnin = 50, \
            sampling_method = 'mcmc', number_of_initial_points = 3, number_of_experiments = 1, number_of_iterations = 60, \
            number_of_features = 1000, optimization_method = 'SLSQP', seed = None): 

    warnings.filterwarnings('ignore')
    check_result_file_exist()

    if seed is not None:
        np.random.seed(seed)

    #For Hartmann6
    x_min = x_minimum
    x_max = x_maximum
    target = target_function




    #For Branin-Hoo
    #x_min = np.asarray([0.0,0.0])
    #x_max = np.asarray([1.0,1.0])
    #target = Branin_Hoo

    d = dimension
    num_of_hyperSets_initial = number_of_hyperparameter_sets
    number_burn = number_of_burnin
    sample_method = sampling_method

    bnds = get_bounds(x_min, x_max)
    opt_method = 'L-BFGS-B'

    #We obtain three random samples

    num_initial_points = number_of_initial_points

    final_result = []


    for pp in range(number_of_experiments):
     

        write_header_to_files(pp)
        warnings.filterwarnings('ignore')
        Xsamples = initial_samples(x_min, x_max, num_initial_points)
        write_data_to_file("Xsamples.txt", Xsamples)


        #Guesses first stores the initilized guesses
        guesses = Xsamples
        write_data_to_file("guesses.txt", guesses)
        Ysamples = np.zeros((Xsamples.shape[0]))

        for i in range(Xsamples.shape[0]):
            Ysamples[i] = target(Xsamples[i,:])

        Ysamples = np.asarray([Ysamples])
        Ysamples = Ysamples.T
        print('Best so far in the initial data ' + str((min(Ysamples))[0]))
        write_data_to_file("Ysamples.txt", Ysamples)


        #We sample from the posterior distribution of the hyper-parameters
        with hide_prints():
            noise, l, sigma = sample_hypers(Xsamples, Ysamples, d, 0.3, num_of_hyperSets_initial, number_burn, sample_method, seed)

            

        #global_minimum = target(np.array([(5-np.pi)/15,12.275/15]))
        #global_minimum = target(np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]))


        valid_evaluation = 1
        log10_scale_vec = []


        for g in range(number_of_iterations):

            print('PES, ' + str(pp) + 'th job, ' + str(g) + 'th iteration')
            start_1 = time.time()
            num_of_hyperSets = num_of_hyperSets_initial
            Xsamples_count_before = len(Xsamples)
            Ysamples_count_before = len(Ysamples)
            guesses_count_before = len(guesses)
            initial_point = guesses[-1,:]

            num_of_features = number_of_features
            num_of_obser = len(Ysamples)
            x_minimum_vec = []
            K_vec = []
            K_star_min_vec = []
            K_plus_W_tilde_inverse_vec = []
            m_f_minimum_vec = []
            v_f_minimum_vec = []
            c_and_m_vec = []
            opt_method = 'L-BFGS-B'

            warnings.filterwarnings("error")
            valid_num_hyperSets = 0

            for j in range(num_of_hyperSets):
                opt_method = 'L-BFGS-B'
                try:
                    result = sample_min_with_randFeatures(num_of_features, d, Xsamples, Ysamples, sigma[j], l[j], noise[j], initial_point, opt_method, False, bnds)
                    x_minimum = result.x
                    x_minimum_vec.append(x_minimum)
                    if opt_method == 'L-BFGS-B':
                        hess_at_min_inverse = result.hess_inv.todense()
                    else:
                        hess_at_min_inverse = result.hess_inv
                    hess_at_min = compute_inverse(hess_at_min_inverse)  

                    value_of_nObservations = (Ysamples.T)[0]
                    K, K_star_min, K_plus_W_tilde_inverse, m_f_minimum, v_f_minimum, c_and_m = Expectation_Propagation(Xsamples, value_of_nObservations, num_of_obser, x_minimum, d, l[j,:], sigma[j], noise[j], hess_at_min)
                    K_vec.append(K)
                    K_star_min_vec.append(K_star_min)
                    K_plus_W_tilde_inverse_vec.append(K_plus_W_tilde_inverse)
                    m_f_minimum_vec.append(m_f_minimum)
                    v_f_minimum_vec.append(v_f_minimum)
                    c_and_m_vec.append(c_and_m)
                    valid_num_hyperSets = valid_num_hyperSets + 1
                except:
                    pass

            num_of_hyperSets = valid_num_hyperSets

            opt_method = optimization_method 

            warnings.filterwarnings("error")

            PES_fail = False

            try:

                PES = functools.partial(PES_aquisition_function_multi, Xsamples = Xsamples, x_minimum = x_minimum_vec, l_vec = l, \
                                        sigma = sigma, noise = noise, K = K_vec, K_star_min = K_star_min_vec, \
                                         K_plus_W_tilde_inverse = K_plus_W_tilde_inverse_vec, \
                                         m_f_minimum = m_f_minimum_vec, v_f_minimum = v_f_minimum_vec, c_and_m = c_and_m_vec, \
                                         num_of_hyperSets = num_of_hyperSets)

                ret = global_optimization(PES, d, x_min, x_max, gradient = None, gridsize = 500, stored_min_guesses = None, \
                                          using_grid = True, optimize_method = opt_method, maxiter = 2000, bnds = bnds)


                optimum = np.array(ret.x)
                optimum_value = np.array([target(optimum)])

            except:
                print('PES falied')
                PES_fail = True
                pass

            if PES_fail:
                warnings.filterwarnings('ignore')
                with hide_prints():
                    noise, l, sigma = sample_hypers(Xsamples, Ysamples, d, 0.3, num_of_hyperSets_initial, number_burn, sample_method, seed)

                print('return back due to PES fail')
                continue
            
            
            Xsamples = np.vstack((Xsamples, optimum))
            Ysamples = np.vstack((Ysamples, optimum_value))
            end_1 = time.time()
            print('PES takes ' + str(end_1 - start_1) + ' seconds')
            print('PES suggests: ')
            print(optimum)
            

            start_2 = time.time()
            #We sample from the posterior distribution of the hyper-parameters
            warnings.filterwarnings('ignore')
            num_of_hyperSets = num_of_hyperSets_initial
            try:
                with hide_prints():
                    noise, l, sigma = sample_hypers(Xsamples, Ysamples, d, 0.3, num_of_hyperSets_initial, number_burn, sample_method, seed)

            except:
                if len(Xsamples) > Xsamples_count_before:
                    Xsamples = Xsamples[:-1,:]
                if len(Ysamples) > Ysamples_count_before:
                    Ysamples = Ysamples[:-1]
                print('Sampling hyperparameters of posterior GP failed')
                continue

            end_2 = time.time()
            print('Retraining the model takes '+ str(end_2 - start_2) + ' seconds')
                
            write_data_to_file("Xsamples.txt", optimum)
            write_data_to_file("Ysamples.txt", optimum_value)


            start_3 = time.time()
            K_plus_I_inverse_vec = []
            num_of_obser = len(Xsamples)

            for w in range(num_of_hyperSets):
                K_plus_I_inverse = covNobeservations(Xsamples, num_of_obser, sigma[w], noise[w], l[w]) + sigma[w]*10**(-10)*np.eye((num_of_obser))
                K_plus_I_inverse_vec.append(np.array(K_plus_I_inverse))

            warnings.filterwarnings("error")
            try: 

                pos_mean_function = functools.partial(posterior_mean_given_nObservations, X_nObservations = Xsamples, value_of_nObservations = Ysamples, \
                                                K_plus_I_inverse = K_plus_I_inverse_vec, l = l, sigma = sigma, \
                                                num_of_hyperSets = num_of_hyperSets)


                pos_mean_grad_function = functools.partial(posterior_gradient_given_nObservations, X_nObservations = Xsamples, value_of_nObservations = Ysamples, \
                                                K_plus_I_inverse = K_plus_I_inverse_vec, l = l, sigma = sigma, \
                                                num_of_hyperSets = num_of_hyperSets, d = d)

                ret_pos = global_optimization(pos_mean_function, d, x_min, x_max, gradient = pos_mean_grad_function, gridsize = 500, \
                                                  stored_min_guesses = None, using_grid = True, optimize_method = opt_method, \
                                                  maxiter = 2000, bnds = bnds)
            except:
                if len(Xsamples) > Xsamples_count_before:
                    Xsamples = Xsamples[:-1,:]
                if len(Ysamples) > Ysamples_count_before:
                    Ysamples = Ysamples[:-1]
                print('Find the minimum of posterior mean failed')
                continue


            pos_optimum = np.array(ret_pos.x)
            write_data_to_file("guesses.txt", pos_optimum)
            current_value = target(pos_optimum)

            if current_value < (min(Ysamples))[0]:
                print('The recommended point ' + str(pos_optimum))
            else:
                current_value = (min(Ysamples))[0]
                print('The recommended point ' + str(Xsamples[np.argmin(Ysamples)]))
            
            end_3 = time.time()
            print('Recommending the point takes '+ str(end_3 - start_3) + ' seconds')
            print('Best so far ' + str(current_value))

            guesses = np.vstack((guesses, pos_optimum))

            


