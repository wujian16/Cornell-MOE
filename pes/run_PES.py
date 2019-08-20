import os
import numpy as np
import numpy.random as npr
from PES.main import *


#User can define target function here
##########################################################
##########################################################


#The Hartmann6 function. The only global minimum is at (0.20169, 0.150011, 0.476874, 
# 0.275332, 0.311652, 0.6573). The minimum value is -3.32237 without noise. Here we 
# add a 10^(-3) noise to the function. The input bounds are 0 <= xi <= 1, i = 1..6. 
def Hartmann6(x):
    
    alpha = [1.00, 1.20, 3.00, 3.20]
    A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                          [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                          [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                          [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
    P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                            [2329, 4135, 8307, 3736, 1004, 9991],
                            [2348, 1451, 3522, 2883, 3047, 6650],
                            [4047, 8828, 8732, 5743, 1091, 381]])



    external_sum = 0
    for i in range(4):
        internal_sum = 0
        for j in range(6):
            internal_sum = internal_sum + A[i, j] * (x[j] - P[i, j]) ** 2
        external_sum = external_sum + alpha[i] * np.exp(-internal_sum)
    external_sum = -external_sum + 10**(-3)
    return external_sum


##########################################################
##########################################################






#Specify the parameters for running PES
target_function = Hartmann6
x_minimum = np.asarray([0.0,0.0,0.0,0.0,0.0,0.0])
x_maximum = np.asarray([1.0,1.0,1.0,1.0,1.0,1.0])
dimension = 6






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
run_PES(target_function, x_minimum, x_maximum, dimension, number_of_hyperparameter_sets = 100, number_of_burnin = 50, \
        sampling_method = 'mcmc', number_of_initial_points = 3, number_of_experiments = 1, number_of_iterations = 60, \
        number_of_features = 1000, optimization_method = 'SLSQP')


