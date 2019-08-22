import numpy as np
import numpy.random as npr



#This file is used to store the function that the user would like to oprimize.
#Users can define their own functions in this file. Here we define two synthetic
#funtions. One is Hartmann6 and the other one is Branin Hoo. Users can define 
#their own functions in the similar way. 




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






#The Hartmann6 function. The global minimums are at (0.12389382, 0.81833333) and 
#(0.961652, 0.165). The minimum value is -1.047394 without noise. Here we add
#a standard gaussian noise with 10^(-3) scale to the function. The input bounds
#are 0 <= xi <= 1, i = 1,2. 
def Branin_Hoo(X):
    x1 = X[0]
    x2 = X[1]
    
    x1bar = 15*x1 - 5
    x2bar = 15 * x2   
    
    term1 = x2bar - 5.1*x1bar**2/(4*np.pi**2) + 5*x1bar/np.pi - 6
    term2 = (10 - 10/(8*np.pi)) * np.cos(x1bar)
    
    ret = (term1**2 + term2 - 44.81) / 51.95 + 10**(-3) * npr.normal(0,1)
    return ret