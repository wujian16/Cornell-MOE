import numpy as np
import numpy.random as npr



#Function to create initial observations. This is the same as the lhsu function of the author's code.
#For the original code, please visit https://bitbucket.org/jmh233/codepesnips2014/src/master/sourceFiles/.
#Parameters: @xmin: the lower bounds for each dimension    
#            @xmax: the upper bounds for each dimension
#            @nsample: the number of samples we want to use as initial observations
def initial_samples(xmin,xmax,nsample):
    n_dimension=len(xmin)
    rv=npr.uniform(0,1, size = (nsample,n_dimension))
    result= np.zeros((nsample,n_dimension))
    for j in range(n_dimension):
        idx = npr.permutation(nsample)+1
        P =(idx.T - rv[:,j])/nsample
        result[:,j] = xmin[j] + np.multiply(P, (xmax[j]-xmin[j]))
    return result
