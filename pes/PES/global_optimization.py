import numpy as np
import numpy.random as npr
import scipy.optimize as spo



#Function to find the global minimum of the input function. Here we use the grid search. We first divide the domain into grids and then 
#evaluate the input function at the grid points. Furthermore, we use the grid point with minimum input function value as our starting 
#point to put into the optimization function.
#Parameters: @function_to_optimize: the input function to optimize
#            @d: the dimensions of the objective function
#            @x_min: the lower bounds for each dimension
#            @x_max: the upper bounds for each dimension
#            @gradient: the gradient of the input function
#            @gridsize: the size of the grid
#            @stored_min_guesses: guesses of the global minimums. If the minimum value of the guesses is smaller than the values of grid 
#                                 points, that guess point would be used as the initial point for optimization.

#            @using_grid: whether to use grid search or not. If false, the function will use the minimum of the guesses as initial point.
#            @optimize_method: method used to optimize the approximate posterior distribution. User can 
#                              choose any method specified in the scipy.optimize.minimize     
#            @maxiter: maximum iterations of scipy.optimize.minimize     
#            @bnds: the bounds of the domain
def global_optimization(function_to_optimize, d,x_min, x_max, gradient = None, gridsize = 500, stored_min_guesses = None, using_grid = True, optimize_method = 'L-BFGS-B', maxiter = 1000, bnds = None):
    
    
    if (using_grid is False) and (stored_min_guesses is None):
        print('For initialization, grid or guessed starting point must be used')
    
    grid = []
    
    #Generate the grid
    if using_grid:
        
        grid_size = gridsize
        x_min = np.asarray(x_min)
        x_max = np.asarray(x_max)
        if d == 1:
            grid = np.array([[x_min], ]*grid_size) +  np.multiply(np.array([[x_max - x_min], ]*grid_size), npr.uniform(size = (grid_size,d)))
        else: 
            grid = np.array([x_min, ]*grid_size) +  np.multiply(np.array([x_max - x_min, ]*grid_size), npr.uniform(size = (grid_size,d)))
        
        if stored_min_guesses is not None:
            stored_min_guesses = np.asarray(stored_min_guesses)
            grid = np.vstack((grid, stored_min_guesses))       
    else:
        stored_min_guesses = [stored_min_guesses]
        grid = np.asarray(stored_min_guesses)
        
    
    grid_function_values = []
    

    #Evaluate input function at grid points
    for i in range(len(grid)):
        if d == 1:
            try:
                grid_function_values.append(function_to_optimize(grid[i,0]))
            except:
                grid_function_values.append(10**(200))
                print('grid value error')
        else:
            try:
                grid_function_values.append(function_to_optimize(grid[i,:]))
            except:
                grid_function_values.append(10**(200))
                print('grid value error')
        
        
    grid_function_values = np.asarray(grid_function_values)
    
    #Find the point with minimal function value
    min_index = np.argmin(grid_function_values)
        
    initial_point = grid[min_index, :]
        
        
    
    if optimize_method not in ('CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact', 'trust-constr'):
        result = spo.minimize(function_to_optimize, initial_point, method= optimize_method, options={'maxiter':maxiter, 'disp': False}, bounds = bnds)
    else:
        if gradient is None:
            result = spo.minimize(function_to_optimize, initial_point, method = 'SLSQP', bounds = bnds, options={'maxiter':maxiter, 'disp': False})
        else:
            result = spo.minimize(function_to_optimize, initial_point, method = optimize_method, jac = gradient, options={'maxiter':maxiter, 'disp': False}, bounds = bnds)
    
    return result