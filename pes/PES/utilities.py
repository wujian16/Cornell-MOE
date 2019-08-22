import numpy as np
import scipy.linalg as spla
import scipy.stats as sps
import os
import sys



#Function to compute the log_cdf(x)
#Parameters: @x
def log_Phi(x):
    if isinstance(x, np.ndarray):
        result = []
        for value in x:
            if value > 5:
                result.append(-sps.norm.sf(value))
            else:
                result.append(sps.norm.logcdf(value))
        result = np.array(result)
    else:
        if x > 5:
            result = -sps.norm.sf(x)
        else:
            result = sps.norm.logcdf(x)
    return result



    
   

#Function to compute the inverse of matrix A
#Parameters: @A: the input matrix
def compute_inverse(A):
    try:
        A_cholesky = spla.cho_factor(A)
        A_inverse = spla.cho_solve(A_cholesky, np.eye(A.shape[0]))
        return A_inverse
    except:
        A_inverse = spla.inv(A)
        return A_inverse



#Fun to get the bounds of the domain 
#Parameters: @x_min: the lower bounds for each dimension
#            @x_max: the upper bounds for each dimension
def get_bounds(x_min, x_max):
    bnds = np.vstack((x_min, x_max))
    bnds = bnds.T

    result = []
    for item in bnds:
        result.append(item)
        
    return result



#Function to get the off-diagonal elements of a matrix and compress them into a 1-dimensional array
#Parameters: @A: the input matrix
def get_off_diagonal_element(A):
    row_length = A.shape[0]
    column_length = A.shape[1]
    
    non_diag_element = []
    for i in range(row_length):
        for j in range(i+1,row_length):
            non_diag_element.append(A[i,j])
    non_diag_element = np.array(non_diag_element)        
    return non_diag_element
            

#Function to get the diagonal elements of a matrix
#Parameters: @A: the input matrix
def get_diagonal_element(A):
    return A.diagonal()



#Function to check whether current directory have files to store the outputs
#If not, create such files 
#Parameters: None
def check_result_file_exist():
    os.chdir(os.getcwd())
    
    Xfile_exists = os.path.isfile('Xsamples.txt')
    if not Xfile_exists:
        file = open("Xsamples.txt", "w") 
        file.close() 
        
    Yfile_exists = os.path.isfile('Ysamples.txt')
    if not Yfile_exists:
        file = open("Ysamples.txt", "w") 
        file.close() 
        
    guessFile_exists = os.path.isfile('guesses.txt')
    if not guessFile_exists:
        file = open("guesses.txt", "w") 
        file.close() 


#Fucntion to write how many experiments have been run in the files
#Parameters: None
def write_header_to_files(num):
	file = open("Xsamples.txt","a") 
	file.write('Experiment '+ str(num) + " begin" + "\n")
	file.close() 

	file = open("Ysamples.txt","a") 
	file.write('Experiment '+ str(num) + " begin" + "\n")
	file.close() 	

	file = open("guesses.txt","a") 
	file.write('Experiment '+ str(num) + " begin" + "\n")
	file.close() 



#Function to write output to files
#Parameters: @doc: the file we write outputs to
#            @data: the outputs we want to store
def write_data_to_file(doc, data):
	if data.ndim == 1:
		file = open(doc,"a") 
		file.write(",".join(str(x) for x in data) + "\n")
		file.close() 

	else:		
		file = open(doc,"a") 
		for item in data:
			file.write(",".join(str(x) for x in item) + "\n")
		file.close() 



#Class/Function to supress the "print" function
class hide_prints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout