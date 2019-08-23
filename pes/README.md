## Predictive Entropy Search
This code is an implementation of the 'Predictive Entropy Search for Efficient
Global Optimization of Black-box Functions' by J. M. Hernandez-Lobato, M. W. Hoffman, and Z. Ghahramani. 
```bash
J. M. Hernandez-Lobato, M. W. Hoffman, and Z. Ghahramani. Predictive Entropy Search for Efficient
Global Optimization of Black-box Functions. NIPS, 2014.
```

## Installation Guide for Predictive Entropy Search 

#### Step 1, install the requires: python3, python3-pip, python3-dev, build-essential.

```bash
$ sudo apt-get update
$ sudo apt-get install python3 python3-pip python3-dev build-essential
```

#### Step 2, install necessary python libraries: numpy, scipy, matplotlib, Gpy. For the successful installation of Gpy, you need to install the first three libraries. 

```bash
$ pip3 install numpy scipy matplotlib 
$ pip3 install gpy 
```

#### Step 3, download the code and get to the root directory.
```bash
$ git clone https://github.com/wujian16/Cornell-MOE.git
```


## Running PES
Go to the root directory of the 'pes' folder and run the run_PES.py. An example target function is provided in the run_PES.py file, which is Hartmann6. User can also define their own target function in the run_PES.py file and run the code. run_PES.py calls 'run_PES' function from the '/PES/main.py'. User can change different settings to run the optimization by changing the parameters of the 'run_PES' function. A detailed explanation of the parameters is also included in the run_PES.py file. The results will be stored at the root directory in 'Xsamples.txt', 'Ysamples.txt', and 'guesses.txt' files.  
```bash
# Under the root directory of pes folder
$ python3 run_PES.py
```

## Sample Output
A sample of the output of PES run on Hartmann6 is included below. The point sampled by PES is shown under the 'PES suggests'. For 'the recommended point', we use the minimum between the best *evaluated* point so far and the *best posterior mean*. Here, it is saying 'the recommended point' because we assume the algorithm will terminate at this iteration and no more function evaluations will be made. Therefore, it is the final solution. For the function value of the minimum, in this case the function value of the final solution, it is displayed at 'Best so far'. 


```
Best so far in the initial data -0.30428643211919326

PES, 0th job, 0th iteration
PES takes 113.0760588645935 seconds
PES suggests: 
[0.85731774 0.60154422 0.17818856 0.78775361 0.65906122 0.0667066 ]
Retraining the model takes 73.91704773902893 seconds
The recommended point [0.56829938 0.73410096 0.2985657  0.31553598 0.12493136 0.34408206]
Recommending the point takes 3.6333107948303223 seconds
Best so far -0.3931226190740133
```

A sample of the progress plot of PES run on Hartmann6 is shown below:
<center><img src="https://github.com/dukezhang007/pes/blob/master/plots/PES_sample_plot.png" height="325" width="450"></center>

## Reference
```bash
J. M. Hernandez-Lobato, M. W. Hoffman, and Z. Ghahramani. Predictive Entropy Search for Efficient
Global Optimization of Black-box Functions. NIPS, 2014.
```

```bash
GPy.  GPy:  A gaussian process framework in python.http://github.com/SheffieldML/GPy, since 2012.
```




