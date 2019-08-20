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
Go to the root directory of the 'pes' folder and run the run_PES.py. An example target function is provided in the run_PES.py file, which is Hartmann6. User can also define their own target function in the run_PES.py file and run the code. run_PES.py calls 'run_PES' function from the '/PES/main.py'. User can change different settings to run the optimization by changing the parameters of the 'run_PES' function. A detailed explanation of the parameters is also included in the run_PES.py file.
```bash
# Under the root directory of pes folder
$ python3 run_PES.py
```

## Reference
```bash
J. M. Hernandez-Lobato, M. W. Hoffman, and Z. Ghahramani. Predictive Entropy Search for Efficient
Global Optimization of Black-box Functions. NIPS, 2014.
```

```bash
GPy.  GPy:  A gaussian process framework in python.http://github.com/SheffieldML/GPy, since 2012.
```




