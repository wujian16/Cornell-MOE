## What is Cornell-MOE?
Cornell-MOE (Cornell-Metrics-Optimization-Engine) is a package for Bayesian Optimization (BayesOpt).  It is written in Python, with internal routines in C++. Cornell-MOE provides high-performance implementations of BayesOpt methods appropriate for industrial applications.  In addition to standard features available in other BayesOpt packages, Cornell-MOE supports parallel optimization, optimization with derivatives, and high-performance knowledge-gradient acquisition functions.
  
## What is Bayesian Optimization (BayesOpt)?
Bayesian Optimization (BayesOpt) is an approach to solving challenging optimization problems.  It uses a machine learning technique ([Gaussian process regression][30]) to estimate the objective function based on past evaluations, and then uses an [acquisition function][29] to decide where to sample next.  It typically takes longer than other optimization methods to decide where to sample, but uses fewer evaluations to find good solutions.

## When is using BayesOpt a good idea? 
BayesOpt is a good choice when:

1. Each evaluation of the objective function takes a long time to evaluate (minutes, hours, or days), or is expensive (e.g., each evaluation costs $1000 using Amazon Web Services).  This prevents evaluating the objective too many times.  In most BayesOpt applications we expect to evaluate the objective between 50 and 1000 times.

2. The objective function is a continuous function of the inputs.

3. The objective function lacks other special structure, such as convexity or concavity, that could be used by an optimization method purpose-built for problems with that structure.  We say that the objective is a "black box".

4. The inputs can be specified as a vector with a limited number of inputs.  Most successful applications of BayesOpt have fewer than 20 inputs.  Methods exist for high-dimensional Bayesian optimization, but these are not currently included in Cornell-MOE.
 
5. Constraints on the inputs are simple and inexpensive to evaluate.  Cornell-MOE supports box and simplex constraints.  Methods exist for slow- or expensive- to-evaluate constraints, but these are not currently implemented in Cornell-MOE.

6. We are interested in global rather than local optima.
 
Additionally, BayesOpt can operate when:

* evaluations are noisy;
* derivative information is unavailable;  (Most BayesOpt methods do not use derivative information, although Cornell-MOE includes methods that can use it if it is available.)
* past objective function evaluations are available, even if the points evaluated weren't chosen using BayesOpt;
* additional information about the objective is available in the form of a Bayesian prior distribution.  (This additional information might include an estimate of how quickly the objective changes in each direction.)

For more information about BayesOpt, see these [slides][28] from a talk at Stanford or this [tutorial article][29].


## Why is this package called Cornell-MOE?
"MOE" stands for "Metrics Optimization Engine", and the package was developed at Cornell.

## How does Cornell-MOE relate to the MOE BayesOpt package?
Cornell-MOE is based on the [MOE][1] BayesOpt package, developed at Yelp.   MOE is extremely fast, but can be difficult to install and use.  Cornell-MOE is designed to address these usability issues, focusing on ease of installation.  Cornell-MOE also adds algorithmic improvements (e.g., Bayesian treatment of hyperparamters in GP regression, which improves robustness) and support for several new BayesOpt algorithms: an extension of the batch expected improvement (q-EI) to the setting where derivative information is available (d-EI, [Wu et al, 2017][27]); and batch knowledge gradient with (d-KG, [Wu et al, 2017][27]) and without (q-KG, [Wu and Frazier, 2016][26]) derivative information.

## Demos:
Below we briefly describe two demos.  For more detail, please refer to main.py in the folder 'examples'. 

#### Demo 1: Batch BayesOpt

In this demo, we optimize a 1-dimensional derivative-free noisy synthetic function with a batch size of 2, using the q-KG BayesOpt method. The left-hand side shows the statistical model fitted to the objective and the points for evaluation suggested by Cornell-MOE. Note that the function evaluation is subject to noise; the right-hand side visualizes the acquisition function according to q-KG criteria.
<center><img src="https://github.com/wujian16/qKG/blob/jianwu_9_cpp_KG_gradients/qkg-demo.gif" height="350" width="600"></center>

#### Demo 2: BayesOpt with Derivatives

In this demo, we optimize a 1-dimensional synthetic function with derivatives.  We demonstrate two BayesOpt method: d-KG, which uses these derivatives; and d-EI, which does not.  Using derivatives allows d-KG to explore more efficiently.
<center><img src="https://github.com/wujian16/qKG/blob/jianwu_18_cpp_continuous_fidelity/dKG-demo.gif" height="400" width="600"></center>

## Step-by-Step Installation
We recommend installing from source (please see [Install Documentation][7] for details). We have tested the package on both the Ubuntu and CentOS operating systems. Below we provide a step-by-step instruction to install Cornell-MOE on a AWS EC2 instance with Ubuntu as the operating system.

#### step 1, install requires: python 2.6.7+, gcc 4.7.2+, cmake 2.8.9+, boost 1.51+, pip 1.2.1+, doxygen 1.8.5+

```bash
$ sudo apt-get update
$ sudo apt-get install python python-dev gcc cmake libboost-all-dev python-pip libblas-dev liblapack-dev gfortran git python-numpy python-scipy
```

#### step 2, we recommend install Cornell-MOE in the virtual environment

```bash
$ pip install virtualenv
$ virtualenv --no-site-packages ENV_NAME
```

#### step 3, set the correct environment variables for compiling the cpp code. One need to create a script with the content as follows, then **_source_** it.
```bash
export MOE_CC_PATH=/path/to/your/gcc && export MOE_CXX_PATH=/path/to/your/g++
export MOE_CMAKE_OPTS="-D MOE_PYTHON_INCLUDE_DIR=/path/to/where/Python.h/is/found -D MOE_PYTHON_LIBRARY=/path/to/python/shared/library/object"
```
For example, the script that we use on a AWS EC2 with Ubuntu OS is as follows
```bash
#!/bin/bash

export MOE_CC_PATH=/usr/bin/gcc
export MOE_CXX_PATH=/usr/bin/g++

export MOE_CMAKE_OPTS="-D MOE_PYTHON_INCLUDE_DIR=/usr/include/python2.7 -D MOE_PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython2.7.so.1.0"
```

#### step 4, finish the installment in the virtualenv.
```bash
$ source ENV_NAME/bin/activate
$ git clone https://github.com/wujian16/Cornell-MOE.git
$ cd Cornell-MOE
$ pip install -r requirements.txt
$ python setup.py install
```

## Step-by-Step Installation Guide for Python 3 Users

#### step 1, install requires: python 3.2+, gcc 4.7.2+, cmake 2.8.9+, boost 1.51+, pip 1.2.1+, doxygen 1.8.5+

```bash
$ sudo apt-get update
$ sudo apt-get install python3 python3-dev gcc cmake libboost-all-dev python3-pip libblas-dev liblapack-dev gfortran git python3-numpy python3-scipy
```

#### step 2, we recommend install Cornell-MOE in the virtual environment

```bash
$ pip3 install virtualenv
$ python3 -m venv --system-site-packages ENV_NAME
```

#### step 3, set the correct environment variables for compiling the cpp code. One need to create a script with the content as follows, then **_source_** it.
```bash
export MOE_CC_PATH=/path/to/your/gcc && export MOE_CXX_PATH=/path/to/your/g++
export MOE_CMAKE_OPTS="-D MOE_PYTHON_INCLUDE_DIR=/path/to/where/Python.h/is/found -D MOE_PYTHON_LIBRARY=/path/to/python/shared/library/object"
```
For example, the script that we use on a AWS EC2 with Ubuntu OS is as follows
```bash
#!/bin/bash

export MOE_CC_PATH=/usr/bin/gcc
export MOE_CXX_PATH=/usr/bin/g++

export MOE_CMAKE_OPTS="-D MOE_PYTHON_INCLUDE_DIR=/usr/include/python3.6m -D MOE_PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so.1.0"
```

#### step 4, finish the installment in the virtualenv.
```bash
$ source ENV_NAME/bin/activate
$ git clone https://github.com/wujian16/Cornell-MOE.git
$ cd Cornell-MOE
$ pip3 install -r requirements.txt
$ python3 setup.py install
```





## Running Cornell-MOE
See the examples in the folder 'examples'. One can run the main.py following the instruction there. The black-box functions that we would like to optimize are defined in synthetic_functions.py and real_functions.py. One can also define their own functions there.
### Mode: batch knowledge gradient (q-KG)
See [Wu and Frazier, 2016][26]. We define four synthetic functions: Branin, Rosenbrock, Hartmann3 and Hartmann6, and one real-world function: CIFRA10 (tuning a convolutional neural network on CIFAR-10). One can run main.py by the following command
with proper options.
```
# python main.py [obj_func_name] [num_to_sample] [job_id]
# q = num_to_sample
$ python main.py Hartmann3 KG(EI) 4 1
```

### Mode: derivative-enabled knowledge gradient (d-KG)
See [Wu et al, 2017][27]. We provide a large-scale kernel learning example: KISSGP class defined in real_functions.py. One note that there is a line ```self._observations = numpy.arange(self._dim)``` in
```
class KISSGP(object):
    def __init__(self):
        self._dim = 3
        self._search_domain = numpy.array([[-1, 3], [-1, 3], [-1, 3]])
        self._num_init_pts = 1
        self._sample_var = 0.0
        self._min_value = 0.0
        self._observations = numpy.arange(self._dim)
        self._num_fidelity = 0
```
which means that we access the first 3 partial derivatives. One can run this benchmark similarly by
```
$ python main.py KISSGP KG(EI) 4 1
```
If one modifies to ```self._observations = []```, and then rerun the command above, it will execute the q-KG algorithm without exploiting gradient
observations. The comparison between q-KG and d-KG on 10 independent runs are as follows,
<center><img src="https://github.com/wujian16/qKG/blob/jianwu_18_cpp_continuous_fidelity/KISSGP.jpg" height="400" width="450"></center>

## Correct performance evaluation with knowledge gradient acquisition functions

Cornell-MOE implements knowledge-gradient acquisition functions.  Evaluating performance with these acquisition requires some care.  When using expected improvement acquisition functions and deterministic function evaluations, researchers often measure performance at the best *evaluated* point.  However, when using knowledge-gradient acquisition functions, performance should be measured at the point with the *best posterior mean*, even if this point has not been previously evaluated.  This is because (1) using candidate solutions with the best posterior mean improves average-case performance for all acquisition functions; and (2) the knowledge-gradient acquisition function is designed assuming candidate solutions will be selected in this way.

As an example, Cornell MOE’s output for a problem with batch evaluations of size 4 is included below.  The points actually sampled are shown where it says "KG suggests points".  The point with the best posterior mean, which should be taken as a candidate solution when measuring performance, is shown below where it says "the recommended point".  It is called a "recommended" point because it is the point that the Bayesian analysis would recommend for a final solution if no more function evaluations could be taken.

```
best so far in the initial data 0.912960655101

KG, 1th job, 0th iteration, func=Branin, q=4
KG takes 100.078722954 seconds
KG suggests points:
[[  9.28035975   2.4600118 ]
[ 12.9719972   -1.62215938]
[  4.6534124   13.86030534]
[  6.93604372   3.11664763]]
evaluating takes capital 1.0 so far
retraining the model takes 5.53436684608 seconds
the recommended point:  [ 3.12892842  2.28794067]
recommending the point takes 1.66913294792 seconds
KG, VOI 1.70365313419, best so far 0.39866661799
```


## Citing Cornell-MOE
If you find the code useful, please kindly cite our papers [Wu and Frazier, 2016][26] and [Wu et al, 2017][27].

```bash
@inproceedings{wu2016parallel,
  title={The parallel knowledge gradient method for batch bayesian optimization},
  author={Wu, Jian and Frazier, Peter},
  booktitle={Advances in Neural Information Processing Systems},
  pages={3126--3134},
  year={2016}
}

@inproceedings{wu2017bayesian,
  title={Bayesian Optimization with Gradients},
  author={Wu, Jian and Poloczek, Matthias and Wilson, Andrew Gordon and Frazier, Peter I},
  booktitle={Advances in Neural Information Processing Systems},
  note={Accepted for publication},
  year={2017}
}
```

## Contributing
See [Contributing Documentation][8]

## License
Cornell-MOE is licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0

[1]: http://yelp.github.io/MOE/
[7]: http://yelp.github.io/MOE/install.html
[8]: http://yelp.github.io/MOE/contributing.html
[9]: http://yelp.github.io/MOE/moe.optimal_learning.python.python_version.html
[10]: http://www.youtube.com/watch?v=CC6qvzWp9_A
[11]: http://www.slideshare.net/YelpEngineering/optimal-learning-for-fun-and-profit-with-moe
[12]: http://yelp.github.io/MOE/cpp_tree.html
[13]: http://yelp.github.io/MOE/examples.html
[14]: http://yelp.github.io/MOE/objective_functions.html
[15]: http://yelp.github.io/MOE/objective_functions.html#parameters
[20]: http://people.orie.cornell.edu/pfrazier/Presentations/2014.01.Lancaster.BGO.pdf
[21]: http://yelp.github.io/MOE/why_moe.html
[23]: http://google.github.io/styleguide/pyguide.html
[24]: https://google.github.io/styleguide/cppguide.html
[25]: http://yelp.github.io/MOE/contributing.html#making-a-pull-request
[26]: https://arxiv.org/abs/1606.04414
[27]: https://papers.nips.cc/paper/7111-bayesian-optimization-with-gradients
[28]: http://mcqmc2016.stanford.edu/Frazier-Peter.pdf
[29]: https://arxiv.org/abs/1807.02811
[30]: http://www.gaussianprocess.org/gpml/
