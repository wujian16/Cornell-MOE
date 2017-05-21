What is qKG?
====
1. qKG is built upon [MOE][1], which was open sourced by Yelp.
2. We extend the batch expected improvement (qEI) to the setting where derivative information is available [Wu et al, 2017][27].
3. We implement batch knowledge gradient (qKG and dKG) in [Wu and Frazier, 2016][26] and [Wu et al, 2017][27] w/ and w/o derivative information.
4. We implement the Bayesian treatment of hyperparamters in GP, which makes our batch Bayesian optimization algorithms more robust.
5. We provide one example of optimizing Branin function using qKG in the folder qkg_examples, more examples are coming.
6. The project is under active development. We are revising all the comments in the code, which will be ready soon. Any bug report or issue is welcome!

# Introduction:
Below we show a small demo of qKG on a 1-d synthetic function with the batch size q=2. The right hand side shows the fitted statistical model and the points suggested by qKG, note that the function evaluation is subject to noise; the left hand side is the heatmap visualizing the acquisition function according to qKG criteria.
![qKG demo](https://github.com/wujian16/qKG/blob/jianwu_9_cpp_KG_gradients/qkg-demo.gif)

qKG implements a library of batch Bayesian optimization algorithms, it internally works by:

1. Building a Gaussian Process (GP) with the historical data
2. Sampling the hyperparameters of the Gaussian Process (model selection) with MCMC algorithm
3. Finding the set of points of highest gain (by batch Expected Improvement (qEI) or batch knowledge gradient (qKG))
4. Returning the points to sample, then repeat

Externally you can use qKG/MOE through the the Python interface. Please refer to the example in the file bayesian.test.functions.py in the folder qkg_examples.

# Step-by-Step Install

We recommend install from source (please see [Install Documentation][7] for details). We have tested the package on both Ubuntu and CentOS operating systems. Below we provide a step-by-step instruction to install qKG/MOE on a AWS EC2 with Ubuntu operating system.

** step 1, install requires: python 2.6.7+, gcc 4.7.3+, cmake 2.8.9+, boost 1.51+, pip 1.2.1+, doxygen 1.8.5+

```bash
$ apt-get update
$ apt-get install python python-dev gcc cmake libboost-all-dev python-pip doxygen libblas-dev liblapack-dev gfortran git python-numpy python-scipy
```

** step 2, we recommend install qKG/MOE in the virtual environment

```bash
$ pip install virtualenv
$ virtualenv --no-site-packages ENV_NAME
```

** step 3, set the correct environment variables for compiling the cpp code. One need to create a script with the content as follows, then source it.
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

*** step 4, finish the installment in the virtualenv.
```bash
$ source ENV_NAME/bin/activate
$ git clone https://github.com/wujian16/qKG.git
$ cd qKG
$ pip install -r requirements.txt
$ python setup.py install
```

# Citation

If you find the code useful, please kindly cite our papers [Wu and Frazier, 2016][26] and [Wu et al, 2017][27].

# Running MOE/qKG
## Within Python
See the example in the folder qkg_examples. One can run the bayesian.test.functions.py following the instruction there. The black-box functions that we would like to optimize is defined in obj_functions.py. One can also define their own function there.

# Contributing
See [Contributing Documentation][8]

# License
qKG/MOE is licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0

[0]: https://www.youtube.com/watch?v=CC6qvzWp9_A
[1]: http://yelp.github.io/MOE/
[2]: http://yelp.github.io/MOE/moe.views.rest.html
[3]: http://github.com/Yelp/MOE/pulls
[4]: http://yelp.github.io/MOE/moe.views.rest.html#module-moe.views.rest.gp_ei
[5]: http://yelp.github.io/MOE/moe.easy_interface.html
[6]: http://docs.docker.io/
[7]: http://yelp.github.io/MOE/install.html
[8]: http://yelp.github.io/MOE/contributing.html
[9]: http://yelp.github.io/MOE/moe.optimal_learning.python.python_version.html
[10]: http://www.youtube.com/watch?v=CC6qvzWp9_A
[11]: http://www.slideshare.net/YelpEngineering/optimal-learning-for-fun-and-profit-with-moe
[12]: http://yelp.github.io/MOE/cpp_tree.html
[13]: http://yelp.github.io/MOE/examples.html
[14]: http://yelp.github.io/MOE/objective_functions.html
[15]: http://yelp.github.io/MOE/objective_functions.html#parameters
[16]: http://people.orie.cornell.edu/pfrazier/
[17]: http://www.orie.cornell.edu/
[18]: http://optimallearning.princeton.edu/
[19]: http://orfe.princeton.edu/
[20]: http://people.orie.cornell.edu/pfrazier/Presentations/2014.01.Lancaster.BGO.pdf
[21]: http://yelp.github.io/MOE/why_moe.html
[22]: http://stackoverflow.com/questions/10065526/github-how-to-make-a-fork-of-public-repository-private
[23]: http://google.github.io/styleguide/pyguide.html
[24]: https://google.github.io/styleguide/cppguide.html
[25]: http://yelp.github.io/MOE/contributing.html#making-a-pull-request
[26]: https://papers.nips.cc/paper/6307-the-parallel-knowledge-gradient-method-for-batch-bayesian-optimization
[27]: https://arxiv.org/abs/1703.04389
