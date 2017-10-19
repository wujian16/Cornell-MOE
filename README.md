## What is Cornell-MOE?
1. Cornell-MOE is built on [MOE][1], which was open sourced by Yelp.
2. We extend the batch expected improvement (q-EI) to the setting where derivative information is available [Wu et al, 2017][27].
3. We implement batch knowledge gradient (q-KG and d-KG) in [Wu and Frazier, 2016][26] and [Wu et al, 2017][27] w/ and w/o derivative information.
4. We implement the Bayesian treatment of hyperparamters in GP regression, which makes our batch Bayesian optimization algorithms more robust.
5. We provide several examples of optimizing synthetic and real-world functions using q-KG and d-KG in the folder 'examples'. More examples are coming.
6. The project is under active development. We are revising comments in the code, and an update will be ready soon. Bug reports and issues are welcome!

## Introduction:
Below we show a small demo of Cornell-MOE on a 1-d synthetic function with a batch size q=2. The left-hand side shows the fitted statistical model and the points suggested by Cornell-MOE. Note that the function evaluation is subject to noise; the right-hand side visualizes the acquisition function according to q-KG criteria.
<div style="display: flex; justify-content: center;">
<img src="https://github.com/wujian16/qKG/blob/jianwu_9_cpp_KG_gradients/qkg-demo.gif" style="width: 200; height: 200;" />
</div>
Cornell-MOE implements a library of batch Bayesian optimization algorithms. It works by iteratively:

1. Fitting a Gaussian Process (GP) with historical data
2. Sampling the hyperparameters of the Gaussian Process via MCMC
3. Finding the set of points to sample next with highest gain, by batch Expected Improvement (q-EI) or batch knowledge gradient (q-KG) or derivative-enabled knowledge gradient (d-KG) or continuous-fidelity knowledge gradient (cf-KG)
4. Returning the points to sample

Externally you can use Cornell-MOE through the the Python interface. Please refer to the examples in the file main.py in the folder 'examples'.

## Step-by-Step Install
We recommend install from source (please see [Install Documentation][7] for details). We have tested the package on both Ubuntu and CentOS operating systems. Below we provide a step-by-step instruction to install Cornell-MOE on a AWS EC2 with Ubuntu operating system.

#### step 1, install requires: python 2.6.7+, gcc 4.7.3+, cmake 2.8.9+, boost 1.51+, pip 1.2.1+, doxygen 1.8.5+

```bash
$ apt-get update
$ apt-get install python python-dev gcc cmake libboost-all-dev python-pip doxygen libblas-dev liblapack-dev gfortran git python-numpy python-scipy
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

## Running Cornell-MOE
See the examples in the folder 'examples'. One can run the main.py following the instruction there. The black-box functions that we would like to optimize are defined in obj_functions.py. One can also define their own functions there.
### Mode: batch knowledge gradient (q-KG)
We define three synthetic functions: Branin, Hartmann3 and Rosenbrock, and one real-world function: CIFRA10 (tuning a convolutional neural network on CIFAR-10). One can run main.py by the following command
with proper options.
```
# python main.py [obj_func_name] [num_to_sample] [num_lhc] [job_id]
# q = num_to_sample
python main.py Hartmann3 4 1000 1
```

### Mode: derivative-enabled knowledge gradient (d-KG)
We provide a large-scale kernel learning example: KISSGP class defined in obj_functions.py. One note that there is a line ```self._num_observations = 3``` in
```
class KISSGP(object):
    def __init__(self):
        self._dim = 3
        self._search_domain = numpy.array([[-1, 3], [-1, 3], [-1, 3]])
        self._num_init_pts = 1
        self._sample_var = 0.0
        self._min_value = 0.0
        self._num_fidelity = 0
        self._num_observations = 3
```
which means that we access the first 3 partial derivatives. One can run this benchmark similarly by
```
python main.py KISSGP 4 1000 1
```

### Mode: continuous-fidelity knowledge gradient (cf-KG)
coming soon

## Citation
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
[20]: http://people.orie.cornell.edu/pfrazier/Presentations/2014.01.Lancaster.BGO.pdf
[21]: http://yelp.github.io/MOE/why_moe.html
[22]: http://stackoverflow.com/questions/10065526/github-how-to-make-a-fork-of-public-repository-private
[23]: http://google.github.io/styleguide/pyguide.html
[24]: https://google.github.io/styleguide/cppguide.html
[25]: http://yelp.github.io/MOE/contributing.html#making-a-pull-request
[26]: https://papers.nips.cc/paper/6307-the-parallel-knowledge-gradient-method-for-batch-bayesian-optimization
[27]: https://arxiv.org/abs/1703.04389