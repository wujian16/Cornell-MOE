conda-moe
=========

This repository provides the recipes for building binary packages of cornel-moe. The package cornell_moe2 is built with Python 2.7 and cornell\_moe3 is built with Python 3.7.

To build the binaries, follow the steps (this guide is for re-building the cornell_moe2 package. The process for building cornell\_moe3 is similar).

#### Step 1, Go to PATH-TO/Cornell-Moe/conda-recipe/cornell-moe2

#### Step 2, Modify the build.sh file to set the correct arguments for the environment variable MOE_CMAKE_OPTS. 

Specifically, the MOE_PYTHON_INCLUDE_DIR and MOE_PYTHON_LIBRARY arguments should be set to the path where the Python.h file and the python shared library is found (you may refer to the "Step-by-Step 
Installation Guide" section on how to set those arguments). The other arguments can be omitted and deleted if you build the binary with the pre-installed system default boost package. If you choose to build the binary with 
a user-installed version of boost. You need to set the Boost_NO_SYSTEM_PATHS=ON and set CMAKE_FIND_ROOT_PATH and BOOST_ROOT to be where your customized Boost is installed. In particular, if you set the two arguments 
to point to a particular folder, the libboost_.*.so files should live in ${BOOST_ROOT}/lib or ${BOOST_ROOT}/stage/lib and boost header files (e.g., python.hpp) should live in ${BOOST_ROOT}/boost or ${BOOST_ROOT}/include/boost. 

#### Step 3, Modify the meta.yaml file so the version of python and boost matches the version you use to build the package. 

For example, if you build the package with python 2.7 and boost with version 1.65.1, you need to specify the version of boost and python in both the "build" and "run" section under the "requirements" section.  

#### Step 4, Run the following commands to build the package
````bash
conda install conda-build
cd PATH-TO/Cornell-Moe/conda-recipe
conda build cornell-moe2
```

