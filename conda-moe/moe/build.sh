#!/bin/bash

export CMAKE_C_COMPILER=/usr/bin/gcc
export CMAKE_C_COMPILER=/usr/bin/g++
export MOE_CC_PATH=/usr/bin/gcc
export MOE_CXX_PATH=/usr/bin/g++

export MOE_CMAKE_OPTS="-D MOE_PYTHON_INCLUDE_DIR=/usr/include/python3.6m -D MOE_PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so.1.0"

python setup.py install
