#!/bin/bash

export MOE_CC_PATH=/usr/bin/gcc
export MOE_CXX_PATH=/usr/bin/g++

export MOE_CMAKE_OPTS='-D BOOST_ROOT=/home/ubuntu/anaconda3 -D Boost_NO_SYSTEM_PATHS=ON -D CMAKE_FIND_ROOT_PATH=/home/ubuntu/anaconda3  -D MOE_PYTHON_INCLUDE_DIR=/home/ubuntu/anaconda3/include/python3.7m -D MOE_PYTHON_LIBRARY=/home/ubuntu/anaconda3/lib/libpython3.7m.so.1.0 -D Boost_DIR=/home/ubuntu/anaconda3'

python setup.py install
