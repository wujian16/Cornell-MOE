#!/bin/bash

export MOE_CC_PATH=/usr/bin/gcc
export MOE_CXX_PATH=/usr/bin/g++

export MOE_CMAKE_OPTS='-D BOOST_ROOT=/home/ubuntu/anaconda2/envs/boost_pkg -D Boost_NO_SYSTEM_PATHS=ON -D CMAKE_FIND_ROOT_PATH=/home/ubuntu/anaconda2/envs/boost_pkg -D MOE_PYTHON_INCLUDE_DIR=/home/ubuntu/anaconda2/include/python2.7 -D MOE_PYTHON_LIBRARY=/home/ubuntu/anaconda2/lib/libpython2.7.so.1.0 -D Boost_DIR=/home/ubuntu/anaconda2/envs/boost_pkg'

python setup.py install
