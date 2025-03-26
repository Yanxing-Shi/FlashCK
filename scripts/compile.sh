#!/bin/bash 

rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles

MY_PROJECT_SOURCE=$1

cmake                                                                                             \
-D CMAKE_PREFIX_PATH=/opt/rocm                                                                    \
-D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                         \
-D CMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`             \
${MY_PROJECT_SOURCE}

