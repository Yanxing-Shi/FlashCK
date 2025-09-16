#!/bin/bash

rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles

MY_PROJECT_tpl=$1

TORCH_PREFIX=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')
ROCM_PREFIX=/opt/rocm

cmake \
  -D CMAKE_PREFIX_PATH="${TORCH_PREFIX};${ROCM_PREFIX}" \
  -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  ${MY_PROJECT_tpl}

