cmake_minimum_required(VERSION 3.8)

add_library(gemm_common_kernel STATIC gemm_common_kernel.cc)
set_property(TARGET gemm_common_kernel PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(gemm_common_kernel PUBLIC ater_profiler jinja2cpp glog::glog flags)

add_library(gemm_rcr_kernel STATIC gemm_rcr_kernel.cc)
set_property(TARGET gemm_rcr_kernel PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(gemm_rcr_kernel PUBLIC gemm_common_kernel kernel_factory)

