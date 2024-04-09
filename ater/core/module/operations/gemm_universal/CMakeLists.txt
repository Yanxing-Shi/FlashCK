cmake_minimum_required(VERSION 3.8)

add_library(gemm_common_op STATIC gemm_common_op.cc)
set_property(TARGET gemm_common_op PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(gemm_common_op PUBLIC ater_graph ater_profiler kernel_factory OpenSSL::Crypto glog::glog flags enforce)

add_library(gemm_rcr_op STATIC gemm_rcr_op.cc)
set_property(TARGET gemm_rcr_op PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(gemm_rcr_op PUBLIC gemm_common_op gemm_rcr_kernel)

