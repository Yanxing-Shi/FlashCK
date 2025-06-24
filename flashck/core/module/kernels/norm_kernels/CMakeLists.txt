cmake_minimum_required(VERSION 3.8)

add_library(norm_common_kernel STATIC norm_common_kernel.cc)
set_property(TARGET norm_common_kernel PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(norm_common_kernel PUBLIC kernel_factory flashck_profiler log flags)

add_library(layer_norm_kernel STATIC layer_norm_kernel.cc)
set_property(TARGET layer_norm_kernel PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(layer_norm_kernel PUBLIC norm_common_kernel)

add_library(rms_norm_kernel STATIC rms_norm_kernel.cc)
set_property(TARGET rms_norm_kernel PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(rms_norm_kernel PUBLIC norm_common_kernel)
