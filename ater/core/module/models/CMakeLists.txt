cmake_minimum_required(VERSION 3.8)

add_executable(test_model double_gemm.cc)
set_property(TARGET test_model PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(test_model PUBLIC ater_graph linear_layer memory_utils)





