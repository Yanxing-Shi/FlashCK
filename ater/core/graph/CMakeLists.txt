cmake_minimum_required(VERSION 3.8)

find_package(Threads REQUIRED)

set(AterGraphFile
    shape.cc
    context.cc
    node.cc
    layer.cc
    tensor.cc
    variable.cc
    operation.cc)

add_library(ater_graph STATIC ${AterGraphFile})
set_property(TARGET ater_graph PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ater_graph PUBLIC memory_utils ater_memory ater_profiler enforce printf gemm_rcr_op)