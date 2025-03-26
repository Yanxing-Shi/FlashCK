cmake_minimum_required(VERSION 3.8)

find_package(Threads REQUIRED)

set(GraphFile
    shape.cc
    context.cc
    node.cc
    layer.cc
    tensor.cc
    variable.cc
    operation.cc)

add_library(lightinfer_graph STATIC ${GraphFile})
set_property(TARGET lightinfer_graph PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(lightinfer_graph PUBLIC memory_utils lightinfer_memory lightinfer_profiler enforce printf)