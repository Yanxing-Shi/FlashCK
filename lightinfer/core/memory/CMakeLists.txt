cmake_minimum_required(VERSION 3.8)

set(MemmoryFile
    allocator.cc
    memory_manager.cc)

add_library(lightinfer_memory STATIC ${MemmoryFile})
set_property(TARGET lightinfer_memory PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(lightinfer_memory PUBLIC rocm_info enforce glog::glog)




