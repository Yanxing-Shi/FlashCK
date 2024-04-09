cmake_minimum_required(VERSION 3.8)

set(AterMemmoryFile
    allocator.cc
    memory_manager.cc)

add_library(ater_memory STATIC ${AterMemmoryFile})
set_property(TARGET ater_memory PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(ater_memory PUBLIC rocm_info enforce glog::glog)




