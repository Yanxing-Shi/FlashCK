cmake_minimum_required(VERSION 3.8)

set(MemmoryFile
    allocator.cc
    memory_manager.cc)

add_library(flashck_memory STATIC ${MemmoryFile})
set_property(TARGET flashck_memory PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(flashck_memory PUBLIC rocm_info enforce glog::glog)




