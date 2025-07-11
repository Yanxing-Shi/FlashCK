cmake_minimum_required(VERSION 3.8)

set(MemmoryFile
    allocator.cc
    memory_manager.cc)

add_library(memory STATIC ${MemmoryFile})
set_property(TARGET memory PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(memory PUBLIC utils)




