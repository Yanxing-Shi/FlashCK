cmake_minimum_required(VERSION 3.8)

set(MemmoryFile
    allocator.cc
    memory_manager.cc)

add_library(memory STATIC ${MemmoryFile})
target_include_directories(memory PUBLIC
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(memory PUBLIC utils)
