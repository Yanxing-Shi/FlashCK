cmake_minimum_required(VERSION 3.8)

file(GLOB NormFile CONFIGURE_DEPENDS "*.cc")

add_library(norm_kernel STATIC ${NormFile})
target_include_directories(norm_kernel PUBLIC
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(norm_kernel PUBLIC utils profiling kernel_factory)
