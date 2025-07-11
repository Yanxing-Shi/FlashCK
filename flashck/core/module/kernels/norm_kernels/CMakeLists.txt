cmake_minimum_required(VERSION 3.8)

file(GLOB NormFile CONFIGURE_DEPENDS "*.cc")

add_library(norm STATIC ${NormFile})
set_property(TARGET norm PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(norm PUBLIC utils)
