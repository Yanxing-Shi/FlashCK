cmake_minimum_required(VERSION 3.8)

file(GLOB GraphFile CONFIGURE_DEPENDS "*.cc")

add_library(graph STATIC ${GraphFile})
set_property(TARGET graph PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(graph PUBLIC utils)
