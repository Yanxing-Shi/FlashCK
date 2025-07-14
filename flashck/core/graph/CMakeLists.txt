cmake_minimum_required(VERSION 3.8)

file(GLOB GraphFile CONFIGURE_DEPENDS "*.cc")

add_library(graph STATIC ${GraphFile})
target_include_directories(graph PUBLIC
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(graph PUBLIC utils memory)
