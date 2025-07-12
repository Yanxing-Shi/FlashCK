cmake_minimum_required(VERSION 3.8)

file(GLOB UtilsFile CONFIGURE_DEPENDS "*.cc")

add_library(utils STATIC ${UtilsFile})
target_include_directories(utils PUBLIC
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(utils PUBLIC
    glog::glog gflags jinja2cpp fmt::fmt hip::device)
