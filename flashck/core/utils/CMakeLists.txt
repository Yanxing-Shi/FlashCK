cmake_minimum_required(VERSION 3.8)

set(UtilsFile
    timer.cc
    subprocess_utils.cc
    string_utils.cc
    printf.cc
    macros.cc
    init.cc
    flags.cc
    dylib_utils.cc
    dtype.cc
    errors.cc
    file_utils.cc
    memory_utils.cc
    hip_utils.cc)

add_library(utils STATIC ${UtilsFile})
set_property(TARGET utils PROPERTY POSITION_INDEPENDENT_CODE ON)
# target_link_libraries(utils PUBLIC)










