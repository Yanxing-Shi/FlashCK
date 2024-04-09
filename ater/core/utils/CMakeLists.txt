cmake_minimum_required(VERSION 3.8)

add_library(flags STATIC flags.cc)
set_property(TARGET flags PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(flags PUBLIC gflags)

add_library(errors STATIC errors.cc)
set_property(TARGET errors PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(errors PUBLIC fmt::fmt)

add_library(enforce STATIC enforce.cc)
set_property(TARGET enforce PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(enforce PUBLIC errors dl fmt::fmt glog::glog flags hip::device hip::hiprand rccl)

add_library(rocm_info STATIC rocm_info.cc)
set_property(TARGET rocm_info PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(rocm_info PUBLIC enforce glog::glog flags hip::device)

add_library(printf STATIC printf.cc)
set_property(TARGET printf PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(printf PUBLIC enforce glog::glog flags hip::device)

add_library(memory_utils STATIC memory_utils.cc)
set_property(TARGET memory_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(memory_utils PUBLIC hip::device hip::hiprand enforce glog::glog flags)





