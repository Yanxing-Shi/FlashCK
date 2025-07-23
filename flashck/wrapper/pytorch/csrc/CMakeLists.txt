find_package(pybind11 REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)

add_library(layer_norm INTERFACE norm/layer_norm.h)
target_include_directories(layer_norm INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/norm
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/torch/csrc/api/include
)

pybind11_add_module(py_module pybind.cc)
target_link_libraries(py_module
    PRIVATE layer_norm pybind11::module
)

find_package(Torch REQUIRED)
target_link_libraries(py_module PRIVATE ${TORCH_LIBRARIES})