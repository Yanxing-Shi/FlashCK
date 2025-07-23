add_library(pybind INTERFACE norm/layer_norm.h)
target_include_directories(pybind INTERFACE
    ${PROJECT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/torch/csrc/api/include
)
target_link_libraries(pybind INTERFACE layer)

pybind11_add_module(py_module pybind.cc)
target_link_libraries(py_module
    PRIVATE pybind pybind11::module
)

target_link_libraries(py_module PRIVATE ${TORCH_LIBRARIES})