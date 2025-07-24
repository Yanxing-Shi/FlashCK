set(CMAKE_STRIP "")
set(CMAKE_INSTALL_DO_STRIP OFF)

add_library(layer_norm INTERFACE norm/layer_norm.h)
target_include_directories(layer_norm INTERFACE
    ${PROJECT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/torch/csrc/api/include
)
target_link_libraries(layer_norm INTERFACE layer ${TORCH_LIBRARIES})

find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

add_definitions(-DTORCH_EXTENSION_NAME=flash_ck)
pybind11_add_module(flash_ck pybind.cc)
target_include_directories(flash_ck PUBLIC
    ${PYBIND11_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/torch/csrc/api/include
)
target_link_libraries(flash_ck
    PUBLIC layer_norm pybind11::module ${TORCH_PYTHON_LIBRARY}
)
set_target_properties(flash_ck PROPERTIES ENABLE_EXPORTS ON)

