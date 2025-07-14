# CMake minimum version requirement
cmake_minimum_required(VERSION 3.21)

# Project configuration
project(FlashCK
    LANGUAGES CXX
)

# --------------------------------------------------
# Third party
# --------------------------------------------------
find_package(ROCM REQUIRED PATHS /opt/rocm COMPONENTS hip)

find_package(HIP REQUIRED)

if(NOT HIP_FOUND)
    message(FATAL_ERROR "HIP toolkit not found. Install ROCm first.")
endif()

message(STATUS "Building with HIP ${HIP_VERSION}")

find_package(glog REQUIRED)
add_compile_definitions(-DGLOG_USE_GLOG_EXPORT)

find_package(gflags REQUIRED)

find_package(fmt REQUIRED)

set(JINJA2CPP_DEPS_MODE internal)
find_package(jinja2cpp REQUIRED)

find_package(SQLite3 REQUIRED)

# Enable testing and find Google Test
enable_testing()
find_package(GTest REQUIRED)

# Find PyTorch for testing
find_package(Torch REQUIRED)

# --------------------------------------------------
# Output Configuration
# --------------------------------------------------
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --------------------------------------------------
# Build Configuration
# --------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Debug)

# set(CMAKE_BUILD_TYPE RelWithDebInfo)
# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O0 -Wall")
message(STATUS "CMake C++ Compiler ID: ${CMAKE_CXX_COMPILER_ID}")

enable_language(HIP)
set(CMAKE_HIP_STANDARD 23)
set(CMAKE_HIP_EXTENSIONS ON)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
message(STATUS "HIP Compiler ID: ${CMAKE_HIP_COMPILER_ID}")
message(STATUS "HIP Compiler Version: ${CMAKE_HIP_COMPILER_VERSION}")

# --------------------------------------------------
# Add path
# --------------------------------------------------
include_directories(
    ${PROJECT_tpl_DIR}
)

add_subdirectory(flashck)
# add_subdirectory(examples)
add_subdirectory(tests/cpp)
