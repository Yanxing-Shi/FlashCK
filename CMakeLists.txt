# ==============================================================================
# FlashCK CMake Configuration
# ==============================================================================
cmake_minimum_required(VERSION 3.21)

# Project configuration
project(FlashCK
    VERSION 1.0.0
    DESCRIPTION "FlashCK: fast and memory-efficient CK kernel"
    LANGUAGES CXX
)

# Add CMake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ==============================================================================
# Build Options
# ==============================================================================
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_SHARED_LIBS "Build shared libraries instead of static" OFF)

# ==============================================================================
# Build Type Configuration
# ==============================================================================
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ==============================================================================
# C++ Configuration
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==============================================================================
# HIP Configuration
# ==============================================================================
enable_language(HIP)
set(CMAKE_HIP_STANDARD 23)
set(CMAKE_HIP_EXTENSIONS ON)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# ==============================================================================
# Output Directory Configuration
# ==============================================================================
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ==============================================================================
# Dependencies
# ==============================================================================
message(STATUS "Finding dependencies...")

# ROCm/HIP
find_package(ROCM REQUIRED PATHS /opt/rocm COMPONENTS hip)
find_package(HIP REQUIRED)

if(NOT HIP_FOUND)
    message(FATAL_ERROR "HIP toolkit not found. Please install ROCm first.")
endif()

message(STATUS "Found HIP version: ${HIP_VERSION}")
message(STATUS "HIP Compiler ID: ${CMAKE_HIP_COMPILER_ID}")
message(STATUS "HIP Compiler Version: ${CMAKE_HIP_COMPILER_VERSION}")

# Logging
find_package(glog REQUIRED)
find_package(gflags REQUIRED)
add_compile_definitions(-DGLOG_USE_GLOG_EXPORT)

# String formatting
find_package(fmt REQUIRED)

# Code generation
set(JINJA2CPP_DEPS_MODE internal)
find_package(jinja2cpp REQUIRED)

# Database
find_package(SQLite3 REQUIRED)

# Pytorch python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    find_package(Python REQUIRED COMPONENTS Development)
    find_package(Torch REQUIRED)
endif()

# Testing framework
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    message(STATUS "Found GTest: ${GTest_VERSION}")
endif()

# ==============================================================================
# Include Directories
# ==============================================================================
# Add project root to include path
include_directories(${PROJECT_SOURCE_DIR})

# ==============================================================================
# Core Library
# ==============================================================================
add_subdirectory(core)

# ==============================================================================
# Cpp Library
# ==============================================================================
add_subdirectory(include/flashck)

# ==============================================================================
# Examples
# ==============================================================================
if(BUILD_EXAMPLES)
    message(STATUS "Building examples...")
    add_subdirectory(examples)
endif()

# ==============================================================================
# Tests
# ==============================================================================
if(BUILD_TESTS)
    message(STATUS "Building tests...")
    add_subdirectory(tests)
endif()

# ==============================================================================
# Installation
# ==============================================================================
# # Install headers
# install(DIRECTORY flashck/
# DESTINATION flashck
# FILES_MATCHING PATTERN "*.h"
# )

# # Install CMake config files
# include(CMakePackageConfigHelpers)

# # Create version file
# write_basic_package_version_file(
# "${CMAKE_CURRENT_BINARY_DIR}/FlashCKConfigVersion.cmake"
# VERSION ${PROJECT_VERSION}
# COMPATIBILITY SameMajorVersion
# )

# # Create config file
# configure_package_config_file(
# "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FlashCKConfig.cmake.in"
# "${CMAKE_CURRENT_BINARY_DIR}/FlashCKConfig.cmake"
# INSTALL_DESTINATION lib/cmake/FlashCK
# )

# # Install config files
# install(FILES
# "${CMAKE_CURRENT_BINARY_DIR}/FlashCKConfig.cmake"
# "${CMAKE_CURRENT_BINARY_DIR}/FlashCKConfigVersion.cmake"
# DESTINATION lib/cmake/FlashCK
# )

# ==============================================================================
# Configuration Summary
# ==============================================================================
message(STATUS "")
message(STATUS "FlashCK Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  HIP standard: ${CMAKE_HIP_STANDARD}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  Examples: ${BUILD_EXAMPLES}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "  Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Shared libraries: ${BUILD_SHARED_LIBS}")
message(STATUS "")
message(STATUS "Compiler:")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  HIP Compiler: ${CMAKE_HIP_COMPILER_ID} ${CMAKE_HIP_COMPILER_VERSION}")
message(STATUS "")
