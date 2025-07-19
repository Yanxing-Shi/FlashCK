cmake_minimum_required(VERSION 3.21)

# Find required packages
find_package(GTest REQUIRED)

# Add subdirectories for modular tests
add_subdirectory(norm)

# Set output directory for all test executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

# Enable CTest
include(CTest)
include(GoogleTest)
