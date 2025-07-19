# FlashCK C++ Wrapper Library
# This provides header-only C++ wrappers for FlashCK operations

cmake_minimum_required(VERSION 3.21)

# Create interface library for FlashCK C++ wrappers
add_library(flashck_cpp_wrapper INTERFACE)

# Set up include directories
target_include_directories(flashck_cpp_wrapper INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link to FlashCK core libraries
target_link_libraries(flashck_cpp_wrapper INTERFACE
    flashck_core
    flashck_profiling
    flashck_memory
    flashck_utils
    flashck_graph
    flashck_layers
)

# Set C++ standard requirements
target_compile_features(flashck_cpp_wrapper INTERFACE cxx_std_20)

# Add compile definitions
target_compile_definitions(flashck_cpp_wrapper INTERFACE
    -DFLASHCK_WRAPPER_HEADER_ONLY
)

# Create alias for easier usage
add_library(FlashCK::cpp_wrapper ALIAS flashck_cpp_wrapper)

# Install targets
install(TARGETS flashck_cpp_wrapper
    EXPORT FlashCKTargets
    INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY norm/ gemm/ fmha/
    DESTINATION include/flashck/wrapper/c++
    FILES_MATCHING PATTERN "*.h"
)

# Install main wrapper header
install(FILES flashck_wrapper.h
    DESTINATION include/flashck/wrapper/c++
)

# Create a pkg-config file
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/flashck_cpp_wrapper.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/flashck_cpp_wrapper.pc"
    @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/flashck_cpp_wrapper.pc"
    DESTINATION lib/pkgconfig
)

# Print configuration summary
message(STATUS "FlashCK C++ Wrapper configuration:")
message(STATUS "  - Header-only library: YES")
message(STATUS "  - Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  - Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  - Build tests: ${BUILD_TESTS}")