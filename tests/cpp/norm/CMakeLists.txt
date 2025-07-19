# ==============================================================================
# LayerNorm Header-Only Wrapper Tests
# ==============================================================================

# Find required packages
find_package(GTest REQUIRED)

# Correctness test for LayerNorm
add_executable(test_layer_norm_correctness test_layer_norm_correctness.cpp)
target_link_libraries(test_layer_norm_correctness PRIVATE
    FlashCK::LayerNorm
    GTest::gtest_main
)

# Performance benchmark for LayerNorm
add_executable(bench_layer_norm_performance bench_layer_norm_performance.cpp)
target_link_libraries(bench_layer_norm_performance PRIVATE
    FlashCK::LayerNorm
    GTest::gtest_main
)

# Specific optimizations for benchmark (only where needed)
target_compile_options(bench_layer_norm_performance PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -DNDEBUG>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /DNDEBUG>
)

# Add tests to CTest
include(GoogleTest)
gtest_discover_tests(test_layer_norm_correctness)

# Set output directory
set_target_properties(
    test_layer_norm_correctness
    bench_layer_norm_performance
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/norm"
)
