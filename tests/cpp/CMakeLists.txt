cmake_minimum_required(VERSION 3.21)

# Test 1: Basic layer normalization test
add_executable(test_layer_norm test_layer_norm.cpp)
target_link_libraries(test_layer_norm PRIVATE
    flashck_cpp_wrapper
    GTest::gtest_main
)
target_compile_features(test_layer_norm PRIVATE cxx_std_20)
target_include_directories(test_layer_norm PRIVATE ${PROJECT_SOURCE_DIR}/flashck/wrapper/c++)

# Test 2: Dynamic layer normalization test
add_executable(test_layer_norm_dynamic test_layer_norm_dynamic.cpp)
target_link_libraries(test_layer_norm_dynamic PRIVATE
    flashck_cpp_wrapper
    GTest::gtest_main
)
target_compile_features(test_layer_norm_dynamic PRIVATE cxx_std_20)
target_include_directories(test_layer_norm_dynamic PRIVATE ${PROJECT_SOURCE_DIR}/flashck/wrapper/c++)

# Test 3: Performance benchmarks
add_executable(benchmark_layer_norm benchmark_layer_norm.cpp)
target_link_libraries(benchmark_layer_norm PRIVATE
    flashck_cpp_wrapper
    GTest::gtest_main
)
target_compile_features(benchmark_layer_norm PRIVATE cxx_std_20)
target_include_directories(benchmark_layer_norm PRIVATE ${PROJECT_SOURCE_DIR}/flashck/wrapper/c++)

# Add tests to CTest
include(GoogleTest)
gtest_discover_tests(test_layer_norm)
gtest_discover_tests(test_layer_norm_dynamic)

# Add header-only norm tests subdirectory
add_subdirectory(norm)

# Set output directory
set_target_properties(
    test_layer_norm
    test_layer_norm_dynamic
    benchmark_layer_norm
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
)
