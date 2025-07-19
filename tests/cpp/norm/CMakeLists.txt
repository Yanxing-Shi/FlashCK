# ==============================================================================
# Unified Norm Test Framework
# ==============================================================================

# Include paths for the test framework and wrappers
set(TEST_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/flashck/wrapper/cpp
    ${PROJECT_SOURCE_DIR}/tests/cpp/common
    ${PROJECT_SOURCE_DIR}
)

# Common properties for all norm tests
function(setup_norm_test target_name)
    target_link_libraries(${target_name} PRIVATE
        layernorm_cpp_wrapper
        rmsnorm_cpp_wrapper
        GTest::gtest_main
    )
    target_include_directories(${target_name} PRIVATE ${TEST_INCLUDE_DIRS})

    # Set output directory
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/norm"
    )

    # Add to CTest
    gtest_discover_tests(${target_name})
endfunction()

# Unified norm tests
add_executable(test_layer_norm test_layer_norm.cc)
setup_norm_test(test_layer_norm)

add_executable(test_rms_norm test_rms_norm.cc)
setup_norm_test(test_rms_norm)
