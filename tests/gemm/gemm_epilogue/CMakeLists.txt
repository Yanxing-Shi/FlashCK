cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE TEST_CPPS "${PROJECT_SOURCE_DIR}/tests/gemm/gemm_epilogue/test_*.cc")

foreach (TestSource ${TEST_CPPS})
    get_filename_component(TestFilename ${TestSource} NAME)
    string(REPLACE ".cc" "" TestExecutableName ${TestFilename})
    add_executable(${TestExecutableName}  ${TestSource})
    target_link_libraries(${TestExecutableName} PUBLIC GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main "${TORCH_LIBRARIES}" linear_layer)
    add_test(NAME ${TestExecutableName} COMMAND $<TARGET_FILE:${TestExecutableName}>)
endforeach ()



