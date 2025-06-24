add_executable(test_layer_norm test_layer_norm.cc)
target_link_libraries(test_layer_norm PUBLIC GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main "${TORCH_LIBRARIES}" layer_norm_layer)
add_test(NAME test_layer_norm COMMAND $<TARGET_FILE:test_layer_norm>)