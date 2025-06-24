cmake_minimum_required(VERSION 3.8)

add_executable(test_memory_efficient_attention test_memory_efficient_attention.cc)
target_link_libraries(test_memory_efficient_attention PUBLIC GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main "${TORCH_LIBRARIES}" memory_efficient_attention_layer)
add_test(NAME test_memory_efficient_attention COMMAND $<TARGET_FILE:test_memory_efficient_attention>)

# add_executable(test_fuse_multi_head_attention test_fuse_multi_head_attention.cc)
# target_link_libraries(test_fuse_multi_head_attention PUBLIC GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main "${TORCH_LIBRARIES}" fuse_multi_head_attention_layer)
# add_test(NAME test_fuse_multi_head_attention COMMAND $<TARGET_FILE:test_fuse_multi_head_attention>)

add_executable(test_memory_efficient_attention_decoder test_memory_efficient_attention_decoder.cc)
target_link_libraries(test_memory_efficient_attention_decoder PUBLIC GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main "${TORCH_LIBRARIES}" memory_efficient_attention_decoder_layer)
add_test(NAME test_memory_efficient_attention_decoder COMMAND $<TARGET_FILE:test_memory_efficient_attention_decoder>)