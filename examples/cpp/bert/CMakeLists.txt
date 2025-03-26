add_executable(bert_example bert_example.cc)
set_property(TARGET bert_example PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(bert_example PUBLIC memory_utils bert_model)


