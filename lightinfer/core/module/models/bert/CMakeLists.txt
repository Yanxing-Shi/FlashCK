cmake_minimum_required(VERSION 3.8)

add_library(bert_weight STATIC bert_weight.cc)
set_property(TARGET bert_weight PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(bert_weight PUBLIC transformer_encoder_layer_weight log)

add_library(bert_model STATIC bert_model.cc)
set_property(TARGET bert_model PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(bert_model PUBLIC bert_weight embedding_layer transformer_encoder_layer linear_layer feed_data_map memory_utils)

