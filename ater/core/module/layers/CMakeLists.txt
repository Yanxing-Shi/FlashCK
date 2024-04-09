add_library(linear_layer STATIC linear_layer.cc)
set_property(TARGET linear_layer PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(linear_layer PUBLIC ater_graph gemm_rcr_op)