add_library(layer_norm_op STATIC layer_norm_op.cc)
set_property(TARGET layer_norm_op PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(layer_norm_op PUBLIC flashck_graph OpenSSL::Crypto layer_norm_kernel)

add_library(rms_norm_op STATIC rms_norm_op.cc)
set_property(TARGET rms_norm_op PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(rms_norm_op PUBLIC flashck_graph OpenSSL::Crypto rms_norm_kernel)







