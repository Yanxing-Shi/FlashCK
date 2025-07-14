add_library(layer_norm_op STATIC layer_norm_op.cc)
target_include_directories(layer_norm_op PUBLIC
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(layer_norm_op PUBLIC norm_kernel)

# add_library(rms_norm_op STATIC rms_norm_op.cc)
# set_property(TARGET rms_norm_op PROPERTY POSITION_INDEPENDENT_CODE ON)
# target_link_libraries(rms_norm_op PUBLIC graph norm_kernel)