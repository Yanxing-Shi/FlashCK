add_library(bmm_rcr_op STATIC bmm_rcr_op.cc)
set_property(TARGET bmm_rcr_op PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(bmm_rcr_op PUBLIC bmm_rcr_kernel)
