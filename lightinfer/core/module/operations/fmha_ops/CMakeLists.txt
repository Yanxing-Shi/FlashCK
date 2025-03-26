cmake_minimum_required(VERSION 3.8)

add_library(fmha_fwd_op STATIC fmha_fwd_op.cc)
set_property(TARGET fmha_fwd_op PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fmha_fwd_op PUBLIC fmha_fwd_kernel lightinfer_graph OpenSSL::Crypto log flags enforce)

add_library(fmha_fwd_appendkv_op STATIC fmha_fwd_appendkv_op.cc)
set_property(TARGET fmha_fwd_appendkv_op PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fmha_fwd_appendkv_op PUBLIC fmha_fwd_appendkv_kernel lightinfer_graph OpenSSL::Crypto log flags enforce)

add_library(fmha_fwd_splitkv_op STATIC fmha_fwd_splitkv_op.cc)
set_property(TARGET fmha_fwd_splitkv_op PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fmha_fwd_splitkv_op PUBLIC fmha_fwd_splitkv_kernel lightinfer_graph OpenSSL::Crypto log flags enforce)

add_library(fmha_fwd_splitkv_combine_op STATIC fmha_fwd_splitkv_combine_op.cc)
set_property(TARGET fmha_fwd_splitkv_combine_op PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fmha_fwd_splitkv_combine_op PUBLIC fmha_fwd_splitkv_combine_kernel lightinfer_graph OpenSSL::Crypto log flags enforce)



