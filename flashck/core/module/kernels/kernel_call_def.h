#pragma once

#include <hip/hip_runtime.h>

#include "flashck/core/utils/dylib_utils.h"
#include "flashck/core/utils/enforce.h"

namespace flashck {

#define LOAD_SYMBOL(kernel_func, name_str)                                                                             \
    if (!Target::Instance()->kernel_lib_.has_symbol(name_str)) {                                                       \
        LI_THROW(Unavailable("Kernel symbol not found {}", name_str));                                                 \
    }                                                                                                                  \
    kernel_func = Target::Instance()->kernel_lib_.get2_function<decltype(kernel_func)>(kernel_func_name);

/*--------------------------------------------gemm---------------------------------------------------------*/
enum class GemmKernelCallType {
    Undefined            = 0,
    Gemm                 = 1,
    GemmBias             = 2,
    GemmBiasPermute      = 3,
    GemmBiasElementwise  = 4,
    Bmm                  = 5,
    BmmSoftmaxBmmPermute = 6,
    SplitKGemm           = 7,
};

// gemm kernel
void Gemm(void*, void*, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, hipStream_t);

// gemm+bias kernel
void GemmBias(void*, void*, void*, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, hipStream_t);

// gemm+bias+permute kernel
void GemmBiasPermute(void*,
                     void*,
                     void*,
                     void*,
                     int64_t,
                     int64_t,
                     int64_t,
                     int64_t,
                     int64_t,
                     int64_t,
                     int64_t,
                     int64_t,
                     int64_t,
                     hipStream_t);

// gemm+ bias + elementwise kernel
void GemmBiasElementwise(void* /*in*/,
                         void* /*weight*/,
                         void*,
                         void*,
                         void*,
                         int64_t,
                         int64_t,
                         int64_t,
                         int64_t,
                         int64_t,
                         int64_t,
                         hipStream_t);

// bmm kernel
void Bmm(void* /*in_ptr*/,
         void* /*weight_ptr*/,
         void* /*out_ptr*/,
         int64_t /*a_dim0*/,
         int64_t /*a_dim1*/,
         int64_t /*a_dim2*/,
         int64_t /*b_dim0*/,
         int64_t /*b_dim1*/,
         int64_t /*b_dim2*/,
         int64_t /*c_dim0*/,
         int64_t /*c_dim1*/,
         int64_t /*c_dim2*/,
         hipStream_t /*stream*/);

// bmm + softmax + bmm + permute kernel
void BmmSoftmaxBmmPermute(void* /*in_ptr*/,
                          void* /*weight_ptr*/,
                          void* /*out_ptr*/,
                          void* /*bias_ptr*/,
                          int64_t /*a_dim0*/,
                          int64_t /*a_dim1*/,
                          int64_t /*a_dim2*/,
                          int64_t /*b_dim0*/,
                          int64_t /*b_dim1*/,
                          int64_t /*b_dim2*/,
                          int64_t /*b1_dim0*/,
                          int64_t /*b1_dim1*/,
                          int64_t /*b1_dim2*/,
                          int64_t /*p_dim0*/,
                          hipStream_t /*stream*/);
// split_k_gemm kernel
void SplitKGemm(void*, void*, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, hipStream_t);

/*-----------------------------------------------embedding----------------------------------------------*/
enum class EmbeddingKernelCallType {
    Undefined                = 0,
    Embedding                = 1,
    EmbeddingAddAddLayerNorm = 2,
};
// embedding kernel
void Embedding(void*, void*, void*, void*, void*, int64_t, int64_t, float, hipStream_t);

// embedding + add + add + layer norm kernel
void EmbeddingAddAddLayerNorm(
    void*, void*, void*, void*, void*, void*, void*, void*, void*, int64_t, int64_t, float, hipStream_t);

/*-------------------------------------------------layernorm-----------------------------------------------------*/
enum class NormKernelCallType {
    Undefined = 0,
    LayerNorm = 1,
};

// layer norm kernel,
void LayerNorm(void* /*x_ptr*/,
               void* /*x_residual_ptr*/,
               void* /*smooth_scale_ptr*/,
               void* /*x_bias_ptr*/,
               void* /*gamma_ptr*/,
               void* /*beta_ptr*/,
               void* /*y_ptr*/,
               void* /*y_residual_ptr*/,
               void* /*y_scale_ptr*/,
               int64_t /*m*/,
               int64_t /*n*/,
               float /*eps*/,
               int64_t /*x_stride*/,
               int64_t /*xr_stride*/,
               int64_t /*y_stride*/,
               int64_t /*yr_stride*/,
               hipStream_t /*stream*/);

// rms norm kernel,
void RMSNorm(void* /*x_ptr*/,
             void* /*x_residual_ptr*/,
             void* /*smooth_scale_ptr*/,
             void* /*gamma_ptr*/,
             void* /*y_ptr*/,
             void* /*y_residual_ptr*/,
             void* /*y_scale_ptr*/,
             int64_t /*m*/,
             int64_t /*n*/,
             float /*eps*/,
             int64_t /*x_stride*/,
             int64_t /*xr_stride*/,
             int64_t /*y_stride*/,
             int64_t /*yr_stride*/,
             hipStream_t /*stream*/);

/*-------------------------------------------------fmha-----------------------------------------------------*/
void FmhaFwd(void* /*q_buf_ptr*/,
             void* /*k_buf_ptr*/,
             void* /*v_buf_ptr*/,
             void* /*bias_buf_ptr*/,
             void* /*o_buf_ptr*/,
             int64_t* /*seqstart_q_ptr*/,
             int64_t* /*seqstart_k_ptr*/,
             int64_t* /*seqlen_k_ptr*/,
             int64_t /*batch*/,
             int64_t /*seqlen_q*/,
             int64_t /*seqlen_k*/,
             int64_t /*nhead_q*/,
             int64_t /*nhead_k*/,
             int64_t /*hdim_q*/,
             int64_t /*hdim_v*/,
             int64_t /*max_seqlen_q*/,
             float /*scale*/,
             std::array<int64_t, 2> /*window_size*/,
             uint32_t /*mask_type*/,
             hipStream_t /*stream*/);

void FmhaFwdAppendKV(void* /*q_buf_ptr*/,
                     void* /*k_buf_ptr*/,
                     void* /*knew_buf_ptr*/,
                     void* /*v_buf_ptr*/,
                     void* /*vnew_buf_ptr*/,
                     void* /*cache_seqlen_k_buf_ptr*/,
                     void* /*rotary_cos_buf_ptr*/,
                     void* /*rotary_sin_buf_ptr*/,
                     void* /*block_table_buf_ptr*/,
                     int64_t* /*cache_batch_idx_buf_ptr*/,
                     int64_t /*batch*/,
                     int64_t /*seqlen_q*/,
                     int64_t /*seqlen_k*/,
                     int64_t /*nhead_q*/,
                     int64_t /*nhead_k*/,
                     int64_t /*hdim_q*/,
                     int64_t /*hdim_v*/,
                     int64_t /*seqlen_knew*/,
                     int64_t /*max_num_page_blocks*/,
                     int64_t /*page_block_size*/,
                     int64_t /*rotary_dim*/,
                     bool /*has_mask*/,
                     hipStream_t /*stream*/);

void FmhaFwdSplitKV(void* /*q_buf_ptr*/,
                    void* /*k_buf_ptr*/,
                    void* /*v_buf_ptr*/,
                    void* /*bias_buf_ptr*/,
                    void* /*lse_acc_buf_ptr*/,
                    void* /*o_acc_buf_ptr*/,
                    int64_t* /*seqstart_q_ptr*/,
                    int64_t* /*seqstart_k_ptr*/,
                    int64_t* /*seqlen_k_ptr*/,
                    void* /*block_table_buf_ptr*/,
                    int64_t* /*cache_batch_idx_ptr*/,
                    int64_t /*batch*/,
                    int64_t /*shape_seqlen_q*/,
                    int64_t /*shape_seqlen_k*/,
                    int64_t /*nhead_q*/,
                    int64_t /*nhead_k*/,
                    int64_t /*hdim_q*/,
                    int64_t /*hdim_v*/,
                    int64_t /*max_seqlen_q*/,
                    int64_t /*max_num_page_blocks*/,
                    int64_t /*page_block_size*/,
                    float /*scale*/,
                    std::array<int64_t, 2> /*window_size*/,
                    uint32_t /*mask_type*/,
                    hipStream_t /*stream*/);

void FmhaFwdSplitKVCombine(void* /*lse_acc_buf_ptr*/,
                           void* /*o_acc_buf_ptr*/,
                           void* /*o_buf_ptr*/,
                           int64_t* /*seqstart_q_ptr*/,
                           int64_t /*batch*/,
                           int64_t /*shape_seqlen_q*/,
                           int64_t /*nhead_q*/,
                           int64_t /*hdim_v*/,
                           int64_t /*max_seqlen_q*/,
                           int64_t /*num_splits*/,
                           hipStream_t /*stream*/);

}  // namespace flashck