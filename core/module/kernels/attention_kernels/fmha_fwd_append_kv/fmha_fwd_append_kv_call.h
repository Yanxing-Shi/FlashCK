#pragma once

#include <hip/hip_runtime.h>

namespace flashck{
struct FmhaFwdAppendKVKernelArgs {
    void* q_ptr_;
    void* k_cache_ptr_;
    void* v_cache_ptr_;
    void* k_ptr_;
    void* v_ptr_;

    void* k_cache_seq_len_ptr_;

    void* rotary_cos_ptr_;
    void* rotary_sin_ptr_;

    void* block_table_ptr_;

    int64_t* cache_batch_idx_ptr_;

    int64_t batch_    = -1;
    int64_t q_seq_len_ = -1;
    int64_t kv_seq_len_ = -1;
    int64_t q_num_heads_  = -1;
    int64_t kv_num_heads_  = -1;
    int64_t qk_head_dim_   = -1;
    int64_t v_head_dim_   = -1;

    int64_t new_k_seq_len_ = -1;

    int64_t rotary_dim_;
    int64_t max_num_page_blocks_;
    int64_t paged_block_size_;
    bool    has_mask_;

    hipStream_t stream_;
};


void FmhaFwdAppendKV(void* /*q_buf_ptr*/,
                     void* /*k_buf_ptr*/,
                     void* /*knew_buf_ptr*/,
                     void* /*v_buf_ptr*/,
                     void* /*vnew_buf_ptr*/,
                     void* /*cache_kv_seq_len_buf_ptr*/,
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
} // namespace flashck

