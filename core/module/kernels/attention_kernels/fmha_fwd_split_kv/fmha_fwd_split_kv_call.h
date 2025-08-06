#pragma once

#include <hip/hip_runtime.h>

namespace flashck{
struct FmhaFwdSplitKVKernelArgs {
    void* q_ptr_;
    void* k_ptr_;
    void* v_ptr_;
    void* bias_ptr_;
    void* lse_acc_ptr_;
    void* out_acc_ptr_;

    int64_t* q_seq_start_ptr_;
    int64_t* kv_seq_start_ptr_;
    int64_t* k_seq_len_ptr_;

    void*    block_table_ptr_;
    int64_t* cache_batch_idx_ptr_;

    int64_t batch_    = -1;
    int64_t q_seq_len_ = -1;
    int64_t kv_seq_len_ = -1;
    int64_t q_num_heads_  = -1;
    int64_t kv_num_heads_  = -1;
    int64_t qk_head_dim_   = -1;
    int64_t v_head_dim_   = -1;

    int64_t q_max_seq_len_ = -1;

    int64_t max_num_page_blocks_;
    int64_t paged_block_size_;

    float scale_;

    std::array<int64_t, 2> window_size_;
    uint32_t               mask_type_;

    hipStream_t stream_;
};

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

} // namespace flashck

