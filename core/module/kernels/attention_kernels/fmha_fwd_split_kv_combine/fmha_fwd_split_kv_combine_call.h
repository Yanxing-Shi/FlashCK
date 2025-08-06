#pragma once

#include <hip/hip_runtime.h>

namespace flashck{
struct FmhaFwdSplitKVCombineKernelArgs {
    
    void* lse_acc_ptr_;
    void* out_acc_ptr_;
    void* out_ptr_;

    int64_t* q_seq_start_ptr_;

    int64_t batch_    = -1;
    int64_t q_seq_len_ = -1;
    int64_t q_num_heads_  = -1;
    int64_t v_head_dim   = -1;

    int64_t q_max_seq_len_ = -1;

    int64_t num_splits_ = -1;

    hipStream_t stream_;
};


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