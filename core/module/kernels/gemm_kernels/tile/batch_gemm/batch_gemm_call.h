#pragma once

#include <hip/hip_runtime.h>

namespace flashck {

namespace tile{

struct BatchGemmKernelArgs {
    void* a_ptr_;
    void* b_ptr_;
    void* c_ptr_;

    int64_t split_k_;
    int64_t m_;
    int64_t n_;
    int64_t k_;

    int64_t a_stride_;
    

    hipStream_t            stream_;

};

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

} // namespace tile

} // namespace flashck