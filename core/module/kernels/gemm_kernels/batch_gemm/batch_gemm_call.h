#pragma once

#include <hip/hip_runtime.h>

namespace flashck {

struct BatchGemmKernelArgs {
    void* a_ptr_;
    void* b_ptr_;
    void* c_ptr_;

    int64_t batch_;
    int64_t split_k_;
    int64_t m_;
    int64_t n_;
    int64_t k_;

    int64_t a_stride_;
    int64_t b_stride_;
    int64_t c_stride_;

    int64_t a_batch_stride_;
    int64_t b_batch_stride_;
    int64_t c_batch_stride_;

    hipStream_t            stream_;

};

void BatchGemm(void* /*a_ptr*/, 
               void* /*b_ptr*/, 
               void* /*c_ptr*/, 
               int64_t /*batch*/, 
               int64_t /*split_k*/, 
               int64_t /*m*/, 
               int64_t /*n*/, 
               int64_t /*k*/, 
               int64_t /*a_stride*/, 
               int64_t /*b_stride*/, 
               int64_t /*c_stride*/,
               int64_t /*a_batch_stride*/,
               int64_t /*b_batch_stride*/,
               int64_t /*c_batch_stride*/,
               hipStream_t /*stream*/
            );

} // namespace flashck