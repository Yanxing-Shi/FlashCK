
#pragma once

#include <hip/hip_runtime.h>

namespace flashck {

struct NormKernelArgs {
    void* x_ptr_;
    void* x_residual_ptr_;
    void* smooth_scale_ptr_;
    void* x_bias_ptr_;  // layer_norm
    void* gamma_ptr_;
    void* beta_ptr_;  // layer_norm
    void* y_ptr_;
    void* y_residual_ptr_;
    void* y_scale_ptr_;

    int64_t x_dim_0_ = -1;
    int64_t x_dim_1_ = -1;

    float eps_;

    int64_t x_stride_  = -1;
    int64_t xr_stride_ = -1;
    int64_t y_stride_  = -1;
    int64_t yr_stride_ = -1;

    hipStream_t stream_;
};

}  // namespace flashck