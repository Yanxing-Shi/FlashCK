
#pragma once

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

    int64_t x_stride_;
    int64_t xr_stride_;
    int64_t y_stride_;
    int64_t yr_stride_;

    hipStream_t stream_;
};

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

}  // namespace flashck