
#pragma once

namespace flashck {

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