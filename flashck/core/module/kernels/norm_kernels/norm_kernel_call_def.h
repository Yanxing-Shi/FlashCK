
#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

namespace flashck {

/**
 * @brief LayerNorm kernel function interface
 * @param x_ptr Input tensor pointer
 * @param x_residual_ptr Input residual tensor pointer (optional)
 * @param smooth_scale_ptr Smooth scaling factor pointer (optional)
 * @param x_bias_ptr Input bias pointer
 * @param gamma_ptr Gamma/scale parameter pointer
 * @param beta_ptr Beta/bias parameter pointer
 * @param y_ptr Output tensor pointer
 * @param y_residual_ptr Output residual tensor pointer (optional)
 * @param y_scale_ptr Output scaling factor pointer (optional)
 * @param m First dimension size (batch/sequence)
 * @param n Second dimension size (features)
 * @param eps Epsilon value for numerical stability
 * @param x_stride Input tensor row stride
 * @param xr_stride Input residual tensor row stride
 * @param y_stride Output tensor row stride
 * @param yr_stride Output residual tensor row stride
 * @param stream HIP stream for kernel execution
 */
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

/**
 * @brief RMSNorm kernel function interface
 * @param x_ptr Input tensor pointer
 * @param x_residual_ptr Input residual tensor pointer (optional)
 * @param smooth_scale_ptr Smooth scaling factor pointer (optional)
 * @param gamma_ptr Gamma/scale parameter pointer
 * @param y_ptr Output tensor pointer
 * @param y_residual_ptr Output residual tensor pointer (optional)
 * @param y_scale_ptr Output scaling factor pointer (optional)
 * @param m First dimension size (batch/sequence)
 * @param n Second dimension size (features)
 * @param eps Epsilon value for numerical stability
 * @param x_stride Input tensor row stride
 * @param xr_stride Input residual tensor row stride
 * @param y_stride Output tensor row stride
 * @param yr_stride Output residual tensor row stride
 * @param stream HIP stream for kernel execution
 */
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