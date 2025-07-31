#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

namespace flashck {

/**
 * @brief Argument structure for normalization kernels (LayerNorm, RMSNorm)
 *
 * Contains all input/output tensors and configuration parameters
 * needed for normalization kernel execution.
 */
struct LayerNormKernelArgs {
    void* x_ptr_;             ///< Input tensor pointer
    void* x_residual_ptr_;    ///< Input residual tensor pointer (optional)
    void* smooth_scale_ptr_;  ///< Smooth scaling factor pointer (optional)
    void* x_bias_ptr_;        ///< Input bias pointer (LayerNorm only)
    void* gamma_ptr_;         ///< Gamma/scale parameter pointer
    void* beta_ptr_;          ///< Beta/bias parameter pointer (LayerNorm only)
    void* y_ptr_;             ///< Output tensor pointer
    void* y_residual_ptr_;    ///< Output residual tensor pointer (optional)
    void* y_scale_ptr_;       ///< Output scaling factor pointer (optional)

    int64_t x_dim_0_ = -1;  ///< First dimension size (batch/sequence)
    int64_t x_dim_1_ = -1;  ///< Second dimension size (features)

    float eps_;  ///< Epsilon value for numerical stability

    int64_t x_stride_  = -1;  ///< Input tensor row stride
    int64_t xr_stride_ = -1;  ///< Input residual tensor row stride
    int64_t y_stride_  = -1;  ///< Output tensor row stride
    int64_t yr_stride_ = -1;  ///< Output residual tensor row stride

    hipStream_t stream_;  ///< HIP stream for kernel execution
};

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

} // namespace flashck