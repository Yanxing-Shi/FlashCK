
#pragma once

#include <hip/hip_runtime.h>

namespace flashck {

/**
 * @brief Argument structure for normalization kernels (LayerNorm, RMSNorm)
 *
 * Contains all input/output tensors and configuration parameters
 * needed for normalization kernel execution.
 */
struct NormKernelArgs {
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

}  // namespace flashck