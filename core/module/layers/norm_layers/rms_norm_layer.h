#pragma once

#include <memory>

#include "core/graph/layer.h"
#include "core/graph/node.h"
#include "core/module/operations/norm_ops/rms_norm_op.h"

namespace flashck {

/**
 * @brief RMSNorm position types for transformer architectures
 */
enum class RMSNormType {
    PreRMSNorm  = 0,  ///< Pre-normalization (before attention/FFN)
    PostRMSNorm = 1,  ///< Post-normalization (after attention/FFN)
    Unknown     = 2   ///< Unspecified position
};

/**
 * @brief High-level RMSNorm layer with parameter management
 *
 * Root Mean Square Layer Normalization is a simplified variant of LayerNorm
 * that omits the mean centering and bias term, focusing only on the scaling
 * by the root mean square. Often used in transformer architectures like LLaMA.
 *
 * Input shape: [M0, M1, ..., Mp, N1, N2, ..., ND]
 * Normalized_shape: [N1, N2, ..., ND]
 * Gamma (scale parameter) has the same shape as normalized_shape.
 *
 * @tparam T Data type (float, _Float16, ushort)
 */
template<typename T>
class RMSNormLayer: public Layer {
public:
    /**
     * @brief Construct RMSNorm layer with configuration
     * @param normalized_shape Shape of the normalization dimension
     * @param eps Epsilon for numerical stability (default: 1e-5)
     * @param fused_add Type of fused addition operation
     * @param fused_quant Type of fused quantization operation
     */
    RMSNormLayer(Shape          normalized_shape,
                 float          eps         = 1e-5,
                 FusedAddEnum   fused_add   = FusedAddEnum::NO_ADD,
                 FusedQuantEnum fused_quant = FusedQuantEnum::NO_SWEEP);

    ~RMSNormLayer() = default;

    /**
     * @brief Apply RMSNorm to input tensor
     * @param x Input tensor to normalize
     * @param x_residual Optional residual connection input
     * @param smooth_scale Optional smooth scaling factor
     * @param y_residual Optional output residual tensor
     * @param y_scale Optional output scaling factor
     * @return Normalized output tensor
     */
    Variable* operator()(Variable* x,
                         float     eps          = 1e-5,
                         Variable* x_residual   = nullptr,
                         Variable* smooth_scale = nullptr,
                         Variable* y_residual   = nullptr,
                         Variable* y_scale      = nullptr);

    /**
     * @brief Load gamma parameter from host memory
     * @param gamma_ptr Pointer to gamma (scale) parameters
     */
    void LoadParam(const T* gamma_ptr);

private:
    std::unique_ptr<Variable> gamma_var_;  ///< Learnable scale parameters

    std::unique_ptr<RMSNormOp<T>> rms_norm_op_;  ///< Underlying RMSNorm operation
};

}  // namespace flashck