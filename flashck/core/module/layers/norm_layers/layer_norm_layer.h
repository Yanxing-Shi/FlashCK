#pragma once

#include <memory>

#include "flashck/core/graph/layer.h"
#include "flashck/core/graph/node.h"
#include "flashck/core/module/operations/norm_ops/layer_norm_op.h"

namespace flashck {

/**
 * @brief High-level LayerNorm layer with parameter management
 *
 * Provides a convenient interface for LayerNorm operations with automatic
 * parameter (gamma/beta) management and support for various fusion options.
 *
 * @tparam T Data type (float, _Float16, ushort)
 */
template<typename T>
class LayerNormLayer: public Layer {
public:
    /**
     * @brief Construct LayerNorm layer with configuration
     * @param normalized_shape Shape of the normalization dimension
     * @param is_add_bias Whether to include bias term
     * @param fused_add Type of fused addition operation
     * @param fused_quant Type of fused quantization operation
     */
    LayerNormLayer(Shape          normalized_shape,
                   NormBiasEnum   is_add_bias = NormBiasEnum::NO_BIAS,
                   FusedAddEnum   fused_add   = FusedAddEnum::NO_ADD,
                   FusedQuantEnum fused_quant = FusedQuantEnum::NO_SWEEP);

    ~LayerNormLayer() = default;

    /**
     * @brief Apply LayerNorm to input tensor
     * @param x Input tensor to normalize
     * @param eps Epsilon for numerical stability (default: 1e-5)
     * @param x_bias Optional input bias tensor
     * @param x_residual Optional residual connection input
     * @param smooth_scale Optional smooth scaling factor
     * @param y_residual Optional output residual tensor
     * @param y_scale Optional output scaling factor
     * @return Normalized output tensor
     */
    Variable* operator()(Variable* x,
                         float     eps          = 1e-5,
                         Variable* x_bias       = nullptr,
                         Variable* x_residual   = nullptr,
                         Variable* smooth_scale = nullptr,
                         Variable* y_residual   = nullptr,
                         Variable* y_scale      = nullptr);

    /**
     * @brief Load gamma and beta parameters from host memory
     * @param gamma_ptr Pointer to gamma (scale) parameters
     * @param beta_ptr Pointer to beta (shift) parameters
     */
    void LoadParam(const T* gamma_ptr, const T* beta_ptr);

private:
    std::unique_ptr<Variable> gamma_var_;  ///< Learnable scale parameters
    std::unique_ptr<Variable> beta_var_;   ///< Learnable shift parameters

    std::unique_ptr<LayerNormOp<T>> layer_norm_op_;  ///< Underlying LayerNorm operation
};

}  // namespace flashck