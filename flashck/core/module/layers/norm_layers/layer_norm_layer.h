#pragma once

#include <memory>

#include "flashck/core/graph/layer.h"
#include "flashck/core/graph/node.h"

#include "flashck/core/module/operations/norm_ops/layer_norm_op.h"

namespace flashck {

template<typename T>
class LayerNormLayer: public Layer {
public:
    LayerNormLayer(Shape          normalized_shape,
                   NormBiasEnum   is_add_bias = NormBiasEnum::NO_BIAS,
                   FusedAddEnum   fused_add   = FusedAddEnum::NO_ADD,
                   FusedQuantEnum fused_quant = FusedQuantEnum::NO_SWEEP);
    ~LayerNormLayer() = default;

    Variable* operator()(Variable* x,
                         Variable* x_bias       = nullptr,
                         Variable* x_residual   = nullptr,
                         Variable* smooth_scale = nullptr,
                         Variable* y_residual   = nullptr,
                         Variable* y_scale      = nullptr,
                         float     eps          = 1e-5);

    void LoadParam(const T* gamma_ptr, const T* beta_ptr);

private:
    std::unique_ptr<Variable> gamma_var_;
    std::unique_ptr<Variable> beta_var_;

    std::unique_ptr<LayerNormOp<T>> layer_norm_op_;
};

}  // namespace flashck