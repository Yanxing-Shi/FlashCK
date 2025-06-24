#pragma once

#include <memory>

#include "flashck/core/graph/layer.h"
#include "flashck/core/graph/node.h"

#include "flashck/core/module/operations/norm_ops/rms_norm_op.h"

namespace flashck {

enum class RMSNormType {
    PreRMSNorm  = 0,
    PostRMSNorm = 1,
    Unknown     = 2
};

/*
Standalone layernorm op.
Applies Layer Normalization over a mini-batch of inputs as described in the
paper Layer Normalization. The mean and standard-deviation are calculated
over the last D dimensions, where D is the dimension of normalized_shape.
Input shape: [M0, M1, ..., Mp, N1, N2, ..., ND]
Normalized_shape: [N1, N2, ..., ND]
Gamma/Beta, if not None, have the same shape as normalized_shape.
*/

template<typename T>
class RMSNormLayer: public Layer {
public:
    RMSNormLayer(Shape          normalized_shape,
                 float          eps         = 1e-5,
                 FusedAddEnum   fused_add   = FusedAddEnum::NO_ADD,
                 FusedQuantEnum fused_quant = FusedQuantEnum::NO_SWEEP);
    ~RMSNormLayer() = default;

    Variable* operator()(Variable* x,
                         Variable* x_residual   = nullptr,
                         Variable* smooth_scale = nullptr,
                         Variable* y_residual   = nullptr,
                         Variable* y_scale      = nullptr);

    void LoadParam(const T* gamma_ptr);

private:
    Shape normalized_shape_;
    float eps_;

    FusedAddEnum   fused_add_   = FusedAddEnum::NO_ADD;
    FusedQuantEnum fused_quant_ = FusedQuantEnum::NO_SWEEP;

    std::unique_ptr<Variable> gamma_var_;

    std::string layer_norm_op_name_ = "rms_norm";

    std::unique_ptr<RMSNormOp<T>> rms_norm_op_;
};

}  // namespace flashck