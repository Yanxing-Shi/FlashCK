#pragma once

#include <memory>

#include "lightinfer/core/graph/layer.h"
#include "lightinfer/core/graph/node.h"

#include "lightinfer/core/module/layers/gemm_layers/linear_layer.h"
#include "lightinfer/core/module/layers/norm_layers/layer_norm_layer.h"

namespace lightinfer {

template<typename T>
class FeedForwardLayer: public Layer {
public:
    FeedForwardLayer(int64_t       hidden_units,
                     int64_t       inter_size,
                     LayerNormType layer_norm_type = LayerNormType::PostLayerNorm,
                     float         epsilon         = 1e-5f);

    ~FeedForwardLayer() = default;

    Variable* operator()(Variable* in);

    void LoadParam(const T* gamma_ptr,
                   const T* beta_ptr,
                   const T* intermediate_weight_ptr,
                   const T* intermediate_bias_ptr,
                   const T* output_weight_ptr,
                   const T* output_bias_ptr);

    int64_t hidden_units_;
    int64_t inter_size_;

    LayerNormType layer_norm_type_;
    float         epsilon_;

    std::unique_ptr<LayerNormLayer<T>> layer_norm_;
    std::unique_ptr<LinearLayer<T>>    intermediate_linear_;
    std::unique_ptr<LinearLayer<T>>    output_linear_;
};

}  // namespace lightinfer