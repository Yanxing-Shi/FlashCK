#pragma once

#include "lightinfer/core/graph/layer.h"
#include "lightinfer/core/graph/node.h"

#include "lightinfer/core/module/layers/gemm_layers/linear_layer.h"
#include "lightinfer/core/module/layers/norm_layers/layer_norm_layer.h"

namespace lightinfer {

template<typename T>
class LlamaMLPLayer: public Layer {
public:
    LlamaMLPLayer(int64_t hidden_dim,
                  int64_t inter_size,
                  bool    use_up_bias   = true,
                  bool    use_down_bias = true,
                  bool    use_gate_bias = true);

    ~LlamaMLPLayer();

    Variable* operator()(Variable* in);

    void LoadParam(const T* gamma_ptr,
                   const T* beta_ptr,
                   const T* up_weight_ptr,
                   const T* up_bias_ptr,
                   const T* down_weight_ptr,
                   const T* down_bias_ptr,
                   const T* gate_weight_ptr,
                   const T* gate_bias_ptr);

    int64_t hidden_dim_;
    int64_t inter_size_;

    bool use_up_bias_;
    bool use_down_bias_;
    bool use_gate_bias_;

    std::unique_ptr<LayerNormLayer<T>> pre_layer_norm_;
    std::unique_ptr<LinearLayer<T>>    gate_proj_;
    std::unique_ptr<LinearLayer<T>>    up_proj_;
    std::unique_ptr<LinearLayer<T>>    down_proj_;
};

}  // namespace lightinfer