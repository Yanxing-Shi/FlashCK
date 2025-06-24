
#pragma once

#include "flashck/core/graph/layer.h"
#include "flashck/core/graph/node.h"

#include "flashck/core/module/layers/attention_layers/memory_efficient_attention_layer.h"
#include "flashck/core/module/layers/gemm_layers/linear_layer.h"
#include "flashck/core/module/layers/norm_layers/layer_norm_layer.h"

namespace flashck {

template<typename T>
class FuseMultiHeadAttentionLayer: public Layer {
public:
    FuseMultiHeadAttentionLayer(int64_t                  q_num_heads,
                                int64_t                  kv_num_heads,
                                int64_t                  qk_head_dim,
                                int64_t                  v_head_dim,
                                float                    scale           = 1.0f,
                                BiasEnum                 bias_enum       = BiasEnum::NO_BIAS,
                                std::array<int64_t, 2>   window_size     = {-1, -1},
                                GenericAttentionMaskEnum mask_enum       = GenericAttentionMaskEnum::NO_MASK,
                                LayerNormType            layer_norm_type = LayerNormType::PostLayerNorm,
                                float                    epsilon         = 1e-5f,
                                bool                     is_qkv_packed   = true,
                                bool                     has_residual    = true,
                                bool                     use_qkv_bias    = true,
                                bool                     use_out_bias    = true);

    ~FuseMultiHeadAttentionLayer() = default;

    Variable* operator()(Variable* x);

    // qkv packed
    void LoadParam(const T* gamma_ptr,
                   const T* beta_ptr,
                   const T* qkv_weight_ptr,
                   const T* qkv_bias_ptr,
                   const T* out_weight_ptr,
                   const T* out_bias_ptr);

    // non-qkv packed
    void LoadParam(const T* gamma_ptr,
                   const T* beta_ptr,
                   const T* q_weight_ptr,
                   const T* q_bias_ptr,
                   const T* k_weight_ptr,
                   const T* k_bias_ptr,
                   const T* v_weight_ptr,
                   const T* v_bias_ptr,
                   const T* out_weight_ptr,
                   const T* out_bias_ptr);

    int64_t q_num_heads_;
    int64_t kv_num_heads_;
    int64_t qk_head_dim_;
    int64_t v_head_dim_;

    float                    scale_;
    BiasEnum                 bias_enum_;
    std::array<int64_t, 2>   window_size_;
    GenericAttentionMaskEnum mask_enum_;

    LayerNormType layer_norm_type_;
    float         epsilon_;

    bool is_qkv_packed_;
    bool has_residual_;
    bool use_qkv_bias_;
    bool use_out_bias_;

    std::unique_ptr<LayerNormLayer<T>> layer_norm_;

    // packed qkv
    std::unique_ptr<LinearLayer<T>> qkv_in_proj_;
    // non-packed qkv
    std::unique_ptr<LinearLayer<T>> q_in_proj_;
    std::unique_ptr<LinearLayer<T>> k_in_proj_;
    std::unique_ptr<LinearLayer<T>> v_in_proj_;

    std::unique_ptr<MemoryEfficientAttentionLayer<T>> fmha_fwd_;
    std::unique_ptr<LinearLayer<T>>                   out_proj_;
};
}  // namespace flashck
