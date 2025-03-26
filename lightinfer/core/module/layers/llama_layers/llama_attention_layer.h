#pragma once

#include "lightinfer/core/graph/layer.h"
#include "lightinfer/core/graph/node.h"

#include "lightinfer/core/module/layers/gemm_layers/linear_layer.h"
#include "lightinfer/core/module/layers/norm_layers/layer_norm_layer.h"
#include "lightinfer/core/module/operations/elementwise_ops/concat_op.h"
#include "lightinfer/core/module/operations/elementwise_ops/reshape_op.h"
#include "lightinfer/core/module/operations/elementwise_ops/split_op.h"
// #include "lightinfer/core/module/operations/embedding_ops/rope_op.h"
#include "lightinfer/core/module/operations/gemm_universal_ops/bmm_softmax_bmm_permute_op.h"

namespace lightinfer {

template<typename T>
class LlamaAttentionLayer: public Layer {
public:
    LlamaAttentionLayer(int64_t         seq_len,
                        int64_t         hidden_dim,
                        int64_t         num_heads,
                        float           scale        = 1.0f,
                        bool            has_residual = true,
                        bool            use_qkv_bias = true,
                        bool            use_out_bias = true,
                        TensorOperation mask         = TensorOperation::MaskDisabled);

    ~LlamaAttentionLayer() = default;

    Variable* operator()(Variable* x, Variable* cache_k, Variable* cache_v);

    void BeforeForward(DDim batch_size_dim, DDim seq_len_dim);

    void LoadParam(const T* gamma_ptr,
                   const T* beta_ptr,
                   const T* qkv_weight_ptr,
                   const T* qkv_bias_ptr,
                   const T* out_weight_ptr,
                   const T* out_bias_ptr);

    int64_t seq_len_;
    int64_t hidden_dim_;
    int64_t num_heads_;
    float   scale_;

    bool has_residual_;
    bool use_qkv_bias_;
    bool use_out_bias_;

    TensorOperation mask_;

    std::unique_ptr<ReshapeOp<T>>      reshape_in_op_;
    std::unique_ptr<LayerNormLayer<T>> pre_layer_norm_;
    std::unique_ptr<LinearLayer<T>>    qkv_in_proj_;
    std::unique_ptr<SplitOp<T>>        split_op_;
    std::unique_ptr<ReshapeOp<T>>      reshape_q_op_;
    std::unique_ptr<ReshapeOp<T>>      reshape_k_op_;
    std::unique_ptr<ReshapeOp<T>>      reshape_v_op_;
    // std::unique_ptr<RoPEOp<T>>                 rope_op_;
    std::unique_ptr<ConcatOp<T>>               concat_k_op_;
    std::unique_ptr<ConcatOp<T>>               concat_v_op_;
    std::unique_ptr<BmmSoftmaxBmmPermuteOp<T>> bmm_softmax_bmm_permute_op_;
    std::unique_ptr<ReshapeOp<T>>              reshape_attn_op_;
    std::unique_ptr<LinearLayer<T>>            out_proj_;
};
}  // namespace lightinfer