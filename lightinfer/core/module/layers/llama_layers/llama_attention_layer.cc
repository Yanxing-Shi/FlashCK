#include "lightinfer/core/module/layers/llama_layers/llama_attention_layer.h"

namespace lightinfer {

template<typename T>
LlamaAttentionLayer<T>::LlamaAttentionLayer(int64_t         seq_len,
                                            int64_t         hidden_dim,
                                            int64_t         num_heads,
                                            float           scale,
                                            bool            has_residual,
                                            bool            use_qkv_bias,
                                            bool            use_out_bias,
                                            TensorOperation mask):
    Layer("LlamaAttentionLayer"),
    seq_len_(seq_len),
    hidden_dim_(hidden_dim),
    num_heads_(num_heads),
    scale_(scale),
    has_residual_(has_residual),
    use_qkv_bias_(use_qkv_bias),
    use_out_bias_(use_out_bias),
    mask_(mask)
{
    LI_ENFORCE_EQ(hidden_dim % num_heads,
                  0,
                  Unavailable("hidden_dim {} should be divisible by num_heads {}", hidden_dim, num_heads));

    if (scale == 1.0f) {
        scale_ = sqrt(1.0 / num_heads);
    }

    reshape_in_op_ = std::make_unique<ReshapeOp<T>>();

    pre_layer_norm_ = std::make_unique<LayerNormLayer<T>>(Shape({hidden_dim_}));

    qkv_in_proj_ = std::make_unique<LinearLayer<T>>(
        hidden_dim_, hidden_dim_ * 3, true, "permute", Shape({DDim({1, seq_len}), DDim(3), DDim(num_heads)}));

    split_op_     = std::make_unique<SplitOp<T>>();
    reshape_q_op_ = std::make_unique<ReshapeOp<T>>();
    reshape_k_op_ = std::make_unique<ReshapeOp<T>>();
    reshape_v_op_ = std::make_unique<ReshapeOp<T>>();

    // rope_op_ = new RoPEOp<T>();

    concat_k_op_ = std::make_unique<ConcatOp<T>>();
    concat_v_op_ = std::make_unique<ConcatOp<T>>();

    bmm_softmax_bmm_permute_op_ = std::make_unique<BmmSoftmaxBmmPermuteOp<T>>(Shape({num_heads_}), mask_, scale_);

    reshape_attn_op_ = std::make_unique<ReshapeOp<T>>();

    if (has_residual_) {
        out_proj_ = std::make_unique<LinearLayer<T>>(hidden_dim_, hidden_dim_, true, "add");
    }

    this->context_ptr_->ExitLayer();
}

template<typename T>
Variable* LlamaAttentionLayer<T>::operator()(Variable* x, Variable* cache_k, Variable* cache_v)
{
    SetInputs({x, cache_k, cache_v});

    DDim batch_size_dim = x->GetShape().GetDim(0);
    DDim seq_len_dim    = x->GetShape().GetDim(1);
    DDim hidden_dim_dim = x->GetShape().GetDim(2);

    Variable* reshape_x = (*reshape_in_op_)(x, {-1, hidden_dim_dim});

    Variable* residual = reshape_x;

    Variable* norm_out = (*pre_layer_norm_)(reshape_x);

    Variable* qkv = (*qkv_in_proj_)(norm_out);

    std::vector<Variable*> qkv_split = (*split_op_)(qkv, {1} /* split size*/, 0 /*axis*/);

    Variable* q_var = qkv_split[0];  // [1, B, num_heads, seqlen, head_dim]
    Variable* k_var = qkv_split[1];  // [1, B, num_heads, seqlen, head_dim]
    Variable* v_var = qkv_split[2];  // [1, B, num_heads, seqlen, head_dim]
    VLOG(1) << "q_var shape: " << q_var->GetShape().ToString();
    VLOG(1) << "k_var shape: " << k_var->GetShape().ToString();
    VLOG(1) << "v_var shape: " << v_var->GetShape().ToString();

    DDim per_head_dim_dim = q_var->GetShape().GetDim(4);

    Variable* q_in = (*reshape_q_op_)(q_var, {-1, seq_len_dim, per_head_dim_dim});  // [B * num_heads, seqlen, head_dim]
    Variable* k_in = (*reshape_k_op_)(k_var, {-1, seq_len_dim, per_head_dim_dim});  // [B * num_heads, seqlen, head_dim]
    Variable* v_in = (*reshape_v_op_)(v_var, {-1, seq_len_dim, per_head_dim_dim});  // [B * num_heads, seqlen, head_dim]

    // input(q / k / v) : (B * num_heads, seqlen, head_dim)
    // attn = (B, S, H) * (B, S, H) = (B, S, S) #RCR
    // softmax on dim - 1(B, S, S)
    // attn @v : (B, S, S) * (B, S, H) = (B, S, H) #RRR
    // reshape : (B, num_head, seqlen, head_dim)
    // permute : (B, Seqlen, num_heads, head_dim)
    VLOG(1) << "q_in shape: " << q_in->GetShape().ToString();
    VLOG(1) << "k_in shape: " << k_in->GetShape().ToString();
    VLOG(1) << "v_in shape: " << v_in->GetShape().ToString();

    Variable* cache_k_out = (*concat_k_op_)({k_in, cache_k}, 1);
    Variable* cache_v_out = (*concat_v_op_)({v_in, cache_v}, 1);

    VLOG(1) << "cache_k shape: " << cache_k->GetShape().ToString();
    VLOG(1) << "cache_v shape: " << cache_v->GetShape().ToString();
    VLOG(1) << "cache_k_out shape: " << cache_k_out->GetShape().ToString();
    VLOG(1) << "cache_v_out shape: " << cache_v_out->GetShape().ToString();

    Variable* bmm_out = (*bmm_softmax_bmm_permute_op_)(q_in, cache_k_out, nullptr, cache_v_out);
    VLOG(1) << "attn_out shape: " << bmm_out->GetShape().ToString();
    Variable* reshape_bmm_out = (*reshape_attn_op_)(bmm_out, {batch_size_dim * seq_len_dim, -1});
    VLOG(1) << "reshape_attn_out shape: " << reshape_bmm_out->GetShape().ToString();
    Variable* residual_out = has_residual_ ? (*out_proj_)(reshape_bmm_out, residual) : reshape_bmm_out;

    SetOutputs({residual_out});
    return residual_out;
}

template<typename T>
void LlamaAttentionLayer<T>::BeforeForward(DDim batch_size_dim, DDim seq_len_dim)
{
    qkv_in_proj_->BeforeForward({seq_len_dim});
    reshape_in_op_->UpdateShape({-1, hidden_dim_});
    reshape_q_op_->UpdateShape({-1, seq_len_dim, hidden_dim_ / num_heads_});
    reshape_k_op_->UpdateShape({-1, seq_len_dim, hidden_dim_ / num_heads_});
    reshape_v_op_->UpdateShape({-1, seq_len_dim, hidden_dim_ / num_heads_});
    reshape_attn_op_->UpdateShape({batch_size_dim * seq_len_dim, -1});
}

template<typename T>
void LlamaAttentionLayer<T>::LoadParam(const T* gamma_ptr,
                                       const T* beta_ptr,
                                       const T* qkv_weight_ptr,
                                       const T* qkv_bias_ptr,
                                       const T* out_weight_ptr,
                                       const T* out_bias_ptr)
{
    pre_layer_norm_->LoadParam(gamma_ptr, beta_ptr);
    qkv_in_proj_->LoadParam(qkv_weight_ptr, qkv_bias_ptr);
    if (has_residual_) {
        out_proj_->LoadParam(out_weight_ptr, out_bias_ptr);
    }
}

template class LlamaAttentionLayer<float>;
template class LlamaAttentionLayer<_Float16>;

}  // namespace lightinfer