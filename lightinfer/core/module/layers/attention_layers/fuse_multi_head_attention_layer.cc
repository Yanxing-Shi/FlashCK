#include "lightinfer/core/module/layers/attention_layers/fuse_multi_head_attention_layer.h"

namespace lightinfer {

template<typename T>
FuseMultiHeadAttentionLayer<T>::FuseMultiHeadAttentionLayer(int64_t                  q_num_heads,
                                                            int64_t                  kv_num_heads,
                                                            int64_t                  qk_head_dim,
                                                            int64_t                  v_head_dim,
                                                            float                    scale,
                                                            BiasEnum                 bias_enum,
                                                            std::array<int64_t, 2>   window_size,
                                                            GenericAttentionMaskEnum mask_enum,
                                                            LayerNormType            layer_norm_type,
                                                            float                    epsilon,
                                                            bool                     is_qkv_packed,
                                                            bool                     has_residual,
                                                            bool                     use_qkv_bias,
                                                            bool                     use_out_bias):
    Layer("FuseMultiHeadAttentionLayer"),
    q_num_heads_(q_num_heads),
    kv_num_heads_(kv_num_heads),
    qk_head_dim_(qk_head_dim),
    v_head_dim_(v_head_dim),
    scale_(scale),
    bias_enum_(bias_enum),
    window_size_(window_size),
    mask_enum_(mask_enum),
    layer_norm_type_(layer_norm_type),
    epsilon_(epsilon),
    is_qkv_packed_(is_qkv_packed),
    has_residual_(has_residual),
    use_qkv_bias_(use_qkv_bias),
    use_out_bias_(use_out_bias)
{
    if (is_qkv_packed && !(qk_head_dim_ == v_head_dim_ && q_num_heads_ == kv_num_heads_)) {
        LI_THROW(Unavailable("qkv packed"));
    }

    int64_t hidden_dim = q_num_heads_ * qk_head_dim_;

    if (layer_norm_type_ == LayerNormType::PreLayerNorm) {
        layer_norm_ = std::make_unique<LayerNormLayer<T>>(Shape({hidden_dim}), epsilon);
    }

    if (is_qkv_packed_) {
        qkv_in_proj_ = std::make_unique<LinearLayer<T>>(
            hidden_dim, hidden_dim * 3, false, true, "permute", Shape({3, q_num_heads_}));
    }
    else {
        q_in_proj_ = std::make_unique<LinearLayer<T>>(hidden_dim, hidden_dim, false, true);
        k_in_proj_ = std::make_unique<LinearLayer<T>>(hidden_dim, hidden_dim, false, true);
        v_in_proj_ = std::make_unique<LinearLayer<T>>(hidden_dim, hidden_dim, false, true);
    }

    fmha_fwd_ = std::make_unique<MemoryEfficientAttentionLayer<T>>(FmhaOperationMode::Batch,
                                                                   q_num_heads_,
                                                                   kv_num_heads_,
                                                                   qk_head_dim_,
                                                                   v_head_dim_,
                                                                   scale_,
                                                                   bias_enum_,
                                                                   window_size_,
                                                                   mask_enum_,
                                                                   is_qkv_packed_);

    if (has_residual_) {
        out_proj_ = std::make_unique<LinearLayer<T>>(hidden_dim, hidden_dim, false, true, "add");
    }

    if (layer_norm_type_ == LayerNormType::PostLayerNorm) {
        layer_norm_ = std::make_unique<LayerNormLayer<T>>(Shape({hidden_dim}), epsilon);
    }

    this->context_ptr_->ExitLayer();
}

template<typename T>
Variable* FuseMultiHeadAttentionLayer<T>::operator()(Variable* x)
{
    SetInputs({x});

    DDim batch_size_dim = x->GetShape().GetDim(0);
    DDim seq_len_dim    = x->GetShape().GetDim(1);
    DDim hidden_dim_dim = x->GetShape().GetDim(2);

    Variable* residual = x;

    Variable* layer_norm_out =
        layer_norm_type_ == LayerNormType::PreLayerNorm ? (*layer_norm_)(x) : x;  // [B, seqlen, hidden_dim]
    VLOG(1) << "layer_norm_out shape: " << layer_norm_out->GetShape().ToString();

    Variable* qkv_proj_out = nullptr;
    Variable* fmha_out     = nullptr;

    if (is_qkv_packed_) {
        qkv_proj_out = (*qkv_in_proj_)(layer_norm_out);  // [3, batch_size, q_seqlen, q_num_heads, v_head_dim]
        VLOG(1) << "qkv_proj_out shape: " << qkv_proj_out->GetShape().ToString();
        auto q_shape = Shape({batch_size_dim, seq_len_dim, DDim(q_num_heads_), hidden_dim_dim / DDim(q_num_heads_)});
        auto offset  = batch_size_dim.GetValues()[1] * seq_len_dim.GetValues()[1] * hidden_dim_dim.GetValues()[1];
        Variable* q_var =
            new Variable("q_var", qkv_proj_out, true);  // [batch_size, q_seqlen, q_num_heads, qk_head_dim]
        q_var->SetOffset(0, q_shape);
        Variable* k_var = new Variable("k_var", qkv_proj_out);  // [batch_size, kv_seqlen, kv_num_heads, qk_head_dim]
        k_var->SetOffset(offset, q_shape);
        Variable* v_var = new Variable("v_var", qkv_proj_out);  // [batch_size, kv_seqlen, kv_num_heads, v_head_dim]
        v_var->SetOffset(2 * offset, q_shape);
        fmha_out = (*fmha_fwd_)(q_var, k_var, v_var);  // [batch_size, q_seqlen, q_num_heads, v_head_dim]
    }
    else {
        Variable* q_proj_out = (*q_in_proj_)(layer_norm_out);  // [batch_size, q_seqlen, q_num_heads, qk_head_dim]
        Variable* k_proj_out = (*k_in_proj_)(layer_norm_out);  // [batch_size, kv_seqlen, kv_num_heads, qk_head_dim]
        Variable* v_proj_out = (*v_in_proj_)(layer_norm_out);  // [batch_size, kv_seqlen, kv_num_heads, v_head_dim]
        fmha_out = (*fmha_fwd_)(q_proj_out, k_proj_out, v_proj_out);  // [batch_size, q_seqlen, q_num_heads, v_head_dim]
    }

    fmha_out->SetShape(
        {batch_size_dim, seq_len_dim, hidden_dim_dim});  // [batch_size, q_seqlen, q_num_heads,
                                                         // v_head_dim]->[batch_size, q_seqlen, hidden_dim]
    Variable* residual_out = has_residual_ ? (*out_proj_)(fmha_out, residual) : fmha_out;
    VLOG(1) << "residual_out shape: " << residual_out->GetShape().ToString();

    Variable* attn_out = layer_norm_type_ == LayerNormType::PostLayerNorm ? (*layer_norm_)(residual_out) : residual_out;
    VLOG(1) << "attn_out shape: " << attn_out->GetShape().ToString();

    SetOutputs({attn_out});
    return attn_out;
}

template<typename T>
void FuseMultiHeadAttentionLayer<T>::LoadParam(const T* gamma_ptr,
                                               const T* beta_ptr,
                                               const T* qkv_weight_ptr,
                                               const T* qkv_bias_ptr,
                                               const T* out_weight_ptr,
                                               const T* out_bias_ptr)
{
    layer_norm_->LoadParam(gamma_ptr, beta_ptr);
    LI_ENFORCE_EQ(is_qkv_packed_, true, Unavailable("is_packed_qkv must true"));
    qkv_in_proj_->LoadParam(qkv_weight_ptr, qkv_bias_ptr);
    if (has_residual_) {
        out_proj_->LoadParam(out_weight_ptr, out_bias_ptr);
    }
}

template<typename T>
void FuseMultiHeadAttentionLayer<T>::LoadParam(const T* gamma_ptr,
                                               const T* beta_ptr,
                                               const T* q_weight_ptr,
                                               const T* q_bias_ptr,
                                               const T* k_weight_ptr,
                                               const T* k_bias_ptr,
                                               const T* v_weight_ptr,
                                               const T* v_bias_ptr,
                                               const T* out_weight_ptr,
                                               const T* out_bias_ptr)
{
    layer_norm_->LoadParam(gamma_ptr, beta_ptr);
    LI_ENFORCE_EQ(is_qkv_packed_, false, Unavailable("is_packed_qkv must false"));
    q_in_proj_->LoadParam(q_weight_ptr, q_bias_ptr);
    k_in_proj_->LoadParam(k_weight_ptr, k_bias_ptr);
    v_in_proj_->LoadParam(v_weight_ptr, v_bias_ptr);
    if (has_residual_) {
        out_proj_->LoadParam(out_weight_ptr, out_bias_ptr);
    }
}

template class FuseMultiHeadAttentionLayer<_Float16>;
template class FuseMultiHeadAttentionLayer<ushort>;

}  // namespace lightinfer