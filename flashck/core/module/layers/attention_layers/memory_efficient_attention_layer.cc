#include "flashck/core/module/layers/attention_layers/memory_efficient_attention_layer.h"

namespace flashck {

template<typename T>
MemoryEfficientAttentionLayer<T>::MemoryEfficientAttentionLayer(FmhaOperationMode        mode,
                                                                int64_t                  q_num_heads,
                                                                int64_t                  kv_num_heads,
                                                                int64_t                  qk_head_dim,
                                                                int64_t                  v_head_dim,
                                                                float                    scale,
                                                                BiasEnum                 bias_enum,
                                                                std::array<int64_t, 2>   window_size,
                                                                GenericAttentionMaskEnum mask_enum,
                                                                bool                     is_qkv_packed):
    Layer("MemoryEfficientAttentionLayer"),
    mode_(mode),
    q_num_heads_(q_num_heads),
    kv_num_heads_(kv_num_heads),
    qk_head_dim_(qk_head_dim),
    v_head_dim_(v_head_dim),
    scale_(scale),
    bias_enum_(bias_enum),
    window_size_(window_size),
    mask_enum_(mask_enum),
    is_qkv_packed_(is_qkv_packed)
{
    fmha_fwd_op_ = std::make_unique<FmhaFwdOp<T>>("fmha_fwd_op",
                                                  mode_,
                                                  q_num_heads_,
                                                  qk_head_dim_,
                                                  kv_num_heads_,
                                                  v_head_dim_,
                                                  scale_,
                                                  bias_enum_,
                                                  window_size_,
                                                  mask_enum_,
                                                  is_qkv_packed);

    this->context_ptr_->ExitLayer();
}

template<typename T>
Variable* MemoryEfficientAttentionLayer<T>::operator()(Variable* q,
                                                       Variable* k,
                                                       Variable* v,
                                                       Variable* bias,
                                                       Variable* seqstart_q,
                                                       Variable* seqstart_k,
                                                       Variable* seqlen_k)
{
    if (mode_ == FmhaOperationMode::Batch) {
        SetInputs({q, k, v, bias});
    }
    else if (mode_ == FmhaOperationMode::Group) {
        SetInputs({q, k, v, bias, seqstart_q, seqstart_k, seqlen_k});
    }
    else {
        FC_THROW(Unimplemented("fmha only supports Batch && group"));
    }

    Variable* fmha_out =
        (*fmha_fwd_op_)(q, k, v, bias, seqstart_q, seqstart_k, seqlen_k);  // [B, seqlen, num_heads, head_dim]

    VLOG(1) << "q_var shape: " << q->GetShape().ToString();
    VLOG(1) << "k_var shape: " << k->GetShape().ToString();
    VLOG(1) << "v_var shape: " << v->GetShape().ToString();
    VLOG(1) << "fmha_out shape: " << fmha_out->GetShape().ToString();

    SetOutputs({fmha_out});
    return fmha_out;
}

template class MemoryEfficientAttentionLayer<ushort>;
template class MemoryEfficientAttentionLayer<_Float16>;

}  // namespace flashck