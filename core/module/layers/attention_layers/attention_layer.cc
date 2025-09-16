#include "core/module/layers/attention_layers/memory_efficient_attention_layer.h"

namespace flashck {

template<typename T>
AttentionLayer<T>::AttentionLayer(FmhaOperationMode        mode,
                                int64_t                  q_num_heads,
                                int64_t                  kv_num_heads,
                                int64_t                  qk_head_dim,
                                int64_t                  v_head_dim,
                                float                    scale,
                                BiasEnum                 bias_enum,
                                std::array<int64_t, 2>   window_size,
                                GenericAttentionMaskEnum mask_enum):
    Layer("AttentionLayer"),
    mode_(mode),
    q_num_heads_(q_num_heads),
    kv_num_heads_(kv_num_heads),
    qk_head_dim_(qk_head_dim),
    v_head_dim_(v_head_dim),
    scale_(scale),
    bias_enum_(bias_enum),
    window_size_(window_size),
    mask_enum_(mask_enum)
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
Variable* AttentionLayer<T>::operator()(Variable* q,
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
        FC_THROW(Unimplemented("fmha only supports Batch && group mode"));
    }

    Variable* fmha_out =
        (*fmha_fwd_op_)(q, k, v, bias, seqstart_q, seqstart_k, seqlen_k);  // [B, seqlen, num_heads, head_dim]

    SetOutputs({fmha_out});
    return fmha_out;
}

template class AttentionLayer<ushort>;
template class AttentionLayer<_Float16>;

}  // namespace lightinfer