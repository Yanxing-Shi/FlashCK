#include "flashck/core/module/layers/attention_layers/memory_efficient_attention_decoder_layer.h"

namespace flashck {

template<typename T>
MemoryEfficientAttentionDecoderLayer<T>::MemoryEfficientAttentionDecoderLayer(FmhaOperationMode        mode,
                                                                              int64_t                  q_num_heads,
                                                                              int64_t                  kv_num_heads,
                                                                              int64_t                  qk_head_dim,
                                                                              int64_t                  v_head_dim,
                                                                              float                    scale,
                                                                              int64_t                  rotary_dim,
                                                                              RopeEnum                 rope_enum,
                                                                              BiasEnum                 bias_enum,
                                                                              std::array<int64_t, 2>   window_size,
                                                                              GenericAttentionMaskEnum mask_enum,
                                                                              int64_t                  paged_block_size,
                                                                              bool    use_cache_batch_idx,
                                                                              int64_t num_splits):
    Layer("MemoryEfficientAttentionDecoderLayer"),
    mode_(mode),
    q_num_heads_(q_num_heads),
    kv_num_heads_(kv_num_heads),
    qk_head_dim_(qk_head_dim),
    v_head_dim_(v_head_dim),
    scale_(scale),
    rotary_dim_(rotary_dim),
    rope_enum_(rope_enum),
    bias_enum_(bias_enum),
    window_size_(window_size),
    mask_enum_(mask_enum),
    paged_block_size_(paged_block_size),
    use_cache_batch_idx_(use_cache_batch_idx),
    num_splits_(num_splits)

{
    // fmha_append_op
    fmha_fwd_appendkv_op_ = std::make_unique<FmhaFwdAppendKVOp<T>>("fmha_fwd_appendkv_op",
                                                                   mode_,
                                                                   q_num_heads_,
                                                                   qk_head_dim_,
                                                                   kv_num_heads_,
                                                                   v_head_dim_,
                                                                   rotary_dim_,
                                                                   rope_enum_,
                                                                   paged_block_size_,
                                                                   use_cache_batch_idx_);

    // fmha_split_kv_op
    fmha_fwd_splitkv_op_ = std::make_unique<FmhaFwdSplitKVOp<T>>("fmha_fwd_splitkv_op",
                                                                 mode_,
                                                                 q_num_heads_,
                                                                 qk_head_dim_,
                                                                 kv_num_heads_,
                                                                 v_head_dim_,
                                                                 scale_,
                                                                 bias_enum_,
                                                                 window_size_,
                                                                 mask_enum_,
                                                                 paged_block_size_,
                                                                 use_cache_batch_idx_,
                                                                 num_splits_);

    // fmha_split_kv_combine_op
    fmha_fwd_splitkv_combine_op_ = std::make_unique<FmhaFwdSplitKVCombineOp<T>>(
        "fmha_fwd_splitkv_combine_op", mode_, q_num_heads_, v_head_dim_, num_splits_);

    this->context_ptr_->ExitLayer();
}

template<typename T>
Variable* MemoryEfficientAttentionDecoderLayer<T>::operator()(Variable* q,
                                                              Variable* cache_k,
                                                              Variable* cache_v,
                                                              Variable* k,
                                                              Variable* v,
                                                              Variable* bias,
                                                              Variable* rotary_cos,
                                                              Variable* rotary_sin,
                                                              Variable* cache_batch_idx,
                                                              Variable* block_table,
                                                              Variable* seqlen_k,
                                                              Variable* seqstart_q,
                                                              Variable* seqstart_k)
{

    std::vector<Variable*> inputs;
    for (auto& input : {q,
                        cache_k,
                        cache_v,
                        k,
                        v,
                        bias,
                        rotary_cos,
                        rotary_sin,
                        cache_batch_idx,
                        block_table,
                        seqlen_k,
                        seqstart_q,
                        seqstart_k}) {
        if (input != nullptr) {
            inputs.push_back(input);
        }
    }

    SetInputs(inputs);

    // fmha_append_op
    std::vector<Variable*> appendkv_out = (*fmha_fwd_appendkv_op_)(
        q, cache_k, cache_v, k, v, block_table, cache_batch_idx, rotary_cos, rotary_sin, seqlen_k);

    // fmha_split_kv_op
    std::vector<Variable*> fmha_split_kv_out = (*fmha_fwd_splitkv_op_)(appendkv_out[0],
                                                                       appendkv_out[1],
                                                                       appendkv_out[2],
                                                                       bias,
                                                                       block_table,
                                                                       cache_batch_idx,
                                                                       seqlen_k,
                                                                       seqstart_q,
                                                                       seqstart_k);

    // fmha_split_kv_combine_op
    Variable* fmha_split_kv_combine_out = (*fmha_fwd_splitkv_combine_op_)(fmha_split_kv_out[0], fmha_split_kv_out[1]);
    SetOutputs({fmha_split_kv_combine_out});
    VLOG(1) << "fmha_split_kv_combine_out shape: " << fmha_split_kv_combine_out->GetShape().ToString();
    return fmha_split_kv_combine_out;

    VLOG(1) << "q_var shape: " << q->GetShape().ToString();
    VLOG(1) << "cache_k_var shape: " << cache_k->GetShape().ToString();
    VLOG(1) << "cache_v_var shape: " << cache_v->GetShape().ToString();
    VLOG(1) << "k_var shape: " << k->GetShape().ToString();
    VLOG(1) << "v_var shape: " << v->GetShape().ToString();
    VLOG(1) << "bias_var shape: " << bias->GetShape().ToString();
    VLOG(1) << "rotary_cos_var shape: " << rotary_cos->GetShape().ToString();
    VLOG(1) << "rotary_sin_var shape: " << rotary_sin->GetShape().ToString();
    VLOG(1) << "cache_batch_idx_var shape: " << cache_batch_idx->GetShape().ToString();
    VLOG(1) << "block_table_var shape: " << block_table->GetShape().ToString();
}

template class MemoryEfficientAttentionDecoderLayer<ushort>;
template class MemoryEfficientAttentionDecoderLayer<_Float16>;

}  // namespace flashck