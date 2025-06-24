

#pragma once

#include "flashck/core/graph/layer.h"
#include "flashck/core/graph/node.h"

#include "flashck/core/module/operations/fmha_ops/fmha_fwd_appendkv_op.h"
#include "flashck/core/module/operations/fmha_ops/fmha_fwd_splitkv_combine_op.h"
#include "flashck/core/module/operations/fmha_ops/fmha_fwd_splitkv_op.h"

namespace flashck {

template<typename T>
class MemoryEfficientAttentionDecoderLayer: public Layer {
public:
    MemoryEfficientAttentionDecoderLayer(FmhaOperationMode        mode,
                                         int64_t                  q_num_heads,
                                         int64_t                  kv_num_heads,
                                         int64_t                  qk_head_dim,
                                         int64_t                  v_head_dim,
                                         float                    scale            = 1.0f,
                                         int64_t                  rotary_dim       = -1,
                                         RopeEnum                 rope_enum        = RopeEnum::NONE,
                                         BiasEnum                 bias_enum        = BiasEnum::NO_BIAS,
                                         std::array<int64_t, 2>   window_size      = {-1, -1},
                                         GenericAttentionMaskEnum mask_enum        = GenericAttentionMaskEnum::NO_MASK,
                                         int64_t                  paged_block_size = -1,
                                         bool                     use_cache_batch_idx = false,
                                         int64_t                  num_splits          = 1);

    ~MemoryEfficientAttentionDecoderLayer() = default;

    Variable* operator()(Variable* q,        // [batch_size, q_seq_len, q_num_heads, qk_head_dim]
                         Variable* cache_k,  // [batch_size, kv_seq_len, kv_num_heads, qk_head_dim] or [num_blocks,
                                             // page_block_size, kv_num_heads, qk_head_dim]
                         Variable* cache_v,  // [batch_size, kv_seq_len, kv_num_heads, v_head_dim] or [num_blocks,
                                             // page_block_size, kv_num_heads, v_head_dim]
                         Variable* k,        // [batch_size, new_kv_seq_len, kv_num_heads, qk_head_dim]
                         Variable* v,        // [batch_size, new_kv_seq_len, kv_num_heads, v_head_dim]
                         Variable* bias,     // element-wise bias: [batch_size, q_num_heads, q_seq_len, kv_seq_len]
                                             // alibi_slopes: [batch_size, q_num_heads]
                         Variable* rotary_cos      = nullptr,  // [max(q_seqlen, kv_seq_len)*2, rotary_dim / 2]
                         Variable* rotary_sin      = nullptr,  // [max(q_seqlen, kv_seq_len)*2, rotary_dim / 2]
                         Variable* cache_batch_idx = nullptr,  // [batch_size]
                         Variable* block_table     = nullptr,  // [batch_size, max_num_page_blocks / batch_size]
                         Variable* seqlen_k        = nullptr,  // [batch_size]
                         Variable* seqstart_q      = nullptr,  // [batch_size+1]
                         Variable* seqstart_k      = nullptr   // [batch_size+1]
    );

    FmhaOperationMode mode_;
    int64_t           q_num_heads_;
    int64_t           kv_num_heads_;
    int64_t           qk_head_dim_;
    int64_t           v_head_dim_;
    float             scale_;
    int64_t           new_kv_seq_len_;
    int64_t           rotary_dim_;
    RopeEnum          rope_enum_;

    BiasEnum                 bias_enum_;
    std::array<int64_t, 2>   window_size_;
    GenericAttentionMaskEnum mask_enum_;

    int64_t paged_block_size_;
    bool    use_cache_batch_idx_;

    int64_t num_splits_;

    std::unique_ptr<FmhaFwdAppendKVOp<T>>       fmha_fwd_appendkv_op_;
    std::unique_ptr<FmhaFwdSplitKVOp<T>>        fmha_fwd_splitkv_op_;
    std::unique_ptr<FmhaFwdSplitKVCombineOp<T>> fmha_fwd_splitkv_combine_op_;
};
}  // namespace flashck
