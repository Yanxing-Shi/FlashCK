#pragma once

#include "core/graph/layer.h"
#include "core/graph/node.h"

#include "core/module/operations/fmha_ops/fmha_fwd_op.h"

namespace flashck {

template<typename T>
class AttentionLayer: public Layer {
public:
    AttentionLayer(FmhaOperationMode        mode,
                    int64_t                  q_num_heads,
                    int64_t                  kv_num_heads,
                    int64_t                  qk_head_dim,
                    int64_t                  v_head_dim,
                    float                    scale         = 1.0f,
                    BiasEnum                 bias_enum     = BiasEnum::NO_BIAS,
                    std::array<int64_t, 2>   window_size   = {-1, -1},
                    GenericAttentionMaskEnum mask_enum     = GenericAttentionMaskEnum::NO_MASK);

    ~AttentionLayer() = default;

    Variable* operator()(Variable* q,                     // [batch_size, q_seq_len, q_num_heads, qk_head_dim]
                         Variable* k,                     // [batch_size, kv_seq_len, kv_num_heads, qk_head_dim]
                         Variable* value,                 // [batch_size, kv_seq_len, kv_num_heads, v_head_dim]
                         Variable* bias = nullptr,        // element-wise bias: [batch_size, q_num_heads, q_seq_len,
                                                          // kv_seq_len] alibi_slopes: [batch_size, q_num_heads]
                         Variable* seqstart_q = nullptr,  // [batch_size+1]
                         Variable* seqstart_k = nullptr,  // [batch_size+1]
                         Variable* seqlen_k   = nullptr   // [batch_size]
    );

    FmhaOperationMode        mode_;
    int64_t                  q_num_heads_;
    int64_t                  kv_num_heads_;
    int64_t                  qk_head_dim_;
    int64_t                  v_head_dim_;
    float                    scale_;
    BiasEnum                 bias_enum_;
    std::array<int64_t, 2>   window_size_;
    GenericAttentionMaskEnum mask_enum_;
    bool                     is_qkv_packed_;

    std::unique_ptr<FmhaFwdOp<T>> fmha_fwd_op_;
};
}  // namespace lightinfer