#pragma once

#include "flashck/core/profiling/codegen/codegen_base.h"

namespace flashck {

class FmhaProblem: public ProblemDescBase {

    DataType                 dtype_;
    FmhaOperationMode        operation_mode_;  // kIsGroupMode_
    FmhaOperationKind        kind_;
    GenericAttentionMaskEnum mask_type_;
    std::array<int64_t, 2>   window_size_;
    BiasEnum                 bias_enum_;

    bool is_static_quant_;

    int64_t batch_size_;

    int64_t q_seq_len_;  // if group-mode, means the average value of seqlen_q
    int64_t q_max_seq_len_;
    int64_t kv_seq_len_;

    int64_t q_num_heads_;   // num of head, for q
    int64_t kv_num_heads_;  // num of head, for k,v;if not equal to num_heads, then this is GQA/MQA case

    int64_t qk_head_dim_;
    int64_t v_head_dim_;

    int64_t paged_block_size_;

    RopeEnum rope_type_;
    int64_t  rotary_dim_;

    int64_t num_splits_;
};
}  // namespace flashck