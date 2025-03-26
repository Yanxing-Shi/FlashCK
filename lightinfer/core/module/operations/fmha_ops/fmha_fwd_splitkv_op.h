#pragma once

#include <tuple>

#include "lightinfer/core/module/operations/fmha_ops/fmha_common_op.h"

namespace lightinfer {

template<typename T>
class FmhaFwdSplitKVOp: public FmhaCommonOp<T, FmhaFwdSplitKVOp<T>> {
public:
    FmhaFwdSplitKVOp(std::string              op_name,
                     FmhaOperationMode        op_mode,
                     int64_t                  q_num_heads,
                     int64_t                  qk_head_dim,
                     int64_t                  kv_num_heads,
                     int64_t                  v_head_dim,
                     float                    qk_scale,
                     BiasEnum                 bias_enum,
                     std::array<int64_t, 2>   window_size,
                     GenericAttentionMaskEnum mask_enum,
                     int64_t                  paged_block_size,
                     bool                     use_cache_batch_idx,
                     int64_t                  num_splits);

    FmhaProblem DefineProblemImpl(const std::vector<int64_t>& inverse_res);

    std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> ExtractDimsImpl();

    void SanityCheck(Variable* q,
                     Variable* k,
                     Variable* v,
                     Variable* bias,
                     Variable* block_table,
                     Variable* cache_batch_idx,
                     Variable* seqlen_k   = nullptr,
                     Variable* seqstart_q = nullptr,
                     Variable* seqstart_k = nullptr);

    std::tuple<Shape, Shape> InferShape(Variable* q, Variable* v);

    std::vector<Variable*> operator()(Variable* q,            // [batch_size, q_seq_len, q_num_heads, qk_head_dim]
                                      Variable* k,            // [batch_size, kv_seq_len, kv_num_heads, qk_head_dim]
                                      Variable* v,            // [batch_size, kv_seq_len, kv_num_heads, v_head_dim]
                                      Variable* bias,         // element-wise bias: [batch_size, q_num_heads, q_seq_len,
                                                              // kv_seq_len] alibi_slopes: [batch_size, q_num_heads]
                                      Variable* block_table,  // [batch_size, max_num_page_blocks/batch_size]
                                      Variable* cache_batch_idx,       // [batch_size]
                                      Variable* seqlen_k   = nullptr,  // [batch_size]
                                      Variable* seqstart_q = nullptr,  // [batch_size+1]
                                      Variable* seqstart_k = nullptr   // [batch_size+1]
    );

    std::function<std::vector<std::string>(const std::string&)> GenBuildCmd();

    ~FmhaFwdSplitKVOp() = default;

    void ForwardImpl();
};

}  // namespace lightinfer