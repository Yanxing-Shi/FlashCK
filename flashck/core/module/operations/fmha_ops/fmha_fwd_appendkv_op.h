#pragma once

#include "flashck/core/module/operations/fmha_ops/fmha_common_op.h"

namespace flashck {
template<typename T>
class FmhaFwdAppendKVOp: public FmhaCommonOp<T, FmhaFwdAppendKVOp<T>> {
public:
    FmhaFwdAppendKVOp(std::string       op_name,
                      FmhaOperationMode op_mode,
                      int64_t           q_num_heads,
                      int64_t           qk_head_dim,
                      int64_t           kv_num_heads,
                      int64_t           v_head_dim,
                      int64_t           rotary_dim,
                      RopeEnum          rope_enum,
                      int64_t           paged_block_size,
                      bool              use_cache_batch_idx);

    std::tuple<Shape, Shape, Shape> InferShape(Variable* q, Variable* cache_k, Variable* k, Variable* v);

    FmhaProblem DefineProblemImpl(const std::vector<int64_t>& inverse_res);

    std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> ExtractDimsImpl();

    void SanityCheck(Variable* q,
                     Variable* cache_k,
                     Variable* cache_v,
                     Variable* k,
                     Variable* v,
                     Variable* block_table,
                     Variable* cache_batch_idx,
                     Variable* rotary_cos,
                     Variable* rotary_sin,
                     Variable* cache_seqlen_k = nullptr);

    std::vector<Variable*>
    operator()(Variable* q,                        // [batch_size, q_seq_len, q_num_heads, qk_head_dim]
               Variable* cache_k,                  // [max_num_page_blocks, paged_block_size, kv_num_heads, qk_head_dim]
                                                   // or [batch_size, kv_seq_len, kv_num_heads, qk_head_dim]
               Variable* cache_v,                  // [max_num_page_blocks, paged_block_size, kv_num_heads, v_head_dim]
                                                   // or [batch_size, kv_seq_len, kv_num_heads, v_head_dim]
               Variable* k,                        // [batch_size, new_kv_seq_len, kv_num_heads, qk_head_dim]
               Variable* v,                        // [batch_size, new_kv_seq_len, kv_num_heads, v_head_dim]
               Variable* block_table,              // [batch_size, max_num_page_blocks / batch_size]
               Variable* cache_batch_idx,          // [batch_size]
               Variable* rotary_cos,               // [max(q_seqlen, kv_seq_len)*2, rotary_dim / 2]
               Variable* rotary_sin,               // [max(q_seqlen, kv_seq_len)*2, rotary_dim / 2]
               Variable* cache_seqlen_k = nullptr  // [batch_size]
    );

    std::function<std::vector<std::string>(const std::string&)> GenBuildCmd();

    ~FmhaFwdAppendKVOp() = default;

    void ForwardImpl();
};

}  // namespace flashck