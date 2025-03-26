#pragma once

#include "lightinfer/core/module/operations/fmha_ops/fmha_common_op.h"

/*
  There are 2 modes for using this function.
  (Mode BMHK) With all the heads having the same seqlen
  (Mode 1MHK) `batch=1` with all tokens across batches concatenated
*/

namespace lightinfer {
template<typename T>
class FmhaFwdOp: public FmhaCommonOp<T, FmhaFwdOp<T>> {
public:
    FmhaFwdOp(std::string              op_name,
              FmhaOperationMode        op_mode,
              int64_t                  q_num_heads,
              int64_t                  qk_head_dim,
              int64_t                  kv_num_heads,
              int64_t                  v_head_dim,
              float                    qk_scale,
              BiasEnum                 bias_enum,
              std::array<int64_t, 2>   window_size,
              GenericAttentionMaskEnum mask,
              bool                     is_packed_qkv = true);

    ~FmhaFwdOp() = default;

    FmhaProblem DefineProblemImpl(const std::vector<int64_t>& inverse_res);

    std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> ExtractDimsImpl();

    void SanityCheck(Variable* q,
                     Variable* k,
                     Variable* v,
                     Variable* bias,
                     Variable* seqstart_q,
                     Variable* seqstart_k,
                     Variable* seqlen_k);

    Shape InferShape(Variable* q, Variable* v);

    Variable* operator()(Variable* q,                     // [batch_size, q_seq_len, q_num_heads, qk_head_dim]
                         Variable* k,                     // [batch_size, kv_seq_len, kv_num_heads, qk_head_dim]
                         Variable* v,                     // [batch_size, kv_seq_len, kv_num_heads, v_head_dim]
                         Variable* bias = nullptr,        // element-wise bias: [batch_size, q_num_heads, q_seq_len,
                                                          // kv_seq_len] alibi_slopes: [batch_size, q_num_heads]
                         Variable* seqstart_q = nullptr,  // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains
                                                          // the position of the first query token for batch $b
                         Variable* seqstart_k = nullptr,  // (Mode 1MHK only) [b+1]: cu_seqlen_k[b] contains the
                                                          // position of the first key token for batch $b
                         Variable* seqlen_k = nullptr     // (Mode 1MHK only) [b]: cu_seqlen_k[b] contains the
                                                          // length of the key sequence for batch $b
    );

    std::function<std::vector<std::string>(const std::string&)> GenBuildCmd();

    void ForwardImpl();

    bool is_packed_qkv_;
};

}  // namespace lightinfer