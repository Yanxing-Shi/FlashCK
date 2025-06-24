#pragma once

#include "flashck/core/module/operations/fmha_ops/fmha_common_op.h"

namespace flashck {

template<typename T>
class FmhaFwdSplitKVCombineOp: public FmhaCommonOp<T, FmhaFwdSplitKVCombineOp<T>> {
public:
    FmhaFwdSplitKVCombineOp(
        std::string op_name, FmhaOperationMode op_mode, int64_t q_num_heads, int64_t v_head_dim, int64_t num_splits);

    FmhaProblem DefineProblemImpl(const std::vector<int64_t>& inverse_res);

    std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> ExtractDimsImpl();

    void SanityCheck(Variable* out_acc, Variable* lse_acc, Variable* seqstart_q);

    Shape InferShape(Variable* out_acc, Variable* lse_acc);

    Variable* operator()(Variable* out_acc,              // [num_splits, batch_size, q_seq_len, q_num_heads, v_head_dim]
                         Variable* lse_acc,              // [num_splits, batch_size, q_seq_len, q_num_heads]
                         Variable* seqstart_q = nullptr  // [batch_size+1]
    );

    std::function<std::vector<std::string>(const std::string&)> GenBuildCmd();

    ~FmhaFwdSplitKVCombineOp() = default;

    void ForwardImpl();
};

}  // namespace flashck