#pragma once

#include "flashck/core/profiler/library.h"
#include "flashck/core/utils/dtype.h"

#include "flashck/core/profiler/fmha_fwd_operation.h"

namespace flashck {

struct FmhaSplitKVCompbineTileDesc {
    int64_t bm0_;
    int64_t bn1_;  // tile size along v head_dim

    std::string GetConfigName();
};

struct FmhaFwdSplitKVCombineOperation {
    FmhaOperationKind operation_kind_;
    TensorOperation   epilogue_op_;

    FmhaOperationMode           operation_mode_;
    DataType                    dtype_;
    FmhaSplitKVCompbineTileDesc tile_desc_;
    int64_t                     hdim_;
    bool                        is_static_quant_;

    bool is_pad_q_seq_len_;   // padding for seqlen_q
    bool is_pad_v_head_dim_;  // paddding for hdim_v

    int block_per_cu_ = -1;  // overwrite occupancy if not -1

    int64_t log_max_splits_;

    std::string GetPadName();

    std::string GetPipelineConfigName();

    std::string GetConfigName();

    std::string Emit();
};

}  // namespace flashck