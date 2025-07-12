#pragma once

#include "flashck/core/profiling/library.h"
#include "flashck/core/utils/dtype.h"

namespace flashck {

struct FmhaAppendKVTileDesc {
    int64_t bs_;   // tile size along q seqlen
    int64_t bsk_;  // tile size along k seqlen
    int64_t bd_;   // tile size along qk gemm unroll
    int64_t bdv_;  // tile size along kv gemm unroll

    std::string GetInstanceName();
};

struct FmhaFwdAppendKVOperation {
    FmhaOperationKind kind_;
    TensorOperation   epilogue_op_;

    DataType             dtype_;
    FmhaAppendKVTileDesc tile_desc_;
    FmhaOperationMode    operation_mode_;
    RopeEnum             rope_type_ = RopeEnum::NONE;
    bool                 is_paged_kv_;

    bool is_pad_q_seq_len_;    // padding for seqlen_q
    bool is_pad_kv_seq_len_;   // padding for seqlen_k
    bool is_pad_qk_head_dim_;  // paddding for hdim_q
    bool is_pad_v_head_dim_;   // paddding for hdim_v
    bool is_pad_qkv_head_dim_;

    int block_per_cu_ = -1;  // overwrite occupancy if not -1

    std::string GetPadName();

    std::string GetPipelineConfigName();

    std::string GetInstanceName();

    std::string Emit();
};

}  // namespace flashck