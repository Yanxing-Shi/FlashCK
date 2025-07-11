#pragma once

#include "flashck/core/profiling/library.h"
#include "flashck/core/utils/dtype.h"

#include "flashck/core/profiling/fmha_fwd_operation.h"

namespace flashck {

struct FmhaFwdSplitKVOperation {
    FmhaOperationKind kind_;
    TensorOperation   epilogue_op_;

    DataType                 dtype_;
    FmhaTileDesc             tile_desc_;
    FmhaOperationMode        operation_mode_;  // kIsGroupMode_
    GenericAttentionMaskEnum mask_type_;
    std::array<int64_t, 2>   window_size_;
    BiasEnum                 bias_enum_;

    bool is_static_quant_;
    bool is_paged_kv_;

    bool is_pad_q_seq_len_;    // padding for seqlen_q
    bool is_pad_kv_seq_len_;   // padding for seqlen_k
    bool is_pad_qk_head_dim_;  // paddding for hdim_q
    bool is_pad_v_head_dim_;   // paddding for hdim_v
    bool is_pad_qkv_head_dim_;

    int block_per_cu_ = -1;  // overwrite occupancy if not -1

    bool has_uneven_splits_;
    bool is_store_lse_;
    bool is_merge_num_head_groups_seq_len_q_;

    BlockFmhaPipelineEnum pipeline_ = BlockFmhaPipelineEnum::QRKSVS;

    std::string GetPadName();

    std::string GetPipelineConfigName();

    std::string GetConfigName();

    std::string Emit();
};

}  // namespace flashck