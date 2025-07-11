#pragma once

namespace flashck {

struct FmhaTileDesc {
    int64_t bm0_;      // tile size along q seqlen (block size)
    int64_t bn0_;      // tile size along k seqlen
    int64_t bk0_;      // tile size along qk gemm unroll
    int64_t bn1_;      // tile size along v head_dim
    int64_t bk1_;      // tile size along kv gemm unroll
    int64_t bk0_max_;  // total length of K0, used for pipeline that need load Q at once (or repeately load Q as a whole
                       // tile)

    int64_t rm0_;  // number of warps for gemm0 along q seqlen
    int64_t rn0_;  // number of warps for gemm0 along k seqlen
    int64_t rk0_;  // number of warps for gemm0 along head dim q (not used)
    int64_t rm1_;  // number of warps for gemm1 along q seqlen
    int64_t rn1_;  // number of warps for gemm1 along head dim v
    int64_t rk1_;  // number of warps for gemm1 along k seqlen (not used)

    int64_t wm0_;  // gemm0 warp size along m (warp size)
    int64_t wn0_;  // gemm0 warp size along n
    int64_t wk0_;  // gemm0 warp size along k
    int64_t wm1_;  // gemm1 warp size along m (warp size)
    int64_t wn1_;  // gemm1 warp size along n
    int64_t wk1_;  // gemm1 warp size along k

    std::string GetConfigName();

    std::string Emit();
};

class FmhaFwdCodeGen: public CodeGenBase {
public:
    std::string GetPadName();

    std::string GetPipelineConfigName();

    std::string GetConfigName();

    std::string Emit();

    FmhaOperationKind kind_;
    TensorOperation   epilogue_op_;

    DataType                 dtype_;
    FmhaTileDesc             tile_desc_;
    FmhaOperationMode        operation_mode_;  // kIsGroupMode_
    GenericAttentionMaskEnum mask_type_;
    std::array<int64_t, 2>   window_size_;
    BiasEnum                 bias_enum_;

    bool is_pad_q_seq_len_;     // padding for seqlen_q
    bool is_pad_kv_seq_len_;    // padding for seqlen_k
    bool is_pad_qk_head_dim_;   // paddding for hdim_q
    bool is_pad_v_head_dim_;    // paddding for hdim_v
    bool is_pad_qkv_head_dim_;  // paddding for hdim_qkv

    int block_per_cu_ = -1;  // overwrite occupancy if not -1

    bool is_static_quant_;

    BlockFmhaPipelineEnum pipeline_ = BlockFmhaPipelineEnum::QRKSVS;
};

}  // namespace flashck