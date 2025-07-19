#pragma once

#include <array>
#include <string>
#include <vector>

#include "flashck/core/profiling/tile/fmha/fmha_library.h"
#include "flashck/core/utils/dtype.h"

namespace flashck {

/**
 * @class FmhaTileDesc
 * @brief Describes the tiling configuration for FMHA operations
 *
 * This class defines how the FMHA computation is divided across thread blocks
 * and how attention computation is tiled across different dimensions.
 * It specifies the work distribution strategy for optimal GPU performance.
 */
class FmhaTileDesc {
public:
    /**
     * @brief Generate a unique name for this tile configuration
     * @return String identifier based on tile parameters
     */
    std::string GetInstanceName() const;

    /**
     * @brief Generate code template parameters for this tile
     * @return String representation for code generation
     */
    std::string Emit() const;

    // ====================== Q-K GEMM Tile Configuration ======================

    int64_t bm0_;      ///< Tile size along query sequence length (block size for Q dimension)
    int64_t bn0_;      ///< Tile size along key sequence length (block size for K dimension)
    int64_t bk0_;      ///< Tile size along Q-K GEMM unroll dimension (head dimension)
    int64_t bk0_max_;  ///< Total length of K0, used for pipelines that load Q at once

    // ====================== Attention-V GEMM Tile Configuration ======================

    int64_t bn1_;  ///< Tile size along value head dimension
    int64_t bk1_;  ///< Tile size along K-V GEMM unroll dimension (sequence length)

    // ====================== Warp Distribution for Q-K GEMM ======================

    int64_t rm0_;  ///< Number of warps for GEMM0 along query sequence length
    int64_t rn0_;  ///< Number of warps for GEMM0 along key sequence length
    int64_t rk0_;  ///< Number of warps for GEMM0 along head dimension (not used)

    // ====================== Warp Distribution for Attention-V GEMM ======================

    int64_t rm1_;  ///< Number of warps for GEMM1 along query sequence length
    int64_t rn1_;  ///< Number of warps for GEMM1 along value head dimension
    int64_t rk1_;  ///< Number of warps for GEMM1 along key sequence length (not used)

    // ====================== Warp-Level Tile Sizes ======================

    int64_t wm0_;  ///< GEMM0 warp tile size along M dimension (query sequence)
    int64_t wn0_;  ///< GEMM0 warp tile size along N dimension (key sequence)
    int64_t wk0_;  ///< GEMM0 warp tile size along K dimension (head dimension)
    int64_t wm1_;  ///< GEMM1 warp tile size along M dimension (query sequence)
    int64_t wn1_;  ///< GEMM1 warp tile size along N dimension (value head dimension)
    int64_t wk1_;  ///< GEMM1 warp tile size along K dimension (key sequence)
};

/**
 * @class FmhaFwdCodeGen
 * @brief Code generator for Forward FMHA operations
 *
 * This class encapsulates all the parameters and configuration needed to generate
 * optimized GPU kernels for Forward Multi-Head Attention operations. It combines
 * problem specifications with tiling strategies and attention-specific configurations.
 */
class FmhaFwdCodeGen {
public:
    /**
     * @brief Default constructor with sensible defaults
     */
    FmhaFwdCodeGen() = default;

    /**
     * @brief Generate padding configuration name
     * @return String identifier for padding configuration
     */
    std::string GetPadName() const;

    /**
     * @brief Generate pipeline configuration name
     * @return String identifier for pipeline configuration
     */
    std::string GetPipelineConfigName() const;

    /**
     * @brief Generate a unique instance name for this configuration
     * @return String identifier combining operation type and parameters
     */
    std::string GetInstanceName() const;

    /**
     * @brief Generate the complete kernel code for this configuration
     * @return String containing the generated GPU kernel code
     */
    std::string Emit() const;

    // ====================== Operation Configuration ======================

    FmhaKind kind_ = FmhaKind::Fwd;  ///< Type of FMHA operation (Fwd, FwdAppendKV, FwdSplitKV, etc.)

    // ====================== Data Type Configuration ======================

    DataType dtype_ = DataType::FLOAT16;  ///< Primary data type for Q, K, V tensors

    // ====================== Attention Configuration ======================

    FmhaMode                 mode_        = FmhaMode::Batch;  ///< Batch or Group mode for attention computation
    GenericAttentionMaskEnum mask_type_   = GenericAttentionMaskEnum::NO_MASK;  ///< Type of attention mask applied
    std::array<int64_t, 2>   window_size_ = {-1, -1};                           ///< Sliding window size [left, right]
    BiasEnum                 bias_enum_   = BiasEnum::NO_BIAS;                  ///< Type of bias applied to attention

    // ====================== Tiling Configuration ======================

    FmhaTileDesc tile_desc_;  ///< Tile configuration for this FMHA operation

    // ====================== Padding Configuration ======================

    bool is_pad_q_seq_len_    = false;  ///< Enable padding for query sequence length
    bool is_pad_kv_seq_len_   = false;  ///< Enable padding for key-value sequence length
    bool is_pad_qk_head_dim_  = false;  ///< Enable padding for query-key head dimension
    bool is_pad_v_head_dim_   = false;  ///< Enable padding for value head dimension
    bool is_pad_qkv_head_dim_ = false;  ///< Enable padding for unified QKV head dimension

    // ====================== Performance Configuration ======================

    int block_per_cu_ = -1;  ///< Override occupancy if not -1 (blocks per compute unit)

    // ====================== Quantization Configuration ======================

    bool is_static_quant_ = false;  ///< Enable static quantization

    // ====================== Pipeline Configuration ======================

    BlockFmhaPipelineEnum pipeline_ = BlockFmhaPipelineEnum::QRKSVS;  ///< FMHA pipeline implementation variant
};

}  // namespace flashck