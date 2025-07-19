#pragma once

#include <array>
#include <string>

#include "flashck/core/profiling/tile/fmha/fmha_fwd_codegen.h"
#include "flashck/core/profiling/tile/fmha/fmha_library.h"
#include "flashck/core/utils/dtype.h"

namespace flashck {

/**
 * @class FmhaFwdSplitKVCodeGen
 * @brief Code generator for Forward FMHA SplitKV operations
 *
 * This class encapsulates all the parameters and configuration needed to generate
 * optimized GPU kernels for Forward Multi-Head Attention SplitKV operations.
 * SplitKV is used for handling very long sequences by splitting key-value computation
 * across multiple blocks and then combining the results.
 */
class FmhaFwdSplitKVCodeGen {
public:
    /**
     * @brief Default constructor with sensible defaults
     */
    FmhaFwdSplitKVCodeGen() = default;

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

    FmhaKind kind_ = FmhaKind::FwdSplitKV;  ///< Type of FMHA operation (always FwdSplitKV)

    // ====================== Data Type Configuration ======================

    DataType dtype_ = DataType::FLOAT16;  ///< Primary data type for Q, K, V tensors

    // ====================== Attention Configuration ======================

    FmhaMode                 mode_        = FmhaMode::Batch;  ///< Batch or Group mode for attention computation
    GenericAttentionMaskEnum mask_type_   = GenericAttentionMaskEnum::NO_MASK;  ///< Type of attention mask applied
    std::array<int64_t, 2>   window_size_ = {-1, -1};                           ///< Sliding window size [left, right]
    BiasEnum                 bias_enum_   = BiasEnum::NO_BIAS;                  ///< Type of bias applied to attention

    // ====================== Tiling Configuration ======================

    FmhaTileDesc tile_desc_;  ///< Tile configuration for this FMHA SplitKV operation

    // ====================== SplitKV Specific Configuration ======================

    bool has_uneven_splits_                  = false;  ///< Enable handling of uneven splits across sequence length
    bool is_store_lse_                       = false;  ///< Enable storing log-sum-exp values for numerical stability
    bool is_merge_num_head_groups_seq_len_q_ = false;  ///< Enable merging head groups with sequence length for Q

    // ====================== Memory Configuration ======================

    bool is_paged_kv_ = false;  ///< Enable paged key-value cache for memory efficiency

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