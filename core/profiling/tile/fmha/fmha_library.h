#pragma once

#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "core/utils/common.h"

namespace flashck {

/**
 * @enum FmhaMode
 * @brief Defines the modes of FMHA operations
 */
enum class FmhaMode : int {
    Batch = 0,  ///< Batch mode attention
    Group = 1,  ///< Group mode attention
    COUNT       // Used for iteration and validation
};

/**
 * @struct FmhaModeInfo
 * @brief Information about FMHA operation modes
 */
struct FmhaModeInfo {
    std::string name;        ///< Human-readable name
    std::string short_name;  ///< Abbreviated name for config strings
};

/**
 * @brief Mapping from FMHA modes to their information
 */
static const std::unordered_map<FmhaMode, FmhaModeInfo> g_fmha_mode_map = {
    {FmhaMode::Batch, {"batch", "B"}},
    {FmhaMode::Group, {"group", "G"}},
};

/**
 * @enum FmhaKind
 * @brief Defines the types of FMHA operations supported
 */
enum class FmhaKind : int {
    Fwd               = 0,  ///< Standard forward attention
    FwdAppendKV       = 1,  ///< Forward with key-value appending
    FwdSplitKV        = 2,  ///< Forward with split key-value
    FwdSplitKVCombine = 3,  ///< Forward split KV with combine step
    BatchPrefill = 4,
    PagedKVPrefill = 5,
    COUNT                   // Used for iteration and validation
};

/**
 * @struct FmhaKindInfo
 * @brief Information about FMHA operation types
 */
struct FmhaKindInfo {
    std::string name;        ///< Human-readable name
    std::string short_name;  ///< Abbreviated name for config strings
};

/**
 * @brief Mapping from FMHA kinds to their information
 */
static const std::unordered_map<FmhaKind, FmhaKindInfo> g_fmha_kind_map = {
    {FmhaKind::Fwd, {"fmha_fwd", "F"}},
    {FmhaKind::FwdAppendKV, {"fmha_fwd_append_kv", "FA"}},
    {FmhaKind::FwdSplitKV, {"fmha_fwd_split_kv", "FS"}},
    {FmhaKind::FwdSplitKVCombine, {"fmha_fwd_split_kv_combine", "FSC"}},
    {FmhaKind::BatchPrefill, {"fmha_batch_prefill", "FBP"}},
    {FmhaKind::PagedKVPrefill, {"fmha_paged_kv_prefill", "FPKP"}},
};

/**
 * @enum QuantMode
 * @brief Defines quantization modes for FMHA operations
 */
enum class QuantMode : int {
    None  = 0,  ///< No quantization
    Auto  = 1,  ///< Automatic quantization
    Quant = 2,  ///< Explicit quantization
    COUNT       // Used for validation
};

/**
 * @struct QuantModeInfo
 * @brief Information about quantization modes
 */
struct QuantModeInfo {
    std::string name;        ///< Human-readable name
    std::string short_name;  ///< Abbreviated name for config strings
};

/**
 * @brief Mapping from quantization modes to their information
 */
static const std::unordered_map<QuantMode, QuantModeInfo> g_quant_mode_map = {
    {QuantMode::None, {"none", "N"}},
    {QuantMode::Auto, {"auto", "A"}},
    {QuantMode::Quant, {"quant", "Q"}},
};

/**
 * @enum GenericAttentionMaskEnum
 * @brief Defines different attention mask types
 */
enum class GenericAttentionMaskEnum : int {
    NO_MASK                = 0,  ///< No attention mask applied
    MASK_FROM_TOP_LEFT     = 1,  ///< Causal mask from top-left
    MASK_FROM_BOTTOM_RIGHT = 2,  ///< Mask from bottom-right
    MASK_GENERIC           = 3,  ///< Generic custom mask
    COUNT                        // Used for validation
};

/**
 * @struct AttentionMaskInfo
 * @brief Information about attention mask types
 */
struct AttentionMaskInfo {
    std::string name;        ///< Human-readable name
    std::string short_name;  ///< Abbreviated name for config strings
};

/**
 * @brief Mapping from attention mask types to their information
 */
static const std::unordered_map<GenericAttentionMaskEnum, AttentionMaskInfo> g_attention_mask_map = {
    {GenericAttentionMaskEnum::NO_MASK, {"no_mask", "NM"}},
    {GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT, {"mask_top_left", "TL"}},
    {GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT, {"mask_bottom_right", "BR"}},
    {GenericAttentionMaskEnum::MASK_GENERIC, {"mask_generic", "GE"}},
};

/**
 * @struct MaskEnumInfo
 * @brief Extended mask information including sliding window size
 */
struct MaskEnumInfo {
    GenericAttentionMaskEnum type_;                 ///< Mask type
    int64_t                  sliding_window_size_;  ///< Size of sliding window (-1 for no window)
};

/**
 * @enum BiasEnum
 * @brief Defines different bias types for attention
 */
enum class BiasEnum : int {
    NO_BIAS          = 0,  ///< No bias applied
    ELEMENTWISE_BIAS = 1,  ///< Element-wise bias addition
    ALIBI            = 2,  ///< ALiBi positional bias
    COUNT                  // Used for validation
};

/**
 * @struct BiasInfo
 * @brief Information about bias types
 */
struct BiasInfo {
    std::string name;        ///< Human-readable name
    std::string class_tag;   ///< C++ class template tag
    std::string short_name;  ///< Abbreviated name for config strings
};

/**
 * @brief Mapping from bias types to their information
 */
static const std::unordered_map<BiasEnum, BiasInfo> g_bias_enum_map = {
    {BiasEnum::NO_BIAS, {"no_bias", "ck_tile::BlockAttentionBiasEnum::NO_BIAS", "NB"}},
    {BiasEnum::ELEMENTWISE_BIAS, {"elementwise_bias", "ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS", "EB"}},
    {BiasEnum::ALIBI, {"alibi", "ck_tile::BlockAttentionBiasEnum::ALIBI", "AB"}},
};

/**
 * @struct BiasEnumInfo
 * @brief Extended bias information with rank details
 */
struct BiasEnumInfo {
    BiasEnum type_;       ///< Bias type
    int      rank_info_;  ///< Rank information for bias tensor shape
    /*
     * Dispatch logic for bias types:
     *
     * if type == ELEMENTWISE_BIAS:
     *      if rank_info == 0: bias is 1*1*s*s
     *      elif rank_info == 1: bias is 1*h*s*s
     *      elif rank_info == 2: bias is b*h*s*s
     *
     * elif type == ALIBI:
     *      if rank_info == 0: alibi in 1*h
     *      elif rank_info == 1: alibi in b*h
     */

    /**
     * @brief Stream output operator for BiasEnumInfo
     */
    friend std::ostream& operator<<(std::ostream& os, const BiasEnumInfo& info)
    {
        if (info.type_ == BiasEnum::NO_BIAS) {
            os << "n";
        }
        else if (info.type_ == BiasEnum::ELEMENTWISE_BIAS) {
            os << "e";
            if (info.rank_info_ != 0) {
                os << "[" << info.rank_info_ << "]";
            }
        }
        else if (info.type_ == BiasEnum::ALIBI) {
            os << "alibi";
            if (info.rank_info_ != 0) {
                os << "[" << info.rank_info_ << "]";
            }
        }
        return os;
    }
};

/**
 * @enum RopeEnum
 * @brief Defines different RoPE (Rotary Position Embedding) modes
 */
enum class RopeEnum : int {
    NONE         = 0,  ///< No rotary embedding
    INTERLEAVED  = 1,  ///< Interleaved rotation (dims 0&1, 2&3, etc.)
    HALF_ROTATED = 2,  ///< Half-rotated mode (dims 0&rotary_dim/2, 1&rotary_dim/2+1, etc.)
    COUNT              // Used for validation
};

/**
 * @struct RopeInfo
 * @brief Information about RoPE embedding types
 */
struct RopeInfo {
    std::string name;        ///< Human-readable name
    std::string class_tag;   ///< C++ class template tag
    std::string short_name;  ///< Abbreviated name for config strings
};

/**
 * @brief Mapping from RoPE types to their information
 */
static const std::unordered_map<RopeEnum, RopeInfo> g_rope_enum_map = {
    {RopeEnum::NONE, {"none", "ck_tile::RotaryEmbeddingEnum::NONE", "N"}},
    {RopeEnum::INTERLEAVED, {"interleaved", "ck_tile::RotaryEmbeddingEnum::INTERLEAVED", "I"}},
    {RopeEnum::HALF_ROTATED, {"half_rotated", "ck_tile::RotaryEmbeddingEnum::HALF_ROTATED", "H"}},
};

/**
 * @enum BlockFmhaPipelineEnum
 * @brief Defines different FMHA pipeline implementations
 */
enum class BlockFmhaPipelineEnum : int {
    QRKSVS            = 0,  ///< QR-KS-VS pipeline
    QRKSVS_ASYNC      = 1,  ///< Asynchronous QR-KS-VS pipeline
    QR_NWARP_SSHUFFLE = 2,  ///< N-warp shuffle QR pipeline
    QSKSVS            = 3,  ///< QS-KS-VS pipeline
    COUNT                   // Used for validation
};

/**
 * @struct PipelineInfo
 * @brief Information about FMHA pipeline types
 */
struct PipelineInfo {
    std::string name;        ///< Human-readable name
    std::string class_tag;   ///< C++ class template tag
    std::string short_name;  ///< Abbreviated name for config strings
};

/**
 * @brief Mapping for forward FMHA pipelines
 */
static const std::unordered_map<BlockFmhaPipelineEnum, PipelineInfo> g_block_fmha_fwd_pipeline_map = {
    {BlockFmhaPipelineEnum::QRKSVS, {"qr_ks_vs", "ck_tile::BlockFmhaPipelineQRKSVS", "QR"}},
    {BlockFmhaPipelineEnum::QRKSVS_ASYNC, {"qr_ks_vs_async", "ck_tile::BlockFmhaPipelineQRKSVSAsync", "QRA"}},
};

/**
 * @brief Mapping for forward split-KV FMHA pipelines
 */
static const std::unordered_map<BlockFmhaPipelineEnum, PipelineInfo> g_block_fmha_fwd_splitkv_pipeline_map = {
    {BlockFmhaPipelineEnum::QRKSVS, {"splitkv_qr_ks_vs", "ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVS", "SKQR"}},
    {BlockFmhaPipelineEnum::QR_NWARP_SSHUFFLE,
     {"splitkv_qr_nwarp_shuffle", "ck_tile::BlockFmhaFwdSplitKVPipelineNWarpSShuffleQRKSVS", "SKNW"}},
    {BlockFmhaPipelineEnum::QRKSVS_ASYNC,
     {"splitkv_qr_ks_vs_async", "ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVSAsync", "SKQRA"}},
};

// ====================== Utility Functions ======================

/**
 * @brief Gets the name string for an FMHA mode
 * @param mode The FMHA mode to query
 * @return The name string, or "unknown" if not found
 */
inline std::string GetFmhaModeName(FmhaMode mode)
{
    auto it = g_fmha_mode_map.find(mode);
    return it != g_fmha_mode_map.end() ? it->second.name : "unknown";
}

/**
 * @brief Gets the short name for an FMHA mode
 * @param mode The FMHA mode to query
 * @return The short name string, or "unknown" if not found
 */
inline std::string GetFmhaModeShortName(FmhaMode mode)
{
    auto it = g_fmha_mode_map.find(mode);
    return it != g_fmha_mode_map.end() ? it->second.short_name : "unknown";
}

/**
 * @brief Gets the name string for an FMHA kind
 * @param kind The FMHA kind to query
 * @return The name string, or "unknown" if not found
 */
inline std::string GetFmhaKindName(FmhaKind kind)
{
    auto it = g_fmha_kind_map.find(kind);
    return it != g_fmha_kind_map.end() ? it->second.name : "unknown";
}

/**
 * @brief Gets the short name for an FMHA kind
 * @param kind The FMHA kind to query
 * @return The short name string, or "unknown" if not found
 */
inline std::string GetFmhaKindShortName(FmhaKind kind)
{
    auto it = g_fmha_kind_map.find(kind);
    return it != g_fmha_kind_map.end() ? it->second.short_name : "unknown";
}

/**
 * @brief Gets the name string for a quantization mode
 * @param mode The quantization mode to query
 * @return The name string, or "unknown" if not found
 */
inline std::string GetQuantModeName(QuantMode mode)
{
    auto it = g_quant_mode_map.find(mode);
    return it != g_quant_mode_map.end() ? it->second.name : "unknown";
}

/**
 * @brief Gets the short name for a quantization mode
 * @param mode The quantization mode to query
 * @return The short name string, or "unknown" if not found
 */
inline std::string GetQuantModeShortName(QuantMode mode)
{
    auto it = g_quant_mode_map.find(mode);
    return it != g_quant_mode_map.end() ? it->second.short_name : "unknown";
}

/**
 * @brief Gets the name string for an attention mask type
 * @param mask The attention mask type to query
 * @return The name string, or "unknown" if not found
 */
inline std::string GetAttentionMaskName(GenericAttentionMaskEnum mask)
{
    auto it = g_attention_mask_map.find(mask);
    return it != g_attention_mask_map.end() ? it->second.name : "unknown";
}

/**
 * @brief Gets the short name for an attention mask type
 * @param mask The attention mask type to query
 * @return The short name string, or "unknown" if not found
 */
inline std::string GetAttentionMaskShortName(GenericAttentionMaskEnum mask)
{
    auto it = g_attention_mask_map.find(mask);
    return it != g_attention_mask_map.end() ? it->second.short_name : "unknown";
}

/**
 * @brief Gets the name string for a bias type
 * @param bias The bias type to query
 * @return The name string, or "unknown" if not found
 */
inline std::string GetBiasName(BiasEnum bias)
{
    auto it = g_bias_enum_map.find(bias);
    return it != g_bias_enum_map.end() ? it->second.name : "unknown";
}

/**
 * @brief Gets the class tag for a bias type
 * @param bias The bias type to query
 * @return The class tag string, or "unknown" if not found
 */
inline std::string GetBiasClassTag(BiasEnum bias)
{
    auto it = g_bias_enum_map.find(bias);
    return it != g_bias_enum_map.end() ? it->second.class_tag : "unknown";
}

/**
 * @brief Gets the short name for a bias type
 * @param bias The bias type to query
 * @return The short name string, or "unknown" if not found
 */
inline std::string GetBiasShortName(BiasEnum bias)
{
    auto it = g_bias_enum_map.find(bias);
    return it != g_bias_enum_map.end() ? it->second.short_name : "unknown";
}

/**
 * @brief Gets the name string for a RoPE type
 * @param rope The RoPE type to query
 * @return The name string, or "unknown" if not found
 */
inline std::string GetRopeName(RopeEnum rope)
{
    auto it = g_rope_enum_map.find(rope);
    return it != g_rope_enum_map.end() ? it->second.name : "unknown";
}

/**
 * @brief Gets the class tag for a RoPE type
 * @param rope The RoPE type to query
 * @return The class tag string, or "unknown" if not found
 */
inline std::string GetRopeClassTag(RopeEnum rope)
{
    auto it = g_rope_enum_map.find(rope);
    return it != g_rope_enum_map.end() ? it->second.class_tag : "unknown";
}

/**
 * @brief Gets the short name for a RoPE type
 * @param rope The RoPE type to query
 * @return The short name string, or "unknown" if not found
 */
inline std::string GetRopeShortName(RopeEnum rope)
{
    auto it = g_rope_enum_map.find(rope);
    return it != g_rope_enum_map.end() ? it->second.short_name : "unknown";
}

/**
 * @brief Gets the pipeline information for forward FMHA
 * @param pipeline The pipeline type to query
 * @return The pipeline info, or default values if not found
 */
inline PipelineInfo GetFwdPipelineInfo(BlockFmhaPipelineEnum pipeline)
{
    auto it = g_block_fmha_fwd_pipeline_map.find(pipeline);
    return it != g_block_fmha_fwd_pipeline_map.end() ? it->second : PipelineInfo{"unknown", "unknown", "UK"};
}

/**
 * @brief Gets the pipeline information for forward split-KV FMHA
 * @param pipeline The pipeline type to query
 * @return The pipeline info, or default values if not found
 */
inline PipelineInfo GetFwdSplitKVPipelineInfo(BlockFmhaPipelineEnum pipeline)
{
    auto it = g_block_fmha_fwd_splitkv_pipeline_map.find(pipeline);
    return it != g_block_fmha_fwd_splitkv_pipeline_map.end() ? it->second : PipelineInfo{"unknown", "unknown", "UK"};
}

/**
 * @brief Gets the class tag for forward FMHA pipeline
 * @param pipeline The pipeline type to query
 * @return The C++ class template tag, or "unknown" if not found
 */
inline std::string GetFwdPipelineClassTag(BlockFmhaPipelineEnum pipeline)
{
    auto it = g_block_fmha_fwd_pipeline_map.find(pipeline);
    return it != g_block_fmha_fwd_pipeline_map.end() ? it->second.class_tag : "unknown";
}

/**
 * @brief Gets the class tag for forward split-KV FMHA pipeline
 * @param pipeline The pipeline type to query
 * @return The C++ class template tag, or "unknown" if not found
 */
inline std::string GetFwdSplitKVPipelineClassTag(BlockFmhaPipelineEnum pipeline)
{
    auto it = g_block_fmha_fwd_splitkv_pipeline_map.find(pipeline);
    return it != g_block_fmha_fwd_splitkv_pipeline_map.end() ? it->second.class_tag : "unknown";
}

// ====================== Validation Functions ======================

/**
 * @brief Validates if an FMHA mode is valid
 * @param mode The FMHA mode to validate
 * @return true if valid, false otherwise
 */
inline bool IsValidFmhaMode(FmhaMode mode)
{
    return static_cast<int>(mode) >= 0 && static_cast<int>(mode) < static_cast<int>(FmhaMode::COUNT);
}

/**
 * @brief Validates if an FMHA kind is valid
 * @param kind The FMHA kind to validate
 * @return true if valid, false otherwise
 */
inline bool IsValidFmhaKind(FmhaKind kind)
{
    return static_cast<int>(kind) >= 0 && static_cast<int>(kind) < static_cast<int>(FmhaKind::COUNT);
}

/**
 * @brief Validates if a quantization mode is valid
 * @param mode The quantization mode to validate
 * @return true if valid, false otherwise
 */
inline bool IsValidQuantMode(QuantMode mode)
{
    return static_cast<int>(mode) >= 0 && static_cast<int>(mode) < static_cast<int>(QuantMode::COUNT);
}

/**
 * @brief Validates if an attention mask type is valid
 * @param mask The attention mask type to validate
 * @return true if valid, false otherwise
 */
inline bool IsValidAttentionMask(GenericAttentionMaskEnum mask)
{
    return static_cast<int>(mask) >= 0 && static_cast<int>(mask) < static_cast<int>(GenericAttentionMaskEnum::COUNT);
}

/**
 * @brief Validates if a bias type is valid
 * @param bias The bias type to validate
 * @return true if valid, false otherwise
 */
inline bool IsValidBiasType(BiasEnum bias)
{
    return static_cast<int>(bias) >= 0 && static_cast<int>(bias) < static_cast<int>(BiasEnum::COUNT);
}

/**
 * @brief Validates if a RoPE type is valid
 * @param rope The RoPE type to validate
 * @return true if valid, false otherwise
 */
inline bool IsValidRopeType(RopeEnum rope)
{
    return static_cast<int>(rope) >= 0 && static_cast<int>(rope) < static_cast<int>(RopeEnum::COUNT);
}

/**
 * @brief Validates if a pipeline type is valid
 * @param pipeline The pipeline type to validate
 * @return true if valid, false otherwise
 */
inline bool IsValidPipelineType(BlockFmhaPipelineEnum pipeline)
{
    return static_cast<int>(pipeline) >= 0
           && static_cast<int>(pipeline) < static_cast<int>(BlockFmhaPipelineEnum::COUNT);
}

}  // namespace flashck