#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace flashck {

/**
 * @enum NormKind
 * @brief Defines the types of normalization operations supported
 */
enum class NormKind : int {
    LayerNorm = 0,  ///< Layer normalization
    RMSNorm   = 1,  ///< Root Mean Square normalization
    // Add new norm types here
    COUNT  // Used for iteration and validation
};

/**
 * @struct NormTag
 * @brief Configuration tags for norm operation code generation
 *
 * Contains template and class name mappings used during code generation
 * for different normalization kernels.
 */
struct NormTag {
    std::string name;         ///< Human-readable name of the normalization
    std::string problem_tag;  ///< Template tag for problem definition
    std::string trait_tag;    ///< Template tag for kernel traits
    std::string fwd_tag;      ///< Template tag for forward implementation
    std::string pass_tag;     ///< Template tag for pipeline pass strategy
};

/**
 * @brief Mapping from norm types to their configuration tags
 */
static const std::unordered_map<NormKind, NormTag> g_norm_map = {
    {NormKind::LayerNorm,
     {"layer_norm",
      "Layernorm2dFwdPipelineProblem",
      "Layernorm2dFwdTraits",
      "Layernorm2dFwd",
      "Layernorm2dFwdPipelineTwoPass"}},
    {NormKind::RMSNorm,
     {"rms_norm",
      "Rmsnorm2dFwdPipelineProblem",
      "Rmsnorm2dFwdTraits",
      "Rmsnorm2dFwd",
      "Layernorm2dFwdPipelineOnePass"}},
};

/**
 * @enum NormBiasEnum
 * @brief Defines bias handling modes for normalization operations
 */
enum class NormBiasEnum : int {
    NO_BIAS  = 0,  ///< No bias term applied
    ADD_BIAS = 1,  ///< Add bias before fused operations
    COUNT          // Used for validation
};

/**
 * @struct NormBiasInfo
 * @brief Information about bias configuration
 */
struct NormBiasInfo {
    std::string name;        ///< Full descriptive name
    std::string short_name;  ///< Abbreviated name for config strings
};

/**
 * @brief Mapping from bias modes to their information
 */
static const std::unordered_map<NormBiasEnum, NormBiasInfo> g_norm_bias_map = {
    {NormBiasEnum::NO_BIAS, {"no_bias", "nb"}},
    {NormBiasEnum::ADD_BIAS, {"add_bias", "ab"}},
};

/**
 * @enum FusedAddEnum
 * @brief Defines fused addition operation modes
 */
enum class FusedAddEnum : int {
    NO_ADD        = 0,  ///< No fused addition
    PRE_ADD_STORE = 1,  ///< Fused add before norm with global store
    PRE_ADD       = 2,  ///< Fused add before norm without store
    COUNT               // Used for validation
};

/**
 * @struct FusedAddInfo
 * @brief Information about fused addition configuration
 */
struct FusedAddInfo {
    std::string name;        ///< Full descriptive name
    std::string short_name;  ///< Abbreviated name for config strings
};

/**
 * @brief Mapping from fused add modes to their information
 */
static const std::unordered_map<FusedAddEnum, FusedAddInfo> g_fused_add_map = {
    {FusedAddEnum::NO_ADD, {"no_add", "na"}},
    {FusedAddEnum::PRE_ADD_STORE, {"pre_add_store", "pas"}},
    {FusedAddEnum::PRE_ADD, {"pre_add", "pa"}},
};

/**
 * @enum FusedQuantEnum
 * @brief Defines fused quantization operation modes
 */
enum class FusedQuantEnum : int {
    NO_SWEEP             = 0,  ///< No quantization applied
    SMOOTH_DYNAMIC_QUANT = 1,  ///< Smooth outlier + rowwise quantization
    DYNAMIC_QUANT        = 2,  ///< Rowwise quantization only
    COUNT                      // Used for validation
};

/**
 * @struct FusedQuantInfo
 * @brief Information about fused quantization configuration
 */
struct FusedQuantInfo {
    std::string name;        ///< Full descriptive name
    std::string short_name;  ///< Abbreviated name for config strings
};

/**
 * @brief Mapping from fused quantization modes to their information
 */
static const std::unordered_map<FusedQuantEnum, FusedQuantInfo> g_fused_quant_map = {
    {FusedQuantEnum::NO_SWEEP, {"no_sweep", "ns"}},
    {FusedQuantEnum::SMOOTH_DYNAMIC_QUANT, {"smooth_dynamic_quant", "sdq"}},
    {FusedQuantEnum::DYNAMIC_QUANT, {"dynamic_quant", "dq"}},
};

// ====================== Utility Functions ======================

/**
 * @brief Gets the name string for a norm kind
 * @param kind The norm kind to query
 * @return The name string, or "unknown" if not found
 */
inline std::string GetNormKindName(NormKind kind)
{
    auto it = g_norm_map.find(kind);
    return it != g_norm_map.end() ? it->second.name : "unknown";
}

/**
 * @brief Gets the bias info name for a bias enum
 * @param bias The bias enum to query
 * @return The name string, or "unknown" if not found
 */
inline std::string GetNormBiasName(NormBiasEnum bias)
{
    auto it = g_norm_bias_map.find(bias);
    return it != g_norm_bias_map.end() ? it->second.name : "unknown";
}

/**
 * @brief Gets the fused add info name for a fused add enum
 * @param add The fused add enum to query
 * @return The name string, or "unknown" if not found
 */
inline std::string GetFusedAddName(FusedAddEnum add)
{
    auto it = g_fused_add_map.find(add);
    return it != g_fused_add_map.end() ? it->second.name : "unknown";
}

/**
 * @brief Gets the fused quant info name for a fused quant enum
 * @param quant The fused quant enum to query
 * @return The name string, or "unknown" if not found
 */
inline std::string GetFusedQuantName(FusedQuantEnum quant)
{
    auto it = g_fused_quant_map.find(quant);
    return it != g_fused_quant_map.end() ? it->second.name : "unknown";
}

inline std::string GetFusedAddShortName(FusedAddEnum add)
{
    auto it = g_fused_add_map.find(add);
    return it != g_fused_add_map.end() ? it->second.short_name : "unknown";
}

inline std::string GetFusedQuantShortName(FusedQuantEnum quant)
{
    auto it = g_fused_quant_map.find(quant);
    return it != g_fused_quant_map.end() ? it->second.short_name : "unknown";
}

inline std::string GetNormBiasShortName(NormBiasEnum bias)
{
    auto it = g_norm_bias_map.find(bias);
    return it != g_norm_bias_map.end() ? it->second.short_name : "unknown";
}

inline std::string GetNormKindShortName(NormKind kind)
{
    auto it = g_norm_map.find(kind);
    return it != g_norm_map.end() ? it->second.name : "unknown";
}

inline std::string GetNormKindProblemTag(NormKind kind)
{
    auto it = g_norm_map.find(kind);
    return it != g_norm_map.end() ? it->second.problem_tag : "unknown";
}

inline std::string GetNormKindTraitTag(NormKind kind)
{
    auto it = g_norm_map.find(kind);
    return it != g_norm_map.end() ? it->second.trait_tag : "unknown";
}

inline std::string GetNormKindFwdTag(NormKind kind)
{
    auto it = g_norm_map.find(kind);
    return it != g_norm_map.end() ? it->second.fwd_tag : "unknown";
}

inline std::string GetNormKindPassTag(NormKind kind)
{
    auto it = g_norm_map.find(kind);
    return it != g_norm_map.end() ? it->second.pass_tag : "unknown";
}

/**
 * @brief Validates if a norm kind is valid
 * @param kind The norm kind to validate
 * @return true if valid, false otherwise
 */
inline bool IsValidNormKind(NormKind kind)
{
    return static_cast<int>(kind) >= 0 && static_cast<int>(kind) < static_cast<int>(NormKind::COUNT);
}

/**
 * @brief Validates if a bias enum is valid
 * @param bias The bias enum to validate
 * @return true if valid, false otherwise
 */
inline bool IsValidBiasEnum(NormBiasEnum bias)
{
    return static_cast<int>(bias) >= 0 && static_cast<int>(bias) < static_cast<int>(NormBiasEnum::COUNT);
}

/**
 * @brief Validates if a fused add enum is valid
 * @param add The fused add enum to validate
 * @return true if valid, false otherwise
 */
inline bool IsValidFusedAddEnum(FusedAddEnum add)
{
    return static_cast<int>(add) >= 0 && static_cast<int>(add) < static_cast<int>(FusedAddEnum::COUNT);
}

/**
 * @brief Validates if a fused quant enum is valid
 * @param quant The fused quant enum to validate
 * @return true if valid, false otherwise
 */
inline bool IsValidFusedQuantEnum(FusedQuantEnum quant)
{
    return static_cast<int>(quant) >= 0 && static_cast<int>(quant) < static_cast<int>(FusedQuantEnum::COUNT);
}

}  // namespace flashck