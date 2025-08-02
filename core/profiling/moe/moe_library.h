#pragma once

#include <string>
#include <unordered_map>

namespace flashck{

/**
 * @enum MoeGemmWeightPermuteEnum
 * @brief Defines weight permutation strategies for MoE GEMM operations
 * 
 * Weight permutation is crucial for optimal memory access patterns in MoE workloads:
 * - Controls how expert weights are organized in memory
 * - Affects cache efficiency and memory bandwidth utilization
 * - Impacts expert routing and load balancing performance
 */
enum class MoeGemmWeightPermuteEnum
{
    no_permute          = 0,  ///< No permutation: standard weight layout
    b_nr_kr_kw_nw_kv    = 1,  ///< Wave-flattened layout: optimized for GPU warp access patterns
    b_nr_kr_waveflatten = b_nr_kr_kw_nw_kv,  ///< Alias for wave-flattened layout
};

/**
 * @enum ActivationType
 * @brief Supported activation functions for MoE GEMM operations
 * 
 * Activation functions are fused into the MoE computation pipeline for efficiency:
 * - GELU: Gaussian Error Linear Unit, commonly used in transformer models
 * - SiLU/Swish: Sigmoid Linear Unit, used in SwiGLU gates for MoE models
 * - Each activation has specific hardware optimization strategies
 */
enum class ActivationType{
    Gelu = 0,  ///< Gaussian Error Linear Unit activation
    Silu = 1   ///< Sigmoid Linear Unit (Swish) activation
};

/**
 * @struct ActivationTypeInfo
 * @brief Metadata and implementation details for activation functions
 * 
 * Contains:
 * - Human-readable name for logging and configuration
 * - Template tag for code generation 
 * - Short name for instance naming and performance tracking
 */
struct ActivationTypeInfo{
    std::string name_;       ///< Full activation name (e.g., "Gelu", "Silu")
    std::string tag_;        ///< Template implementation tag for code generation
    std::string short_name_; ///< Abbreviated name for instance identification
};

/**
 * @brief Activation type information mapping
 * 
 * Maps each ActivationType enum to its corresponding implementation details:
 * - GELU uses FastGeluAsm for hardware-optimized assembly implementation
 * - SiLU uses standard element-wise Silu implementation
 * - Short names enable compact instance identification in profiling
 */
std::unordered_map<ActivationType, ActivationTypeInfo> activation_type_info_map = {
    {ActivationType::Gelu, {"Gelu", "ck_tile::element_wise::FastGeluAsm{}", "Ge"}},
    {ActivationType::Silu, {"Silu", "ck_tile::element_wise::Silu{}", "Si"}}
};

/**
 * @brief Get activation function template tag for code generation
 * @param act Activation type to get tag for
 * @return Template tag string for the specified activation, or "unknown" if not found
 * 
 * Used during kernel code generation to insert the appropriate activation function
 * implementation into the fused MoE GEMM pipeline.
 */
inline std::string GetActivationTag(ActivationType act)
{
    auto it = activation_type_info_map.find(act);
    return it != activation_type_info_map.end() ? it->second.tag_ : "unknown";
}

/**
 * @brief Get activation function name for logging and configuration
 * @param act Activation type to get name for
 * @return Human-readable name string, or "unknown" if not found
 */
inline std::string GetActivationName(ActivationType act)
{
    auto it = activation_type_info_map.find(act);
    return it != activation_type_info_map.end() ? it->second.name_ : "unknown";
}

/**
 * @brief Get activation function short name for instance identification
 * @param act Activation type to get short name for
 * @return Abbreviated name string, or "Uk" (unknown) if not found
 */
inline std::string GetActivationShortName(ActivationType act)
{
    auto it = activation_type_info_map.find(act);
    return it != activation_type_info_map.end() ? it->second.short_name_ : "Uk";
}

/**
 * @brief Convert activation string to enum
 * @param act_str String representation of activation (case-insensitive)
 * @return Corresponding ActivationType enum
 * @throws std::invalid_argument if activation string is not recognized
 */
inline ActivationType GetActivationTypeFromString(const std::string& act_str)
{
    std::string lower_str = act_str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);
    
    if (lower_str == "gelu") return ActivationType::Gelu;
    if (lower_str == "silu" || lower_str == "swish") return ActivationType::Silu;
    
    throw std::invalid_argument("Unknown activation type: " + act_str);
}

/**
 * @brief Get weight permutation string representation
 * @param permute Weight permutation enum
 * @return String representation of the permutation strategy
 */
inline std::string GetWeightPermuteString(MoeGemmWeightPermuteEnum permute)
{
    switch (permute) {
        case MoeGemmWeightPermuteEnum::no_permute: return "no_permute";
        case MoeGemmWeightPermuteEnum::b_nr_kr_kw_nw_kv: return "b_nr_kr_waveflatten";
        default: return "unknown";
    }
}

} // namespace flashck