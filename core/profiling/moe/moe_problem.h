#pragma once

#include <sstream>

#include "core/profiling/problem_base.h"
#include "core/profiling/moe/moe_library.h"
#include "core/utils/dtype.h"

namespace flashck {

/**
 * @class TopKSoftmaxProblem
 * @brief Represents a TopK Softmax operation problem configuration for MoE routing
 *
 * TopK Softmax is a critical component in Mixture of Experts (MoE) models for expert routing:
 * - Selects the top-K experts for each input token based on routing probabilities
 * - Applies softmax normalization to the selected expert weights
 * - Ensures load balancing and sparsity in expert selection
 * 
 * Key Features:
 * - Configurable top-K selection for expert sparsity control
 * - Support for different data types (FP16, BF16, FP32)
 * - Hardware-optimized atomic operations for expert assignment
 * - Smooth quantization support for efficient inference
 * - Local expert masking for distributed MoE deployments
 */
class TopKSoftmaxProblem: public ProblemBase<TopKSoftmaxProblem> {
public:

    std::string GetTypeImpl()
    {
        return "TopKSoftmaxProblem";
    }

    std::string SerializeImpl()
    {
        std::ostringstream oss;
        oss << "TopKSoftmax_"
            << "tokens" << num_tokens_ << "_"
            << "experts" << num_experts_ << "_"
            << "hidden" << hidden_dim_ << "_"
            << "topk" << topk_ << "_"
            << "dtype" << DataTypeToString(input_dtype_) << "_"
            << "wtype" << DataTypeToString(weight_dtype_) << "_"
            << "itype" << DataTypeToString(index_dtype_) << "_"
            << "act" << GetActivationShortName(activation_) << "_"
            << (is_only_gate_ ? "gate" : "full") << "_"
            << (use_smooth_quant_ ? "squant" : "nosquant") << "_"
            << "atomic" << atomic_ << "_"
            << (local_expert_mask_ ? "localmask" : "globalmask");
        return oss.str();
    }

    // ====================== Problem Configuration ======================
    
    /// @brief Input token data type (typically FP16 or BF16 for efficiency)
    DataType input_dtype_;
    
    /// @brief Expert routing weight data type (typically FP16 or BF16)
    DataType weight_dtype_;
    
    /// @brief Expert index data type (INT32 for large expert counts)
    DataType index_dtype_ = DataType::INT32;

    /// @brief Number of input tokens to route to experts
    int64_t num_tokens_;
    
    /// @brief Total number of available experts in the MoE model
    int64_t num_experts_;
    
    /// @brief Hidden dimension size of input tokens
    int64_t hidden_dim_;
    
    /// @brief Number of top experts to select per token (sparsity parameter)
    int64_t topk_;

    /// @brief Memory stride for input token data
    int64_t input_stride_;
    
    /// @brief Memory stride for output routing data
    int64_t output_stride_;

    /// @brief Activation function used in expert gating (GELU, SiLU, etc.)
    ActivationType activation_;
    
    /// @brief Whether to use only gate weights (no up projection)
    bool is_only_gate_;
    
    /// @brief Enable smooth quantization for efficient inference
    bool use_smooth_quant_;
    
    /// @brief Atomic operation mode: 0=no atomic, 1=atomic FP16/BF16, 2=atomic FP32
    int atomic_;

    /// @brief Use local expert masking for distributed MoE setups
    bool local_expert_mask_;
    
};

/**
 * @class MoeGemmProblem  
 * @brief Represents a dual-stage MoE GEMM operation problem configuration
 *
 * MoE GEMM encompasses the core computation in Mixture of Experts models:
 * - Stage 0: Token-to-Intermediate transformation (input → gate/up projection)
 * - Stage 1: Intermediate-to-Output transformation (intermediate → down projection)
 * - Expert routing and load balancing across selected experts
 * - Activation function fusion between stages
 * 
 * Architecture Features:
 * - Dual-stage processing with different matrix dimensions per stage
 * - Expert-specific weight matrices and routing patterns
 * - Support for various activation functions (SwiGLU, GELU, ReLU)
 * - Hardware-optimized memory access patterns
 * - Load balancing and capacity factor management
 */
class MoeGemmProblem: public ProblemBase<MoeGemmProblem> {
public:

    std::string GetTypeImpl()
    {
        return "MoeGemmProblem";
    }

    std::string SerializeImpl()
    {
        std::ostringstream oss;
        oss << "MoeGemm_"
            << "m" << m_ << "_"
            << "n" << n_ << "_"
            << "k" << k_ << "_"
            << "inter" << intermediate_size_ << "_"
            << "experts" << num_experts_ << "_"
            << "topk" << top_k_ << "_"
            << "cap" << static_cast<int>(capacity_factor_ * 100) << "_"
            << "dtype" << DataTypeToString(input_dtype_) << "_"
            << "wtype" << DataTypeToString(weight_dtype_) << "_"
            << "otype" << DataTypeToString(output_dtype_) << "_"
            << "itype" << DataTypeToString(intermediate_dtype_) << "_"
            << "act" << GetActivationShortName(activation_) << "_"
            << (use_smooth_quant_ ? "squant" : "nosquant") << "_"
            << "permute" << static_cast<int>(weight_permute_);
        return oss.str();
    }

    // ====================== Stage 0: Token-to-Intermediate Configuration ======================
    
    /// @brief Number of input tokens (batch size * sequence length)
    int64_t m_;
    
    /// @brief Input feature dimension (hidden size)
    int64_t k_;
    
    /// @brief Intermediate dimension size (typically 4x or 8x hidden size)
    int64_t intermediate_size_;

    // ====================== Stage 1: Intermediate-to-Output Configuration ======================
    
    /// @brief Output feature dimension (typically same as input hidden size)
    int64_t n_;

    // ====================== Expert Configuration ======================
    
    /// @brief Total number of experts in the MoE model
    int64_t num_experts_;
    
    /// @brief Number of top experts selected per token
    int64_t top_k_;
    
    /// @brief Capacity factor for load balancing (1.0 = perfect balance)
    double capacity_factor_ = 1.25;

    // ====================== Data Type Configuration ======================
    
    /// @brief Input token data type
    DataType input_dtype_;
    
    /// @brief Expert weight data type
    DataType weight_dtype_;
    
    /// @brief Output data type
    DataType output_dtype_;
    
    /// @brief Intermediate computation data type
    DataType intermediate_dtype_;
    
    /// @brief Expert index data type
    DataType index_dtype_ = DataType::INT32;

    // ====================== Processing Configuration ======================
    
    /// @brief Activation function used between gate and up projections
    ActivationType activation_;
    
    /// @brief Enable smooth quantization for weights
    bool use_smooth_quant_ = false;
    
    /// @brief Weight permutation strategy for memory optimization
    MoeGemmWeightPermuteEnum weight_permute_ = MoeGemmWeightPermuteEnum::no_permute;

    // ====================== Memory Layout Configuration ======================
    
    /// @brief Input memory stride
    int64_t input_stride_;
    
    /// @brief Weight memory stride
    int64_t weight_stride_;
    
    /// @brief Output memory stride  
    int64_t output_stride_;
    
    /// @brief Intermediate memory stride
    int64_t intermediate_stride_;

    // ====================== Problem Kind Identification ======================
    
    /// @brief MoE GEMM operation kind for instance management
    MoeGemmKind kind_;
};

}  // namespace flashck