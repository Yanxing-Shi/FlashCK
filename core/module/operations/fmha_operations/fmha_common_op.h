#pragma once

#include <string>
#include <vector>

#include "core/graph/node.h"
#include "core/module/kernels/kernel_factory.h"
#include "core/module/kernels/norm_kernels/layer_norm_kernel.h"

namespace flashck {

/**
 * @brief LayerNorm operation with support for bias, residual add, and quantization
 * @tparam T Data type (float, _Float16, ushort)
 */
template<typename T>
class LayerNormOp: public Operation {
public:
    /// @brief Constructor with normalization configuration
    /// @param normalized_shape Shape of the normalized dimensions
    /// @param is_add_bias Enable bias addition
    /// @param fused_add Enable fused residual addition
    /// @param fused_quant Enable quantization
    LayerNormOp(Shape          normalized_shape,
                NormBiasEnum   is_add_bias = NormBiasEnum::NO_BIAS,
                FusedAddEnum   fused_add   = FusedAddEnum::NO_ADD,
                FusedQuantEnum fused_quant = FusedQuantEnum::NO_SWEEP);

    /// @brief Infer output shape from input
    Shape InferShape(Variable* x);

    /// @brief Execute layer normalization operation
    /// @param x Input tensor
    /// @param gamma Scale parameter (required)
    /// @param beta Bias parameter (required)
    /// @param x_bias Input bias (optional)
    /// @param x_residual Residual tensor for addition (optional)
    /// @param smooth_scale Smooth quantization scale (optional)
    /// @param y_residual Output residual storage (optional)
    /// @param y_scale Output quantization scale (optional)
    /// @param eps Epsilon for numerical stability
    /// @return Output variable
    Variable* operator()(Variable*   x,
                         Variable*   gamma,
                         Variable*   beta,
                         Variable*   x_bias       = nullptr,
                         Variable*   x_residual   = nullptr,
                         Variable*   smooth_scale = nullptr,
                         Variable*   y_residual   = nullptr,
                         Variable*   y_scale      = nullptr,
                         const float eps          = 1e-5);

    /// @brief Extract profiling configurations for different strategies
    void ExtractRunningInfo(const ProfilingStrategy& profiling_strategy = ProfilingStrategy::kMax,
                            const int                step_value         = 1);

    /// @brief Check and load existing profiling results from database
    void IsBuildProfilingEngine();

    /// @brief Generate tuning code for performance profiling
    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    CodeGenForTuning(const ProfilingStrategy& profiling_strategy = ProfilingStrategy::kMax) override;

    /// @brief Get profiling command for specific workload
    std::vector<std::string> GetTuningCmd(const std::string&                profiling_file_prefix,
                                          const std::string&                profiler_filename,
                                          const std::map<std::string, int>& profiling_key_map);

    /// @brief Execute profiling for single workload configuration
    void TuningSingleWorkload(const std::string&  profiling_file_prefix,
                              const std::string&  profiling_workload,
                              GPUProfilingRunner& profiling_runner);

    /// @brief Run performance tuning across all workloads
    void Tuning(GPUProfilingRunner& profiling_runner, const std::string& folder_name) override;

    /// @brief Generate optimized runtime kernel code
    std::string CodeGenForRunning() override;

    /// @brief Execute forward pass computation
    void Forward() override;

private:
    // Configuration
    NormKind       op_kind_     = NormKind::LayerNorm;       ///< Operation type
    NormBiasEnum   is_add_bias_ = NormBiasEnum::NO_BIAS;     ///< Bias addition flag
    FusedAddEnum   fused_add_   = FusedAddEnum::NO_ADD;      ///< Residual addition flag
    FusedQuantEnum fused_quant_ = FusedQuantEnum::NO_SWEEP;  ///< Quantization flag

    Shape normalized_shape_;  ///< Normalization dimensions
    float eps_;               ///< Numerical stability epsilon

    // Profiling and code generation
    std::map<std::string, NormCodeGen> instance_map_;         ///< Generated kernel instances
    std::map<std::string, RunningItem> running_infos_;        ///< Profiling configurations
    std::shared_ptr<Kernel>            register_kernel_ptr_;  ///< Registered kernel
};
}  // namespace flashck