#pragma once

#include <string>
#include <vector>

#include "flashck/core/graph/node.h"

#include "flashck/core/module/kernels/kernel_factory.h"
#include "flashck/core/module/kernels/norm_kernels/layer_norm_kernel.h"

namespace flashck {

template<typename T>
class LayerNormOp: public Operation {
public:
    LayerNormOp(Shape          normalized_shape,
                NormBiasEnum   is_add_bias = NormBiasEnum::NO_BIAS,
                FusedAddEnum   fused_add   = FusedAddEnum::NO_ADD,
                FusedQuantEnum fused_quant = FusedQuantEnum::NO_SWEEP);

    Shape InferShape(Variable* x);

    Variable* operator()(Variable*   x,
                         Variable*   gamma,
                         Variable*   beta,
                         Variable*   x_bias,
                         Variable*   x_residual,
                         Variable*   smooth_scale,
                         Variable*   y_residual,
                         Variable*   y_scale,
                         const float eps = 1e-5);

    void ExtractRunningInfo(const ProfilingStrategy& dynamic_profiling_strategy = ProfilingStrategy::kMax,
                            const int                step_value                 = 1);

    void IsBuildProfilingEngine();

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    CodeGenForTuning(const ProfilingStrategy& dynamic_profiling_strategy = ProfilingStrategy::kMax) override;

    std::vector<std::string> GetTuningCmd(const std::string&                profiling_file_prefix,
                                          const std::string&                profiler_filename,
                                          const std::map<std::string, int>& profiling_key_map);

    void TuningSingleWorkload(const std::string&  profiling_file_prefix,
                              const std::string&  profiling_workload,
                              GPUProfilingRunner& profiling_runner);

    void Tuning(GPUProfilingRunner& profiling_runner, const std::string& folder_name) override;

    // std::string GenOpFunction() override;

    // void Forward() override;

    NormKind       op_kind_     = NormKind::LayerNorm;
    NormBiasEnum   is_add_bias_ = NormBiasEnum::NO_BIAS;
    FusedAddEnum   fused_add_   = FusedAddEnum::NO_ADD;
    FusedQuantEnum fused_quant_ = FusedQuantEnum::NO_SWEEP;

    Shape normalized_shape_;

    float eps_;

    std::map<std::string, NormCodeGen> norm_instance_map_;
    std::map<std::string, RunningItem> running_infos_;

    std::shared_ptr<Kernel> register_kernel_ptr_;
};
}  // namespace flashck