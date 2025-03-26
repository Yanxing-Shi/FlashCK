#pragma once

#include <string>
#include <vector>

#include "lightinfer/core/graph/node.h"

#include "lightinfer/core/module/kernels/kernel_factory.h"
#include "lightinfer/core/module/kernels/norm_kernels/layer_norm_kernel.h"
#include "lightinfer/core/profiler/library.h"

namespace lightinfer {

template<typename T>
class LayerNormOp: public Operation {
public:
    LayerNormOp(Shape          normalized_shape,
                NormBiasEnum   is_add_bias = NormBiasEnum::NO_BIAS,
                FusedAddEnum   fused_add   = FusedAddEnum::NO_ADD,
                FusedQuantEnum fused_quant = FusedQuantEnum::NO_SWEEP,
                std::string    op_name     = "layer_norm");

    void CheckParamShape(const Shape& x_shape, const Shape& param_shape, const std::string& param_name);

    void CheckShape(const Shape& x_shape,
                    const Shape& gamma_shape,
                    const Shape& beta_shape,
                    const Shape& x_bias_shape,
                    const Shape& x_residual_shape,
                    const Shape& smooth_scale_shape,
                    const Shape& y_residual_shape,
                    const Shape& y_scale_shape,
                    const Shape& normalized_shape);

    std::vector<Shape> GetInputShape(Variable* x,
                                     Variable* gamma,
                                     Variable* beta,
                                     Variable* x_bias,
                                     Variable* x_residual,
                                     Variable* smooth_scale,
                                     Variable* y_residual,
                                     Variable* y_scale);

    void SanityCheck(Variable* x,
                     Variable* gamma,
                     Variable* beta,
                     Variable* x_bias,
                     Variable* x_residual,
                     Variable* smooth_scale,
                     Variable* y_residual,
                     Variable* y_scale);

    Shape InferShape(Variable* x);

    Variable* operator()(Variable*   x,
                         Variable*   gamma,
                         Variable*   beta,
                         Variable*   x_bias,
                         Variable*   x_residual,
                         Variable*   smooth_scale,
                         Variable*   y_residual,
                         Variable*   y_scale,
                         Shape       normalized_shape,
                         const float eps = 1e-5);

    std::vector<int64_t> InvertExecKey(const std::string& key);

    std::string GenExecKey(const std::map<std::string, std::vector<int64_t>>& name_value_mapping);

    void ExtractExecPath(const DynamicProfileStrategy& dynamic_profiling_strategy = DynamicProfileStrategy::MAX,
                         const int                     step_value                 = 1);

    void IfShouldBuildProfiler(const std::vector<std::string>& workloads);

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenOpProfiler(const DynamicProfileStrategy& dynamic_profiling_strategy = DynamicProfileStrategy::MAX) override;

    std::vector<std::string> GenOpProfileCmd(const std::string&          profiler_prefix,
                                             const std::string&          profiler_filename,
                                             const std::vector<int64_t>& input_shape);

    void ProfileSingleWorkload(const std::string&                        profiler_prefix,
                               const std::string&                        workload,
                               const std::shared_ptr<GPUProfilerRunner>& profiler_runner_ptr,
                               bool                                      force_cache);

    void Profile(const std::shared_ptr<GPUProfilerRunner>& profiler_runner_ptr,
                 const std::string&                        folder_name) override;

    std::string GenOpFunction() override;

    std::unordered_map<std::string, std::any> GetAttrsMap();

    void Forward() override;

    std::string op_name_ = "layer_norm";

    NormOperationKind op_kind_     = NormOperationKind::LayerNorm;
    TensorOperation   epilogue_op_ = TensorOperation::PassThrough;

    Shape normalized_shape_;
    Shape default_normalized_shape_;

    NormBiasEnum   is_add_bias_ = NormBiasEnum::NO_BIAS;
    FusedAddEnum   fused_add_   = FusedAddEnum::NO_ADD;
    FusedQuantEnum fused_quant_ = FusedQuantEnum::NO_SWEEP;

    float eps_;

    int64_t x_stride_  = -1;
    int64_t xr_stride_ = -1;
    int64_t y_stride_  = -1;
    int64_t yr_stride_ = -1;

    std::map<std::string, std::shared_ptr<ExecItem>> exec_path_;
    std::vector<std::string>                         exec_key_;

    std::shared_ptr<Kernel> register_kernel_ptr_;

    std::map<std::string, std::shared_ptr<void>> kernel_instance_map_;

    std::vector<Variable*> input_var_;
    std::vector<Variable*> output_var_;
};
}  // namespace lightinfer