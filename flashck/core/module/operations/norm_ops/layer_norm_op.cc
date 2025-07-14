#include "flashck/core/module/operations/norm_ops/layer_norm_op.h"

FC_DECLARE_bool(FC_FORCE_PROFILING);
FC_DECLARE_string(FC_HOME_PATH);

namespace flashck {

template<typename T>
LayerNormOp<T>::LayerNormOp(Shape          normalized_shape,
                            NormBiasEnum   is_add_bias,
                            FusedAddEnum   fused_add,
                            FusedQuantEnum fused_quant):
    Operation("layer_norm"), is_add_bias_(is_add_bias), fused_add_(fused_add), fused_quant_(fused_quant)
{
    normalized_shape_ = normalized_shape;
}

// template<typename T>
// void LayerNormOp<T>::CheckParamShape(const Shape& x_shape, const Shape& param_shape, const std::string& param_name)
// {
//     FC_ENFORCE_LT(
//         normalized_shape.GetNumDim(),
//         x_shape.GetNumDim(),
//         Unavailable(
//             "Layernorm normalized_shape length must be smaller than the input. Current normalized_shape: {}, input
//             shape: {}", normalized_shape.ToString(), x_shape.ToString()));

//     FC_ENFORCE_GE(x_shape.GetNumDim(),
//                   2,
//                   Unimplemented("norm only supports 2D or higher input, runtime rank: {}", x_shape.GetNumDim()));

//     if (fused_add_ != FusedAddEnum::NO_ADD) {
//         FC_ENFORCE_EQ(
//             x_residual_shape.GetNumDim(),
//             x_shape.GetNumDim(),
//             Unimplemented("norm only supports 2D or higher input, runtime rank: {}", x_residual_shape.GetNumDim()));
//     }

//     if (param_name != "normalized" && param_shape.GetNumDim()) {
//         return;
//     }

//     if (!x_shape.GetLastDim().IsStatic()) {
//         FC_THROW(Unimplemented("layernorm requires reduction dim to be static. Current input shape: {}",
//                                x_shape.ToString()));
//     }

//     for (const auto& shape : param_shape.ToVector()) {
//         if (!shape.IsStatic()) {
//             FC_THROW(Unimplemented(
//                 "Layernorm {} shape must be immutable values. Current value: {}", param_name,
//                 param_shape.ToString()));
//         }
//     }

//     int64_t batch_ndims = x_shape.GetNumDim() - param_shape.GetNumDim();
//     for (int i = 0; i < param_shape.GetNumDim(); i++) {
//         if (param_shape[i].GetValues() != x_shape[batch_ndims + i].GetValues()) {
//             FC_THROW(Unavailable(
//                 "Layernorm {} shape must be broadcastable to the input shape. Current {} shape: {}, input shape: {}",
//                 param_name,
//                 param_shape.ToString(),
//                 x_shape.ToString()));
//         }
//     }
// }

// template<typename T>
// void LayerNormOp<T>::CheckParamValue()
// {
// }

// template<typename T>
// void LayerNormOp<T>::SanityCheck(Variable* x,
//                                  Variable* gamma,
//                                  Variable* beta,
//                                  Variable* x_bias,
//                                  Variable* x_residual,
//                                  Variable* smooth_scale,
//                                  Variable* y_residual,
//                                  Variable* y_scale)
// {
//     // TODO: Add more sanity checks
//     FC_ENFORCE_NOT_NULL(x, Unavailable("Input variable x is null"));
//     FC_ENFORCE_NOT_NULL(gamma, Unavailable("Gamma variable is null"));
//     FC_ENFORCE_NOT_NULL(beta, Unavailable("Beta variable is null"));
// }

template<typename T>
Shape LayerNormOp<T>::InferShape(Variable* x)
{
    return x->GetShape();
}

template<typename T>
Variable* LayerNormOp<T>::operator()(Variable*   x,
                                     Variable*   gamma,
                                     Variable*   beta,
                                     Variable*   x_bias,
                                     Variable*   x_residual,
                                     Variable*   smooth_scale,
                                     Variable*   y_residual,
                                     Variable*   y_scale,
                                     const float eps)
{
    // SanityCheck(x, gamma, beta, x_bias, x_residual, smooth_scale, y_residual, y_scale);

    eps_                             = eps;
    std::vector<Variable*> input_var = {x, gamma, beta, x_bias, x_residual, smooth_scale, y_residual, y_scale};
    for (auto var : input_var) {
        if (var != nullptr) {
            input_var_.push_back(var);
        }
    }

    Shape output_shape    = InferShape(x);
    auto  max_output_size = std::get<1>(output_shape.GetElementSizeTuple());
    output_var_           = {new Variable("layer_norm_output", max_output_size, CppTypeToDataType<T>::Type())};
    output_var_[0]->SetShape(output_shape);

    std::vector<Node*> parents_node;
    for (auto var : input_var_) {
        if (var != nullptr) {
            parents_node.push_back(var);
        }
    }
    this->SetParentsNode(parents_node);
    this->SetChildrenNode({static_cast<Node*>(output_var_[0])});

    return output_var_[0];
}

template<typename T>
void LayerNormOp<T>::ExtractRunningInfo(const ProfilingStrategy& profiling_strategy, const int step_value)
{
    FC_ENFORCE_EQ(
        normalized_shape_.GetNumDim(),
        1,
        Unimplemented("For profiling, normalized_shape must be 1D, but got {}", normalized_shape_.GetNumDim()));

    auto broadcast_shape_func = [&](Variable* x) {
        auto x_shape_vec = x->GetShape().ToVector();
        DDim dim_value{1};
        std::for_each(x_shape_vec.begin(), x_shape_vec.end() - 1, [&](const DDim& dim) {
            dim_value = dim_value * dim.GetValues();
        });

        return std::make_tuple(dim_value.GetValues()[0], dim_value.GetValues()[1]);
    };

    int64_t m_max = std::get<1>(broadcast_shape_func(input_var_[0]));
    int64_t m_min = std::get<0>(broadcast_shape_func(input_var_[0]));

    int64_t n = input_var_[0]->GetShape().GetLastDim().GetValues()[0];

    std::map<std::string, std::vector<int64_t>> shape_values_map = {
        {"m", {m_min, m_max}},
        {"n", {n}},
    };

    if (profiling_strategy == ProfilingStrategy::kMax) {
        std::map<std::string, std::vector<int64_t>> max_values = {
            {"m", {m_max}},
            {"n", {n}},
        };

        running_infos_[GenWorkLoad(max_values)] = RunningItem{
            .running_cond_  = GenWorkLoad(shape_values_map),
            .instance_name_ = "",
            .perf_result_   = PerfResult(),
        };
    }
    else if (profiling_strategy == ProfilingStrategy::kMin) {
        std::map<std::string, std::vector<int64_t>> min_values = {
            {"m", {m_min}},
            {"n", {n}},
        };

        running_infos_[GenWorkLoad(min_values)] = RunningItem{
            .running_cond_  = GenWorkLoad(shape_values_map),
            .instance_name_ = "",
            .perf_result_   = PerfResult(),
        };
    }
    else if (profiling_strategy == ProfilingStrategy::kIteration) {
        std::map<std::string, std::vector<int64_t>> iter_values_map;
        for (int64_t m = m_min; m <= m_max; m += step_value) {
            iter_values_map["m"].push_back(m);
        }

        iter_values_map["n"] = {n};
        iter_values_map["n"].resize(iter_values_map["n"].size());
        std::fill(iter_values_map["n"].begin(), iter_values_map["n"].end(), iter_values_map["N"].front());

        std::vector<std::map<std::string, std::vector<int64_t>>> iter_values_vec;
        for (size_t i = 0; i < iter_values_map["m"].size(); i++) {
            std::map<std::string, std::vector<int64_t>> iter_value{};
            for (const auto& [name, values] : iter_values_map) {
                iter_value[name].push_back(values[i]);
            }
            iter_values_vec.push_back(iter_value);
        }

        for (const auto& iter_value : iter_values_vec) {
            running_infos_[GenWorkLoad(iter_value)] = RunningItem{
                .running_cond_  = GenWorkLoad(iter_value),
                .instance_name_ = "",
                .perf_result_   = PerfResult(),
            };
        }
    }

    else {
        FC_THROW(Unimplemented("{}", "norm only supports MIN or MAX dynamic profiling"));
    }
}

template<typename T>
void LayerNormOp<T>::IsBuildProfilingEngine()
{
    for (const auto& [profiling_workload, running_info] : running_infos_) {
        std::map<std::string, int> profiling_key_map = ExtractWorkLoad(profiling_workload);
        auto                       instance_data     = InstanceData{
            Environment{.device_name_ = GetDeviceName(),
                                                  .rocm_version_ = ProfilingEngine::GetInstance()->GetCompiler()->GetROCmVersion()},
            Setting{},
            CodeGenKind::Norm,
            NormProblem{.x_dtype_            = CppTypeToDataType<T>::Type(),
                                                  .y_dtype_            = CppTypeToDataType<T>::Type(),
                                                  .smooth_scale_dtype_ = DataType::FLOAT32,
                                                  .y_scale_dtype_      = DataType::FLOAT32,
                                                  .m_                  = profiling_key_map.at("m"),
                                                  .n_                  = profiling_key_map.at("n"),
                                                  .kind_               = op_kind_,
                                                  .is_add_bias_        = is_add_bias_,
                                                  .fused_add_          = fused_add_,
                                                  .fused_quant_        = fused_quant_},
        };

        auto [instance_name, perf_result] = ProfilingEngine::GetInstance()->GetProfilingDB()->Query(instance_data);

        if (!instance_name.empty() && perf_result.IsValid() && !FLAGS_FC_FORCE_PROFILING) {
            LOG(INFO) << "Load profiling result for layer norm from database, " << instance_name
                      << " split_k: " << perf_result.split_k_ << ", latency: " << perf_result.latency_ << "ms, "
                      << "tflops: " << perf_result.tflops_ << "GB, " << "throughput: " << perf_result.bandwidth_
                      << "GB/s";

            // Update the running info with the found instance
            running_infos_[profiling_workload].instance_name_ = instance_name;
            running_infos_[profiling_workload].perf_result_   = perf_result;
        }
        else {
            // cache miss - we will have to generate and build profilers
            LOG(INFO) << "No profiling result found for layer norm, instance_name: " << instance_name;
        }
    }
}

template<typename T>
std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
LayerNormOp<T>::CodeGenForTuning(const ProfilingStrategy& profiling_strategy)
{
    KernelKey kernel_key(SourceType::TILE, DataLayout::ALL_LAYOUT, CppTypeToDataType<T>::Type());
    register_kernel_ptr_ = KernelFactory::Instance().SelectKernel(GetNormKindName(op_kind_), kernel_key);

    ExtractRunningInfo(profiling_strategy);

    if (!FLAGS_FC_FORCE_PROFILING) {
        IsBuildProfilingEngine();
    }
    else {
        LOG(INFO) << "Forced to use cache, skip building profiling engine for layernorm";
        return {};
    }

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> generated_profilers;

    for (const auto& [workload_key, running_info] : running_infos_) {
        std::map<std::string, int> profiling_key_map = ExtractWorkLoad(workload_key);
        auto                       problem           = NormProblem{.x_dtype_            = CppTypeToDataType<T>::Type(),
                                                                   .y_dtype_            = CppTypeToDataType<T>::Type(),
                                                                   .smooth_scale_dtype_ = DataType::FLOAT32,
                                                                   .y_scale_dtype_      = DataType::FLOAT32,
                                                                   .m_                  = profiling_key_map.at("m"),
                                                                   .n_                  = profiling_key_map.at("n"),
                                                                   .kind_               = op_kind_,
                                                                   .is_add_bias_        = is_add_bias_,
                                                                   .fused_add_          = fused_add_,
                                                                   .fused_quant_        = fused_quant_};

        auto norm_instance_map = NormEmitter::GetInstance()->GetInstanceMap(problem);

        if (norm_instance_map.size() == 0) {
            FC_THROW(Fatal("No layernorm op instances were generated for layernorm"));
        }

        if (!running_info.IsInstanceExist()) {

            generated_profilers = register_kernel_ptr_->CodeGenForTuning(
                context_ptr_->GetName(), GetNormKindName(op_kind_), norm_instance_map);
        }
        else {
            LOG(INFO) << "layer norm already exists, not profile";
        }
    }

    return generated_profilers;
}

// template<typename T>
// std::vector<std::string> LayerNormOp<T>::GetTuningCmd(const std::string&          profiler_prefix,
//                                                       const std::string&          profiler_filename,
//                                                       const std::vector<int64_t>& input_shape)
// {
//     std::filesystem::path exe_path = std::filesystem::path(profiler_prefix) / profiler_filename;

//     FileManager::Instance().CreateDirectoryIfNotExists(exe_path.parent_path());
//     FileManager::Instance().CheckWithRetries(exe_path, 3, 5);

//     std::vector<std::string> cmd = {exe_path.string(),
//                                     "-m=" + std::to_string(input_shape[0]),
//                                     "-n=" + std::to_string(input_shape[1]),
//                                     "-e=" + std::to_string(eps_)};

//     return cmd;
// }

// template<typename T>
// void LayerNormOp<T>::ProfileSingleWorkload(const std::string&                         profiler_prefix,
//                                            const std::string&                         workload,
//                                            const std::shared_ptr<GPUProfilingRunner>& profiler_runner_ptr,
//                                            bool                                       force_cache)
// {
//     std::vector<std::string> kernel_instance_map_key = GetKeyVector(kernel_instance_map_);

//     std::string exec_entry_sha1 = SHA1ToHexString(workload);
//     auto        query           = NormQueryEntry(DataTypeToShortString(CppTypeToDataType<T>::Type()),
//                                 DataTypeToShortString(CppTypeToDataType<T>::Type()),
//                                 DataTypeToShortString(DataType::FLOAT32),
//                                 DataTypeToShortString(DataType::FLOAT32),
//                                 g_norm_operation_kind_names_map.at(op_kind_),
//                                 Target::Instance()->GetTargetDeviceName(),
//                                 g_short_tensor_operation_names_map.at(epilogue_op_),
//                                 exec_entry_sha1,
//                                 g_fused_add_enum_str_map.at(fused_add_),
//                                 g_fused_quant_enum_str_map.at(fused_quant_));

//     auto cache_value = Target::Instance()->QueryProfileCache(CodeGenKind::Norm, query);

//     if (cache_value == std::make_tuple("null", -1) && force_cache) {
//         LOG(WARNING) << "force_cache is enabled but we could not find the following cache available on device. "
//                      << "op_name: " << op_name_ << " exec_entry_sha1: " << exec_entry_sha1;
//     }

//     for (const auto& kernel_config_name : kernel_instance_map_key) {
//         auto GenCallback = [&] {
//             auto process_result_callback =
//                 [&](const std::vector<ProfileResult>&           result,
//                     const std::shared_ptr<ProfilerPostprocess>& postprocessing_delegate_ptr) {
//                     postprocessing_delegate_ptr->AddInstance(
//                         result, CodeGenKind::Norm, GetAttrsMap(), kernel_config_name, workload);
//                 };
//             return process_result_callback;
//         };

//         auto                     input_shape = ExtractWorkLoad(workload);
//         std::vector<std::string> command     = GetTuningCmd(profiler_prefix, kernel_config_name, input_shape);

//         LOG(INFO) << "profile command: " << JoinStrings(command);

//         profiler_runner_ptr->Push(command, GenCallback());
//     }
// }

// template<typename T>
// void LayerNormOp<T>::Tuning(const std::shared_ptr<GPUProfilingRunner>& profiler_runner_ptr,
//                             const std::string&                         folder_name)
// {
//     std::filesystem::path profiler_prefix =
//         std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / context_ptr_->GetName() / "profiling" / op_name_;
//     FileManager::Instance().CreateDirectoryIfNotExists(profiler_prefix);

//     for (const auto& workload : profiling_key_) {

//         if (exec_path_[workload]->algo_ == "") {
//             if (kernel_instance_map_.size() == 0) {
//                 kernel_instance_map_ = register_kernel_ptr_->Init(op_kind_, epilogue_op_);
//             }
//             ProfileSingleWorkload(profiler_prefix, workload, profiler_runner_ptr, FLAGS_FC_FORCE_PROFILER_CACHE);
//         }
//         else {
//             LOG(INFO) << op_name_ << " from cache, not profile";
//         }
//     }
// }

// template<typename T>
// std::string LayerNormOp<T>::GenOpFunction()
// {
//     return register_kernel_ptr_->GenKernelFunction(GetName(), context_ptr_->GetName(), GetAttrsMap());
// }

// template<typename T>
// void LayerNormOp<T>::Forward()
// {
//     Variable* x = GetParentNode(0);

//     T* x_ptr      = (T*)x->GetValue();
//     T* gamma_ptr  = (T*)GetParentNode(1)->GetValue();
//     T* beta_ptr   = (T*)GetParentNode(2)->GetValue();
//     T* x_bias_ptr = is_add_bias_ != NormBiasEnum::NO_BIAS ? (T*)GetParentNode(3)->GetValue() : nullptr;
//     T* x_residual_ptr =
//         fused_add_ != FusedAddEnum::NO_ADD ? (T*)GetParentNode(3 + (x_bias_ptr != nullptr))->GetValue() : nullptr;
//     T* smooth_scale_ptr = fused_quant_ == FusedQuantEnum::SMOOTH_DYNAMIC_QUANT ?
//                               (T*)GetParentNode(3 + (x_residual_ptr != nullptr) + (x_bias_ptr !=
//                               nullptr))->GetValue() : nullptr;
//     T* y_residual_ptr =
//         fused_add_ == FusedAddEnum::PRE_ADD_STORE ?
//             (T*)GetParentNode(3 + (x_bias_ptr != nullptr) + (smooth_scale_ptr != nullptr) + (x_residual_ptr !=
//             nullptr))
//                 ->GetValue() :
//             nullptr;
//     T* y_scale_ptr = fused_quant_ != FusedQuantEnum::NO_SWEEP ?
//                          (T*)GetParentNode(3 + (x_bias_ptr != nullptr) + (y_residual_ptr != nullptr)
//                                            + (smooth_scale_ptr != nullptr) + (x_residual_ptr != nullptr))
//                              ->GetValue() :
//                          nullptr;

//     T* y_ptr = (T*)GetChildNode(0)->GetValue();

//     if (!context_ptr_->IsBuilt()) {
//         return;
//     }

//     Shape c_shape = InferShape(GetParentNode(0));
//     output_var_[0]->SetShape(c_shape);  // must update actual output shape

//     auto broadcast_shape_func = [&](Variable* x) {
//         auto x_shape_vec = x->GetShape().ToVector();
//         int  dim0_value  = 1;
//         std::for_each(
//             x_shape_vec.begin(), x_shape_vec.end() - 1, [&](const DDim& dim) { dim0_value *= dim.GetValues()[0]; });

//         return dim0_value;
//     };

//     VLOG(1) << "norm " << this->op_name_ << ", out shape: " << c_shape.ToString();

//     // PrintToScreen(x_ptr, 3, "[" + this->op_name_ + "]" + "x_ptr");
//     // PrintToScreen(gamma_ptr, 3, "[" + this->op_name_ + "]" + "gamma_ptr");
//     // PrintToScreen(beta_ptr, 3, "[" + this->op_name_ + "]" + "beta_ptr");
//     // PrintToScreen(x_bias_ptr, 3, "[" + this->op_name_ + "]" + "x_bias_ptr");
//     // PrintToScreen(x_residual_ptr, 3, "[" + this->op_name_ + "]" + "x_residual_ptr");
//     // PrintToScreen(smooth_scale_ptr, 3, "[" + this->op_name_ + "]" + "smooth_scale_ptr");
//     // PrintToScreen(y_residual_ptr, 3, "[" + this->op_name_ + "]" + "y_residual_ptr");
//     // PrintToScreen(y_scale_ptr, 3, "[" + this->op_name_ + "]" + "y_scale_ptr");

//     NormKernelArgs layer_norm_args;
//     layer_norm_args.x_ptr_            = x_ptr;
//     layer_norm_args.x_residual_ptr_   = x_residual_ptr;
//     layer_norm_args.smooth_scale_ptr_ = smooth_scale_ptr;
//     layer_norm_args.x_bias_ptr_       = x_bias_ptr;
//     layer_norm_args.gamma_ptr_        = gamma_ptr;
//     layer_norm_args.beta_ptr_         = beta_ptr;
//     layer_norm_args.y_ptr_            = y_ptr;
//     layer_norm_args.y_residual_ptr_   = y_residual_ptr;
//     layer_norm_args.y_scale_ptr_      = y_scale_ptr;

//     layer_norm_args.x_dim_0_ = broadcast_shape_func(x);
//     layer_norm_args.x_dim_1_ = x->GetShape().GetLastDim().GetValues()[0];
//     layer_norm_args.eps_     = eps_;
//     layer_norm_args.stream_  = context_ptr_->GetStream();

//     register_kernel_ptr_->KernelLauncher(GetName(), layer_norm_args);

//     // PrintToScreen(y_ptr, 3, "[" + this->op_name_ + "]" + "y_ptr");
//     // ResultChecker(y_ptr, std::get<0>(c_shape.GetElementSizeTuple()), "[" + this->op_name_ + "]" + "y_ptr");
// }

template class LayerNormOp<float>;
template class LayerNormOp<_Float16>;
template class LayerNormOp<ushort>;

}  // namespace flashck