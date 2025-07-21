#include "flashck/core/module/operations/norm_ops/rms_norm_op.h"

#include "flashck/core/profiling/compiler.h"
#include "flashck/core/profiling/gpu_profiling_runner.h"
#include "flashck/core/profiling/tile/norm/norm_emitter.h"

FC_DECLARE_bool(FC_FORCE_PROFILING);  ///< Force re-profiling flag
FC_DECLARE_string(FC_HOME_PATH);      ///< Base path for generated files

namespace flashck {

template<typename T>
RMSNormOp<T>::RMSNormOp(Shape normalized_shape, FusedAddEnum fused_add, FusedQuantEnum fused_quant):
    Operation("rms_norm"), fused_add_(fused_add), fused_quant_(fused_quant)
{
    normalized_shape_ = normalized_shape;

    // Initialize kernel and extract profiling configurations
    KernelKey kernel_key(SourceType::TILE, DataLayout::ALL_LAYOUT, CppTypeToDataType<T>::value);
    register_kernel_ptr_ = KernelFactory::Instance().SelectKernel(GetNormKindName(op_kind_), kernel_key);
}

template<typename T>
Shape RMSNormOp<T>::InferShape(Variable* x)
{
    return x->GetShape();
}

template<typename T>
Variable* RMSNormOp<T>::operator()(Variable*   x,
                                   Variable*   gamma,
                                   Variable*   x_residual,
                                   Variable*   smooth_scale,
                                   Variable*   y_residual,
                                   Variable*   y_scale,
                                   const float eps)
{
    eps_ = eps;

    // Collect non-null input variables
    std::vector<Variable*> input_var = {x, gamma, x_residual, smooth_scale, y_residual, y_scale};
    for (auto var : input_var) {
        if (var != nullptr) {
            input_var_.push_back(var);
        }
    }

    // Create output variable
    Shape output_shape    = InferShape(x);
    auto  max_output_size = std::get<1>(output_shape.GetElementSizeTuple());
    output_var_ = {new Variable("rms_norm_output", max_output_size, CppTypeToDataType<T>::value, VarType::FixedVar)};
    output_var_[0]->SetShape(output_shape);

    // Build computation graph connections
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
void RMSNormOp<T>::ExtractRunningInfo(const ProfilingStrategy& profiling_strategy, const int step_value)
{
    FC_ENFORCE_EQ(
        normalized_shape_.GetNumDim(),
        1,
        Unimplemented("For profiling, normalized_shape must be 1D, but got {}", normalized_shape_.GetNumDim()));

    // Calculate tensor dimensions for profiling
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
    int64_t n     = input_var_[0]->GetShape().GetLastDim().GetValues()[0];

    std::map<std::string, std::vector<int64_t>> shape_values_map = {
        {"m", {m_min, m_max}},
        {"n", {n}},
    };

    // Generate profiling configurations based on strategy
    if (profiling_strategy == ProfilingStrategy::kMax) {
        std::map<std::string, std::vector<int64_t>> max_values = {
            {"m", {m_max}},
            {"n", {n}},
        };
        running_infos_[GenWorkLoad(max_values)] =
            RunningItem(GenWorkLoad(max_values), GenWorkLoad(shape_values_map), PerfResult());
    }
    else if (profiling_strategy == ProfilingStrategy::kMin) {
        std::map<std::string, std::vector<int64_t>> min_values = {
            {"m", {m_min}},
            {"n", {n}},
        };
        running_infos_[GenWorkLoad(min_values)] =
            RunningItem(GenWorkLoad(min_values), GenWorkLoad(shape_values_map), PerfResult());
    }
    else if (profiling_strategy == ProfilingStrategy::kIteration) {
        // Create iteration-based profiling configurations
        for (int64_t m = m_min; m <= m_max; m += step_value) {
            std::map<std::string, std::vector<int64_t>> iter_value{{"m", {m}}, {"n", {n}}};
            running_infos_[GenWorkLoad(iter_value)] =
                RunningItem(GenWorkLoad(iter_value), GenWorkLoad(iter_value), PerfResult());
        }
    }
    else {
        FC_THROW(Unimplemented("RMSNorm only supports MIN, MAX or ITERATION profiling strategies"));
    }
}

template<typename T>
void RMSNormOp<T>::IsBuildProfilingEngine()
{
    // Check database for existing profiling results
    for (const auto& [profiling_workload, running_info] : running_infos_) {
        std::map<std::string, int> profiling_key_map = ExtractWorkLoad(profiling_workload);

        NormProblem norm_problem;
        norm_problem.kind_               = op_kind_;
        norm_problem.x_dtype_            = CppTypeToDataType<T>::value;
        norm_problem.y_dtype_            = CppTypeToDataType<T>::value;
        norm_problem.smooth_scale_dtype_ = DataType::FLOAT32;
        norm_problem.y_scale_dtype_      = DataType::FLOAT32;
        norm_problem.m_                  = profiling_key_map.at("m");
        norm_problem.n_                  = profiling_key_map.at("n");
        norm_problem.is_add_bias_        = is_add_bias_;
        norm_problem.fused_add_          = fused_add_;
        norm_problem.fused_quant_        = fused_quant_;

        auto instance_data = InstanceData{
            Environment{.device_name_ = GetDeviceName(), .rocm_version_ = Compiler::GetROCmVersion()},
            Setting{},
            CodeGenKind::Norm,
            norm_problem,
        };

        auto [instance_name, perf_result] = ProfilingEngine::GetInstance()->GetProfilingDB()->Query(instance_data);

        if (!instance_name.empty() && perf_result.IsValid() && !FLAGS_FC_FORCE_PROFILING) {
            VLOG(1) << "Loaded cached profiling result: " << instance_name << " [latency: " << perf_result.latency_
                    << "ms, "
                    << "tflops: " << perf_result.tflops_ << ", "
                    << "bandwidth: " << perf_result.bandwidth_ << "GB/s]";

            running_infos_[profiling_workload].instance_name_ = instance_name;
            running_infos_[profiling_workload].perf_result_   = perf_result;
        }
        else {
            VLOG(1) << "No cached profiling result found, will generate new profile";
        }
    }
}

template<typename T>
std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
RMSNormOp<T>::CodeGenForTuning(const ProfilingStrategy& profiling_strategy)
{
    ExtractRunningInfo(profiling_strategy);

    if (!FLAGS_FC_FORCE_PROFILING) {
        IsBuildProfilingEngine();
    }

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> generated_files;

    // Generate tuning code for each workload
    for (const auto& [workload_key, running_info] : running_infos_) {

        std::map<std::string, int> profiling_key_map = ExtractWorkLoad(workload_key);
        NormProblem                problem;
        problem.kind_               = op_kind_;
        problem.x_dtype_            = CppTypeToDataType<T>::value;
        problem.y_dtype_            = CppTypeToDataType<T>::value;
        problem.smooth_scale_dtype_ = DataType::FLOAT32;
        problem.y_scale_dtype_      = DataType::FLOAT32;
        problem.m_                  = profiling_key_map.at("m");
        problem.n_                  = profiling_key_map.at("n");
        problem.is_add_bias_        = is_add_bias_;
        problem.fused_add_          = fused_add_;
        problem.fused_quant_        = fused_quant_;

        instance_map_ = NormEmitter::GetInstance()->GetInstanceMap(problem);
        FC_ENFORCE_NE(instance_map_.size(), 0, Fatal("No RMSNorm instances generated"));

        if (running_info.IsInstanceExist()) {
            VLOG(1) << "RMSNorm instance already exists, skipping profiling";
            continue;
        }

        generated_files =
            register_kernel_ptr_->CodeGenForTuning(context_ptr_->GetName(), GetNormKindName(op_kind_), instance_map_);
    }

    return generated_files;
}

template<typename T>
std::vector<std::string> RMSNormOp<T>::GetTuningCmd(const std::string&                profiling_file_prefix,
                                                    const std::string&                profiling_filename,
                                                    const std::map<std::string, int>& profiling_key_map)
{
    std::filesystem::path exe_path =
        std::filesystem::path(profiling_file_prefix) / profiling_filename / profiling_filename;
    FileManager::CreateDirectoryIfNotExists(exe_path.parent_path());

    if (FileManager::CheckWithRetries(exe_path, 3, 5)) {
        return {exe_path.string(),
                "-m=" + std::to_string(profiling_key_map.at("m")),
                "-n=" + std::to_string(profiling_key_map.at("n")),
                "-e=" + std::to_string(eps_)};
    }

    LOG(ERROR) << "Executable not found: " << exe_path.string();
    return {};
}

template<typename T>
void RMSNormOp<T>::TuningSingleWorkload(const std::string&  profiling_file_prefix,
                                        const std::string&  profiling_workload,
                                        GPUProfilingRunner& profiling_runner)
{
    std::map<std::string, int> profiling_key_map = ExtractWorkLoad(profiling_workload);

    // Check database first
    NormProblem norm_problem;
    norm_problem.kind_               = op_kind_;
    norm_problem.x_dtype_            = CppTypeToDataType<T>::value;
    norm_problem.y_dtype_            = CppTypeToDataType<T>::value;
    norm_problem.smooth_scale_dtype_ = DataType::FLOAT32;
    norm_problem.y_scale_dtype_      = DataType::FLOAT32;
    norm_problem.m_                  = profiling_key_map.at("m");
    norm_problem.n_                  = profiling_key_map.at("n");
    norm_problem.is_add_bias_        = is_add_bias_;
    norm_problem.fused_add_          = fused_add_;
    norm_problem.fused_quant_        = fused_quant_;

    auto instance_data = InstanceData{
        Environment{.device_name_ = GetDeviceName(), .rocm_version_ = Compiler::GetROCmVersion()},
        Setting{},
        CodeGenKind::Norm,
        norm_problem,
    };

    auto [instance_name, perf_result] = ProfilingEngine::GetInstance()->GetProfilingDB()->Query(instance_data);

    if (!instance_name.empty() && perf_result.IsValid() && !FLAGS_FC_FORCE_PROFILING) {
        VLOG(1) << "Using cached profiling result: " << instance_name;
        return;
    }

    // Execute tuning for each instance
    for (auto& [instance_name, _] : instance_map_) {
        auto command = GetTuningCmd(profiling_file_prefix, instance_name, profiling_key_map);
        if (command.empty())
            continue;

        auto callback = [&](PerfResult& perf_result, Postprocesser& postprocesser) {
            instance_data.SetInstanceName(instance_name);
            instance_data.SetPerfResult(perf_result);
            postprocesser.AddInstance(instance_data, running_infos_);
        };

        VLOG(1) << "RMSNorm tuning command: " << JoinStrings(command, " ");
        profiling_runner.Push(command, callback);
    }
}

template<typename T>
void RMSNormOp<T>::Tuning(GPUProfilingRunner& profiling_runner, const std::string& folder_name)
{
    std::filesystem::path profiling_file_prefix = std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name
                                                  / context_ptr_->GetName() / "profiling" / GetNormKindName(op_kind_);
    FileManager::CreateDirectoryIfNotExists(profiling_file_prefix);

    for (const auto& [profiling_workload, running_item] : running_infos_) {
        if (!running_item.IsInstanceExist()) {
            TuningSingleWorkload(profiling_file_prefix, profiling_workload, profiling_runner);
        }
        else {
            VLOG(1) << "RMSNorm instance already exists, skipping tuning";
        }
    }
}

template<typename T>
std::string RMSNormOp<T>::CodeGenForRunning()
{
    return register_kernel_ptr_->CodeGenForRunning(GetName(), context_ptr_->GetName(), running_infos_, instance_map_);
}

template<typename T>
void RMSNormOp<T>::Forward()
{
    if (!context_ptr_->IsBuilt()) {
        return;
    }

    // Extract input/output pointers based on configuration
    Variable* x         = GetParentNode(0);
    T*        x_ptr     = reinterpret_cast<T*>(x->GetValue());
    T*        gamma_ptr = reinterpret_cast<T*>(GetParentNode(1)->GetValue());

    // Optional tensors based on configuration flags
    int offset = 2;
    T*  x_residual_ptr =
        (fused_add_ != FusedAddEnum::NO_ADD) ? reinterpret_cast<T*>(GetParentNode(offset++)->GetValue()) : nullptr;

    T* smooth_scale_ptr = (fused_quant_ == FusedQuantEnum::SMOOTH_DYNAMIC_QUANT) ?
                              reinterpret_cast<T*>(GetParentNode(offset++)->GetValue()) :
                              nullptr;

    T* y_residual_ptr = (fused_add_ == FusedAddEnum::PRE_ADD_STORE) ?
                            reinterpret_cast<T*>(GetParentNode(offset++)->GetValue()) :
                            nullptr;

    T* y_scale_ptr = (fused_quant_ != FusedQuantEnum::NO_SWEEP) ?
                         reinterpret_cast<T*>(GetParentNode(offset++)->GetValue()) :
                         nullptr;

    T* y_ptr = reinterpret_cast<T*>(GetChildNode(0)->GetValue());

    // Update output shape
    Shape c_shape = InferShape(x);
    output_var_[0]->SetShape(c_shape);

    // Calculate tensor dimensions
    auto calc_batch_size = [&](Variable* x) {
        auto x_shape_vec = x->GetShape().ToVector();
        int  dim0_value  = 1;
        std::for_each(
            x_shape_vec.begin(), x_shape_vec.end() - 1, [&](const DDim& dim) { dim0_value *= dim.GetValues()[0]; });
        return dim0_value;
    };

    VLOG(1) << "RMSNorm forward: " << c_shape.ToString();

    // Debug tensor values (only in debug builds)
    if (VLOG_IS_ON(3)) {
        const std::string prefix = "[" + GetName() + "] ";
        PrintToScreen(x_ptr, 3, prefix + "x_ptr");
        PrintToScreen(gamma_ptr, 3, prefix + "gamma_ptr");
        PrintToScreen(x_residual_ptr, 3, prefix + "x_residual_ptr");
        PrintToScreen(smooth_scale_ptr, 3, prefix + "smooth_scale_ptr");
        PrintToScreen(y_residual_ptr, 3, prefix + "y_residual_ptr");
        PrintToScreen(y_scale_ptr, 3, prefix + "y_scale_ptr");
    }

    // Prepare kernel arguments
    NormKernelArgs kernel_args;
    kernel_args.x_ptr_            = x_ptr;
    kernel_args.x_residual_ptr_   = x_residual_ptr;
    kernel_args.smooth_scale_ptr_ = smooth_scale_ptr;
    kernel_args.gamma_ptr_        = gamma_ptr;
    kernel_args.y_ptr_            = y_ptr;
    kernel_args.y_residual_ptr_   = y_residual_ptr;
    kernel_args.y_scale_ptr_      = y_scale_ptr;
    kernel_args.x_dim_0_          = calc_batch_size(x);
    kernel_args.x_dim_1_          = x->GetShape().GetLastDim().GetValues()[0];
    kernel_args.eps_              = eps_;
    kernel_args.x_stride_         = kernel_args.x_dim_1_;
    kernel_args.xr_stride_        = kernel_args.x_dim_1_;
    kernel_args.y_stride_         = kernel_args.x_dim_1_;
    kernel_args.yr_stride_        = kernel_args.x_dim_1_;
    kernel_args.stream_           = context_ptr_->GetStream();

    // Launch kernel
    register_kernel_ptr_->KernelLauncher(GetName(), kernel_args);

    if (VLOG_IS_ON(3)) {
        PrintToScreen(y_ptr, 3, "[" + GetName() + "] y_ptr");
    }
}

template class RMSNormOp<float>;
template class RMSNormOp<_Float16>;
template class RMSNormOp<ushort>;

}  // namespace flashck