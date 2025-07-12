#include "flashck/core/module/operations/norm_ops/rms_norm_op.h"

#include "flashck/core/utils/debug_utils.h"
#include "flashck/core/utils/enforce.h"
#include "flashck/core/utils/flags.h"
#include "flashck/core/utils/log.h"

FC_DECLARE_bool(FC_FORCE_PROFILE);
FC_DECLARE_bool(FC_FORCE_PROFILING_DB);
FC_DECLARE_string(FC_HOME_PATH);

namespace flashck {

template<typename T>
RMSNormOp<T>::RMSNormOp(Shape          normalized_shape,
                        FusedAddEnum   fused_add,
                        FusedQuantEnum fused_quant,
                        std::string    op_name):
    Operation(op_name), fused_add_(fused_add), fused_quant_(fused_quant)
{
    default_normalized_shape_ = normalized_shape;
}

template<typename T>
void RMSNormOp<T>::CheckParamShape(const Shape& x_shape, const Shape& param_shape, const std::string& param_name)
{
    if (param_name != "normalized" && param_shape.GetNumDim()) {
        return;
    }

    if (!x_shape.GetLastDim().IsStatic()) {
        FC_THROW(Unimplemented("layernorm requires reduction dim to be static. Current input shape: {}",
                               x_shape.ToString()));
    }

    for (const auto& shape : param_shape.ToVector()) {
        if (!shape.IsStatic()) {
            FC_THROW(Unimplemented(
                "Layernorm {} shape must be immutable values. Current value: {}", param_name, param_shape.ToString()));
        }
    }

    int64_t batch_ndims = x_shape.GetNumDim() - param_shape.GetNumDim();
    for (int i = 0; i < param_shape.GetNumDim(); i++) {
        if (param_shape[i].GetValues() != x_shape[batch_ndims + i].GetValues()) {
            FC_THROW(Unavailable(
                "Layernorm {} shape must be broadcastable to the input shape. Current {} shape: {}, input shape: {}",
                param_name,
                param_shape.ToString(),
                x_shape.ToString()));
        }
    }
}

template<typename T>
void RMSNormOp<T>::CheckShape(const Shape& x_shape,
                              const Shape& gamma_shape,
                              const Shape& x_residual_shape,
                              const Shape& smmoth_scale_shape,
                              const Shape& y_residual_shape,
                              const Shape& y_scale_shape,
                              const Shape& normalized_shape)
{
    FC_ENFORCE_LT(
        normalized_shape.GetNumDim(),
        x_shape.GetNumDim(),
        Unavailable(
            "Layernorm normalized_shape length must be smaller than the input. Current normalized_shape: {}, input shape: {}",
            normalized_shape.ToString(),
            x_shape.ToString()));

    FC_ENFORCE_GE(x_shape.GetNumDim(),
                  2,
                  Unimplemented("norm only supports 2D or higher input, runtime rank: {}", x_shape.GetNumDim()));

    if (fused_add_ != FusedAddEnum::NO_ADD) {
        FC_ENFORCE_EQ(
            x_residual_shape.GetNumDim(),
            x_shape.GetNumDim(),
            Unimplemented("norm only supports 2D or higher input, runtime rank: {}", x_residual_shape.GetNumDim()));
    }

    CheckParamShape(x_shape, gamma_shape, "gamma");

    CheckParamShape(x_shape, x_residual_shape, "x_residual");
    CheckParamShape(x_shape, smmoth_scale_shape, "smooth_scale");
    CheckParamShape(x_shape, y_residual_shape, "y_residual");
    CheckParamShape(x_shape, y_scale_shape, "y_scale");
    CheckParamShape(x_shape, normalized_shape, "normalized");
}

/*
Return a list of shapes for x, gamma and beta, where gamma_shape and
beta_shape may be None if gamma and beta are None, respectively.
*/
template<typename T>
std::vector<Shape> RMSNormOp<T>::GetInputShape(
    Variable* x, Variable* gamma, Variable* x_residual, Variable* smooth_scale, Variable* y_residual, Variable* y_scale)
{
    Shape x_shape     = x->GetShape();
    Shape gamma_shape = gamma == nullptr ? Shape() : gamma->GetShape();

    Shape x_residual_shape   = x_residual == nullptr ? Shape() : x_residual->GetShape();
    Shape smmoth_scale_shape = smooth_scale == nullptr ? Shape() : smooth_scale->GetShape();
    Shape y_residual_shape   = y_residual == nullptr ? Shape() : y_residual->GetShape();
    Shape y_scale_shape      = y_scale == nullptr ? Shape() : y_scale->GetShape();

    return {x_shape, gamma_shape, x_residual_shape, smmoth_scale_shape, y_residual_shape, y_scale_shape};
}

template<typename T>
void RMSNormOp<T>::SanityCheck(
    Variable* x, Variable* gamma, Variable* x_residual, Variable* smooth_scale, Variable* y_residual, Variable* y_scale)
{
    std::vector<Shape> input_shape = GetInputShape(x, gamma, x_residual, smooth_scale, y_residual, y_scale);
    CheckShape(input_shape[0],
               input_shape[1],
               input_shape[2],
               input_shape[3],
               input_shape[4],
               input_shape[5],
               normalized_shape_);
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
                                   Shape       normalized_shape,
                                   const float eps)
{
    SanityCheck(x, gamma, x_residual, smooth_scale, y_residual, y_scale);

    if (fused_quant_ == FusedQuantEnum::SMOOTH_DYNAMIC_QUANT && smooth_scale == nullptr) {
        FC_THROW(Unimplemented("Fused quant requires smooth_scale to be provided"));
    }
    if (fused_add_ == FusedAddEnum::PRE_ADD_STORE && y_residual == nullptr) {
        FC_THROW(Unimplemented("Fused add requires y_residual to be provided"));
    }
    if (fused_quant_ != FusedQuantEnum::NO_SWEEP && y_scale == nullptr) {
        FC_THROW(Unimplemented("Fused quant requires y_scale to be provided"));
    }

    eps_                             = eps;
    std::vector<Variable*> input_var = {x, gamma, x_residual, smooth_scale, y_residual, y_scale};
    for (auto var : input_var) {
        if (var != nullptr) {
            input_var_.push_back(var);
        }
    }

    normalized_shape_ = normalized_shape.GetNumDim() ? normalized_shape : default_normalized_shape_;

    Shape output_shape    = InferShape(x);
    auto  max_output_size = std::get<1>(output_shape.GetElementSizeTuple());
    output_var_ = {new Variable(op_name_ + std::string("_output"), max_output_size, CppTypeToDataType<T>::Type())};
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

/*
Invert execution key to get input arguments as integers.
*/
template<typename T>
std::vector<int64_t> RMSNormOp<T>::InvertExecKey(const std::string& key)
{
    std::vector<int64_t> tmp;
    std::regex           pattern("(\\d+)");
    std::smatch          m;
    std::string          s = key;
    while (std::regex_search(s, m, pattern)) {
        tmp.push_back(std::stoi(m[0]));
        s = m.suffix().str();
    }
    return tmp;
}

/*
Generate execution key from the name value mapping.
*/
template<typename T>
std::string RMSNormOp<T>::GenExecKey(const std::map<std::string, std::vector<int64_t>>& name_value_mapping)
{
    std::vector<std::string> key_strs;
    for (auto& [name, values] : name_value_mapping) {
        if (values.size() == 1) {
            key_strs.emplace_back(Sprintf("{} == {}", name, values[0]));
        }
        else if (values.size() > 1) {
            key_strs.emplace_back(Sprintf("{} >= {} && {} <= {}", name, values[0], name, values.back()));
        }
        else {
            FC_THROW(Unavailable("norm input has empty dim values: {}", values[0]));
        }
    }

    return JoinStrings(key_strs, " && ");
}

template<typename T>
void RMSNormOp<T>::ExtractExecPath(const ProfilingStrategy& dynamic_profiling_strategy, const int step_value)
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

    if (dynamic_profiling_strategy == ProfilingStrategy::kMax) {
        std::map<std::string, std::vector<int64_t>> max_values = {
            {"m", {m_max}},
            {"n", {n}},
        };

        std::shared_ptr<ExecItem> exec_item_ptr =
            std::make_shared<ExecItem>(GenExecKey(max_values), GenExecKey(shape_values_map), "");

        exec_path_[exec_item_ptr->profiling_key_] = exec_item_ptr;
    }
    else if (dynamic_profiling_strategy == ProfilingStrategy::kMin) {
        std::map<std::string, std::vector<int64_t>> min_values = {
            {"m", {m_min}},
            {"n", {n}},
        };

        std::shared_ptr<ExecItem> exec_item_ptr =
            std::make_shared<ExecItem>(GenExecKey(min_values), GenExecKey(shape_values_map), "");

        exec_path_[exec_item_ptr->profiling_key_] = exec_item_ptr;
    }
    else if (dynamic_profiling_strategy == ProfilingStrategy::kIteration) {
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
            std::shared_ptr<ExecItem> exec_item_ptr =
                std::make_shared<ExecItem>(GenExecKey(iter_value), GenExecKey(iter_value), "");

            exec_path_[exec_item_ptr->profiling_key_] = exec_item_ptr;
        }
    }

    else {
        FC_THROW(Unimplemented("{}", "norm only supports MIN or MAX dynamic profiling"));
    }
}

template<typename T>
void RMSNormOp<T>::IfShouldBuildProfiler(const std::vector<std::string>& workloads)
{
    for (const auto& workload : workloads) {
        std::string exec_entry_sha1 = SHA1ToHexString(workload);
        auto        query           = NormQueryEntry(DataTypeToShortString(CppTypeToDataType<T>::Type()),
                                    DataTypeToShortString(CppTypeToDataType<T>::Type()),
                                    DataTypeToShortString(DataType::FLOAT32),
                                    DataTypeToShortString(DataType::FLOAT32),
                                    g_norm_operation_kind_names_map.at(op_kind_),
                                    Target::Instance()->GetTargetDeviceName(),
                                    g_short_tensor_operation_names_map.at(epilogue_op_),
                                    exec_entry_sha1,
                                    g_fused_add_enum_str_map.at(fused_add_),
                                    g_fused_quant_enum_str_map.at(fused_quant_));

        auto cache_value = Target::Instance()->QueryProfileCache(CodeGenKind::Norm, query);

        if (cache_value != std::make_tuple("null", -1) && !FLAGS_FC_FORCE_PROFILE) {
            std::string best_algo = std::get<0>(cache_value);
            int64_t     split_k   = std::get<1>(cache_value);
            LOG(INFO) << "Load profiling result for" << op_name_ << "from cache, algo" << best_algo << "split_k"
                      << split_k;

            exec_path_[workload]->algo_ = best_algo;
        }
        else {
            // cache miss - we will have to generate and build profilers
            LOG(INFO) << "No profiling result found for" << op_name_ << "in cache, will build profilers";
        }
    }
}

template<typename T>
std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
RMSNormOp<T>::GenOpProfiler(const ProfilingStrategy& dynamic_profiling_strategy)
{
    KernelKey kernel_key(SourceType::CK_TILE, DataLayout::ALL_LAYOUT, CppTypeToDataType<T>::Type());
    register_kernel_ptr_ =
        KernelFactory::Instance().SelectKernel(g_norm_operation_kind_names_map.at(op_kind_), kernel_key);

    // init exec path
    ExtractExecPath(dynamic_profiling_strategy);

    exec_key_ = GetKeyVector(exec_path_);

    if (!FLAGS_FC_FORCE_PROFILER_CACHE) {
        IfShouldBuildProfiler(exec_key_);
    }
    else {
        LOG(INFO) << "Forced to use cache, skip building profilers for " << op_name_;
        return {};
    }

    std::vector<NormProblem> layernorm_problems;
    layernorm_problems.reserve(exec_key_.size());
    std::for_each(exec_key_.begin(), exec_key_.end(), [&](const std::string& key) {
        std::vector<int64_t> inverse_res = InvertExecKey(key);
        NormProblem          layernorm_problem{CppTypeToDataType<T>::Type(),
                                      CppTypeToDataType<T>::Type(),
                                      DataType::FLOAT32,
                                      DataType::FLOAT32,
                                      inverse_res[0],
                                      inverse_res[1],
                                      op_kind_,
                                      epilogue_op_,
                                      NormBiasEnum::NO_BIAS,
                                      fused_add_,
                                      fused_quant_};

        layernorm_problems.emplace_back(layernorm_problem);
    });

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> generated_profilers;

    for (int i = 0; i < exec_key_.size(); i++) {
        Target::Instance()->GenerateKernel(CodeGenKind::Norm, layernorm_problems[i]);

        kernel_instance_map_ = register_kernel_ptr_->Init(op_kind_, epilogue_op_);
        if (kernel_instance_map_.size() == 0) {
            FC_THROW(Fatal("No layernorm op instances were generated for {}", op_name_));
        }

        if (exec_path_[exec_key_[i]]->algo_ == "") {
            generated_profilers = register_kernel_ptr_->GenKernelProfiler(context_ptr_->GetName(), GetAttrsMap());
        }
        else {
            LOG(INFO) << "op_name: " << op_name_ << ", " << "workload: " << exec_key_[i] << " from cache, not profile";
        }
    }

    return generated_profilers;
}

template<typename T>
std::vector<std::string> RMSNormOp<T>::GenOpProfileCmd(const std::string&          profiler_prefix,
                                                       const std::string&          profiler_filename,
                                                       const std::vector<int64_t>& input_shape)
{
    std::filesystem::path exe_path = std::filesystem::path(profiler_prefix) / profiler_filename;

    if (!CheckExistWithRetries(exe_path, 3, 5)) {
        FC_THROW(Fatal("Profiler {} is not executable", exe_path.string()));
    }

    std::vector<std::string> cmd = {exe_path.string(),
                                    "-m=" + std::to_string(input_shape[0]),
                                    "-n=" + std::to_string(input_shape[1]),
                                    "-e=" + std::to_string(eps_),
                                    "-x_stride=" + std::to_string(x_stride_),
                                    "-xr_stride=" + std::to_string(xr_stride_),
                                    "-y_stride=" + std::to_string(y_stride_),
                                    "-yr_stride=" + std::to_string(yr_stride_)};

    return cmd;
}

template<typename T>
void RMSNormOp<T>::ProfileSingleWorkload(const std::string&                         profiler_prefix,
                                         const std::string&                         workload,
                                         const std::shared_ptr<GPUProfilingRunner>& profiler_runner_ptr,
                                         bool                                       force_cache)
{
    std::vector<std::string> kernel_instance_map_key = GetKeyVector(kernel_instance_map_);

    std::string exec_entry_sha1 = SHA1ToHexString(workload);
    auto        query           = NormQueryEntry(DataTypeToShortString(CppTypeToDataType<T>::Type()),
                                DataTypeToShortString(CppTypeToDataType<T>::Type()),
                                DataTypeToShortString(DataType::FLOAT32),
                                DataTypeToShortString(DataType::FLOAT32),
                                g_norm_operation_kind_names_map.at(op_kind_),
                                Target::Instance()->GetTargetDeviceName(),
                                g_short_tensor_operation_names_map.at(epilogue_op_),
                                exec_entry_sha1,
                                g_fused_add_enum_str_map.at(fused_add_),
                                g_fused_quant_enum_str_map.at(fused_quant_));

    auto cache_value = Target::Instance()->QueryProfileCache(CodeGenKind::Norm, query);

    if (cache_value == std::make_tuple("null", -1) && force_cache) {
        LOG(WARNING) << "force_cache is enabled but we could not find the following cache available on device. "
                     << "op_name: " << op_name_ << " exec_entry_sha1: " << exec_entry_sha1;
    }

    for (const auto& kernel_config_name : kernel_instance_map_key) {
        auto GenCallback = [&] {
            auto process_result_callback =
                [&](const std::vector<ProfileResult>&           result,
                    const std::shared_ptr<ProfilerPostprocess>& postprocessing_delegate_ptr) {
                    postprocessing_delegate_ptr->AddInstance(
                        result, CodeGenKind::Norm, GetAttrsMap(), kernel_config_name, workload);
                };
            return process_result_callback;
        };

        auto                     input_shape = InvertExecKey(workload);
        std::vector<std::string> command     = GenOpProfileCmd(profiler_prefix, kernel_config_name, input_shape);

        LOG(INFO) << "profile command: " << JoinStrings(command);

        profiler_runner_ptr->Push(command, GenCallback());
    }
}

/*
Selects the fastest kernel configurations.
*/
template<typename T>
void RMSNormOp<T>::Profile(const std::shared_ptr<GPUProfilingRunner>& profiler_runner_ptr,
                           const std::string&                         folder_name)
{

    std::filesystem::path profiler_prefix =
        std::filesystem::path(FLAGS_FC_HOME_PATH) / folder_name / context_ptr_->GetName() / "profiling" / op_name_;

    for (const auto& workload : exec_key_) {
        if (exec_path_[workload]->algo_ == "") {
            if (kernel_instance_map_.size() == 0) {
                kernel_instance_map_ = register_kernel_ptr_->Init(op_kind_, epilogue_op_);
            }

            ProfileSingleWorkload(profiler_prefix, workload, profiler_runner_ptr, FLAGS_FC_FORCE_PROFILER_CACHE);
        }
        else {
            LOG(INFO) << op_name_ << " from cache, not profile";
        }
    }
}

template<typename T>
std::string RMSNormOp<T>::GenOpFunction()
{
    return register_kernel_ptr_->GenKernelFunction(GetName(), context_ptr_->GetName(), GetAttrsMap());
}

template<typename T>
std::unordered_map<std::string, std::any> RMSNormOp<T>::GetAttrsMap()
{
    std::unordered_map<std::string, std::any> op_attrs_map{{"op_name", op_name_},
                                                           {"normalized_shape", normalized_shape_},
                                                           {"x_dtype", CppTypeToDataType<T>::Type()},
                                                           {"y_dtype", CppTypeToDataType<T>::Type()},
                                                           {"smooth_scale_dtype", DataType::FLOAT32},
                                                           {"y_scale_dtype", DataType::FLOAT32},
                                                           {"eps", eps_},
                                                           {"op_kind", op_kind_},
                                                           {"epilogue_op", epilogue_op_},
                                                           {"exec_path", exec_path_},
                                                           {"kernel_instance_map", kernel_instance_map_},
                                                           {"is_add_bias", is_add_bias_},
                                                           {"fused_add", fused_add_},
                                                           {"fused_quant", fused_quant_}};
    return op_attrs_map;
}

template<typename T>
void RMSNormOp<T>::Forward()
{
    Variable* x = GetParentNode(0);

    T* x_ptr            = (T*)x->GetValue();
    T* gamma_ptr        = (T*)GetParentNode(1)->GetValue();
    T* x_residual_ptr   = fused_add_ != FusedAddEnum::NO_ADD ? (T*)GetParentNode(2)->GetValue() : nullptr;
    T* smooth_scale_ptr = fused_quant_ == FusedQuantEnum::SMOOTH_DYNAMIC_QUANT ?
                              (T*)GetParentNode(2 + (x_residual_ptr != nullptr))->GetValue() :
                              nullptr;
    T* y_residual_ptr =
        fused_add_ == FusedAddEnum::PRE_ADD_STORE ?
            (T*)GetParentNode(2 + (smooth_scale_ptr != nullptr) + (x_residual_ptr != nullptr))->GetValue() :
            nullptr;
    T* y_scale_ptr = fused_quant_ != FusedQuantEnum::NO_SWEEP ?
                         (T*)GetParentNode(2 + (y_residual_ptr != nullptr) + (smooth_scale_ptr != nullptr)
                                           + (x_residual_ptr != nullptr))
                             ->GetValue() :
                         nullptr;

    T* y_ptr = (T*)GetChildNode(0)->GetValue();

    if (!context_ptr_->IsBuilt()) {
        return;
    }

    Shape c_shape = InferShape(GetParentNode(0));
    output_var_[0]->SetShape(c_shape);  // must update actual output shape

    auto broadcast_shape_func = [&](Variable* x) {
        auto x_shape_vec = x->GetShape().ToVector();
        int  dim0_value  = 1;
        std::for_each(
            x_shape_vec.begin(), x_shape_vec.end() - 1, [&](const DDim& dim) { dim0_value *= dim.GetValues()[0]; });

        return dim0_value;
    };

    VLOG(1) << "norm " << this->op_name_ << ", out shape: " << c_shape.ToString();

    // PrintToScreen(x_ptr, 3, "[" + this->op_name_ + "]" + "x_ptr");
    // PrintToScreen(gamma_ptr, 3, "[" + this->op_name_ + "]" + "gamma_ptr");
    // PrintToScreen(x_residual_ptr, 3, "[" + this->op_name_ + "]" + "x_residual_ptr");
    // PrintToScreen(smooth_scale_ptr, 3, "[" + this->op_name_ + "]" + "smooth_scale_ptr");
    // PrintToScreen(y_residual_ptr, 3, "[" + this->op_name_ + "]" + "y_residual_ptr");
    // PrintToScreen(y_scale_ptr, 3, "[" + this->op_name_ + "]" + "y_scale_ptr");

    NormKernelArgs rms_norm_args;
    rms_norm_args.x_ptr_            = x_ptr;
    rms_norm_args.x_residual_ptr_   = x_residual_ptr;
    rms_norm_args.smooth_scale_ptr_ = smooth_scale_ptr;
    rms_norm_args.gamma_ptr_        = gamma_ptr;
    rms_norm_args.y_ptr_            = y_ptr;
    rms_norm_args.y_residual_ptr_   = y_residual_ptr;
    rms_norm_args.y_scale_ptr_      = y_scale_ptr;

    rms_norm_args.x_dim_0_   = broadcast_shape_func(x);
    rms_norm_args.x_dim_1_   = x->GetShape().GetLastDim().GetValues()[0];
    rms_norm_args.eps_       = eps_;
    rms_norm_args.x_stride_  = x_stride_ == -1 ? rms_norm_args.x_dim_1_ : x_stride_;
    rms_norm_args.xr_stride_ = xr_stride_ == -1 ? rms_norm_args.x_dim_1_ : xr_stride_;
    rms_norm_args.y_stride_  = y_stride_ == -1 ? rms_norm_args.x_dim_1_ : y_stride_;
    rms_norm_args.yr_stride_ = yr_stride_ == -1 ? rms_norm_args.x_dim_1_ : yr_stride_;
    rms_norm_args.stream_    = context_ptr_->GetStream();

    register_kernel_ptr_->KernelLauncher(GetName(), rms_norm_args);

    // PrintToScreen(y_ptr, 3, "[" + this->op_name_ + "]" + "y_ptr");
    // ResultChecker(y_ptr, std::get<0>(c_shape.GetElementSizeTuple()), "[" + this->op_name_ + "]" + "y_ptr");
}

template class RMSNormOp<float>;
template class RMSNormOp<_Float16>;
template class RMSNormOp<ushort>;

}  // namespace flashck