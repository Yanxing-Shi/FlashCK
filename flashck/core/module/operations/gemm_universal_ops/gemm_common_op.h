#pragma once

#include <any>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

#include "flashck/core/utils/dtype.h"
#include "flashck/core/utils/enforce.h"
#include "flashck/core/utils/file_utils.h"
#include "flashck/core/utils/flags.h"
#include "flashck/core/utils/layout.h"
#include "flashck/core/utils/log.h"

#include "flashck/core/profiler/base.h"
#include "flashck/core/profiler/gpu_profiler_runner.h"

#include "flashck/core/graph/node.h"

#include "flashck/core/module/kernels/gemm_kernels/gemm_common_kernel.h"
#include "flashck/core/module/kernels/kernel_factory.h"
#include "flashck/core/profiler/gemm_cache_entry.h"

LI_DECLARE_bool(LI_FORCE_PROFILE);
LI_DECLARE_bool(LI_FORCE_PROFILER_CACHE);
LI_DECLARE_string(LI_HOME_PATH);

namespace flashck {

/*
Base gemm operators
*/
template<typename CppType, typename OpType = void>
class GemmCommonOp: public Operation {
public:
    GemmCommonOp() = default;

    GemmCommonOp(std::string op_name): Operation(op_name) {}

    void AlignAB(Variable* a, Variable* b)
    {
        auto a_shape = a->GetShape();
        auto b_shape = b->GetShape();

        if (a_shape.GetLastDim() != b_shape.GetLastDim()) {
            LI_THROW(Unavailable("A/B shape mismatch, A: {}, B: {}", a_shape.ToString(), b_shape.ToString()));
        }

        if (!a_shape.GetLastDim().IsStatic()) {
            LI_THROW(Unavailable("K must be static! k: {}", a_shape.GetLastDim().ToString()));
        }
    }

    void SanityCheck(Variable* a, Variable* b)
    {
        Shape a_shape = a->GetShape();
        Shape b_shape = b->GetShape();
        if (a_shape.GetNumDim() < 2) {
            LI_THROW(Unavailable("Input A shape should be at least 2D, got {}", a_shape.ToString()));
        }

        if (b_shape.GetNumDim() < 2) {
            LI_THROW(Unavailable("Input B shape should be at least 2D, got {}", b_shape.ToString()));
        }

        if (a->GetDtype() != b->GetDtype()) {
            LI_THROW(Unavailable("Input A and B should have the same data type, got {} and {}",
                                 DataTypeToString(a->GetDtype()),
                                 DataTypeToString(b->GetDtype())));
        }
    }

    template<typename... Args>
    Shape InferShape(Args... args)
    {
        return static_cast<OpType*>(this)->InferShapeImpl(args...);
    }

    std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> ExtractDims(bool for_profiling = false)
    {
        return static_cast<OpType*>(this)->ExtractDimsImpl(for_profiling);
    }

    std::string GenExecKey(const std::map<std::string, std::vector<int64_t>>& name_value_mapping)
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
                LI_THROW(Fatal("Gemm input has empty dim values: {}", values[0]));
            }
        }

        return JoinToString(key_strs, " && ");
    }

    static std::vector<int64_t> InverseKeyFunc(const std::string& key)
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

    void ExtractExecPath(const DynamicProfileStrategy& dynamic_profiling_strategy, const int step_value = 1)
    {
        std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> dim_info_map = this->ExtractDims(false);

        // dynamic shape M:{range_lower, range_upper}, K:{range_lower, range_upper}, N:{range_lower, range_upper}
        std::map<std::string, std::vector<DDim>> dim_map;

        for (auto& [name, dim_infos] : dim_info_map) {
            std::shared_ptr<DimInfo> dim_info;
            for (auto& d : dim_infos) {
                if (d->placeholder_) {
                    continue;
                }

                if (dim_info == nullptr) {
                    dim_info = d;
                }
                else if (d->source_ == TensorSource::kInput) {
                    // input should have priority.
                    dim_info = d;
                }
            }

            if (dim_info == nullptr) {
                LI_THROW(Fatal("Couldn't find valid dim info for dim {}", name));
            }

            std::vector<Variable*> var_vec   = dim_info->source_ == TensorSource::kInput ? input_var_ : output_var_;
            std::vector<DDim>      dim_shape = var_vec[dim_info->tensor_idx_]->GetShape().ToVector();

            for (const auto& dim : dim_info->dim_idx_) {
                VLOG(1) << "name:" << name << " idx: " << dim << " value: " << dim_shape[dim].ToString();
                dim_map[name].emplace_back(dim_shape[dim]);
            }
        }

        std::map<std::string, std::vector<int64_t>> shape_values_map;

        int64_t initial_product = 1;
        for (const auto& [name, dims] : dim_map) {
            std::vector<int64_t> min_dims_value, max_dims_value;
            // dynamic shape
            for (const auto& dim : dims) {
                min_dims_value.emplace_back(dim.GetLowerBound());
                max_dims_value.emplace_back(dim.GetUpperBound());
            }

            int64_t min_value = std::accumulate(
                min_dims_value.begin(), min_dims_value.end(), initial_product, std::multiplies<int64_t>());
            int64_t max_value = std::accumulate(
                max_dims_value.begin(), max_dims_value.end(), initial_product, std::multiplies<int64_t>());

            std::vector<int64_t> shape_values{min_value, max_value};
            std::sort(shape_values.begin(), shape_values.end());

            shape_values_map[name] = shape_values;

            VLOG(1) << "name: " << name << " min_dims_value: " << min_value;
            VLOG(1) << "name: " << name << " max_dims_value: " << max_value;
        }

        if (dynamic_profiling_strategy == DynamicProfileStrategy::kMax) {
            std::map<std::string, std::vector<int64_t>> max_values;
            for (auto& [name, shape_values] : shape_values_map) {
                int64_t max_shape_values = *max_element(shape_values.begin(), shape_values.end());
                max_values[name]         = {max_shape_values};
            }

            VLOG(1) << "profiling_key: " << GenExecKey(max_values);
            VLOG(1) << "exec_cond: " << GenExecKey(shape_values_map);

            std::shared_ptr<ExecItem> exec_item_ptr =
                std::make_shared<ExecItem>(GenExecKey(max_values), GenExecKey(shape_values_map), "");

            exec_path_[exec_item_ptr->profiling_key_] = exec_item_ptr;
        }
        else if (dynamic_profiling_strategy == DynamicProfileStrategy::kMin) {
            std::map<std::string, std::vector<int64_t>> min_values;
            for (auto& [name, shape_values] : shape_values_map) {
                int64_t min_shape_values = *min_element(shape_values.begin(), shape_values.end());
                min_values[name]         = {min_shape_values};
            }

            std::shared_ptr<ExecItem> exec_item_ptr =
                std::make_shared<ExecItem>(GenExecKey(min_values), GenExecKey(shape_values_map), "");
            exec_path_[exec_item_ptr->profiling_key_] = exec_item_ptr;
        }
        else if (dynamic_profiling_strategy == DynamicProfileStrategy::kIteration) {
            // iteration
            std::map<std::string, std::vector<int64_t>> iter_values_map;
            size_t                                      max_value_size = 0;
            for (const auto& [name, shape_values] : shape_values_map) {
                for (int64_t i = shape_values[0]; i <= shape_values[1]; i += step_value) {
                    iter_values_map[name].push_back(i);
                }
                max_value_size = std::max(max_value_size, iter_values_map[name].size());
            }

            // if lower bound and upper bound are the same, we need to fill the vector with the same value
            for (auto& [name, values] : iter_values_map) {
                // lower bound and upper bound are the same
                if (values.size() == 1) {
                    VLOG(1) << "name: " << name << " lower bound and upper bound are the same";
                    values.resize(max_value_size);
                    std::fill(values.begin(), values.end(), values.front());
                }
                VLOG(1) << "max_step: " << max_value_size;
                VLOG(1) << "name: " << name << " iter_values: " << JoinToString(values, ",")
                        << " size: " << values.size();
            }

            // generate exec path
            std::vector<std::map<std::string, std::vector<int64_t>>> iter_values_vec;
            for (size_t i = 0; i < max_value_size; i++) {
                std::map<std::string, std::vector<int64_t>> iter_value{};
                for (const auto& [name, values] : iter_values_map) {
                    iter_value[name].push_back(values[i]);
                }
                iter_values_vec.push_back(iter_value);
            }

            for (const auto& iter_value : iter_values_vec) {
                VLOG(1) << "profiling_key: " << GenExecKey(iter_value);
                VLOG(1) << "exec_cond: " << GenExecKey(iter_value);

                std::shared_ptr<ExecItem> exec_item_ptr =
                    std::make_shared<ExecItem>(GenExecKey(iter_value), GenExecKey(iter_value), "");
                exec_path_[exec_item_ptr->profiling_key_] = exec_item_ptr;
            }
        }

        else {
            LI_THROW(Unimplemented("{}", "Gemm only supports MIN or MAX or Interation dynamic profiling"));
        }
    }

    void IfShouldBuildProfiler(const std::vector<std::string>& workloads)
    {
        for (const auto& workload : workloads) {
            std::string exec_entry_sha1 = SHA1ToHexString(workload);
            auto        query           = GemmQueryEntry(DataTypeToShortString(CppTypeToDataType<CppType>::Type()),
                                        DataTypeToShortString(CppTypeToDataType<CppType>::Type()),
                                        DataTypeToShortString(CppTypeToDataType<CppType>::Type()),
                                        DataTypeToShortString(DataType::FLOAT32),
                                        DataLayoutToString(layout_),
                                        op_name_,
                                        Target::Instance()->GetTargetDeviceName(),
                                        g_short_tensor_operation_names_map.at(epilogue_op_),
                                        exec_entry_sha1,
                                        permute_shape_.ToString());

            auto cache_value = Target::Instance()->QueryProfileCache(GenOperationKind::Gemm, query);

            if (cache_value != std::make_tuple("null", -1) && !FLAGS_LI_FORCE_PROFILE) {
                std::string best_algo = std::get<0>(cache_value);
                int64_t     split_k   = std::get<1>(cache_value);
                LOG(INFO) << "Load profiling result for" << op_name_ << "from cache, algo" << best_algo << "split_k"
                          << split_k;

                exec_path_[workload]->algo_    = best_algo;
                exec_path_[workload]->split_k_ = split_k;
            }
            else {
                // cache miss - we will have to generate and build profilers
                LOG(INFO) << "No profiling result found for" << op_name_ << "in cache, will build profilers";
            }
        }
    }

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenOpProfiler(const DynamicProfileStrategy& dynamic_profiling_strategy = DynamicProfileStrategy::kMax) override
    {
        KernelKey kernel_key(SourceType::CK, this->layout_, CppTypeToDataType<CppType>::Type());
        register_kernel_ptr_ = KernelFactory::Instance().SelectKernel(this->op_name_, kernel_key);

        // init exec path
        ExtractExecPath(dynamic_profiling_strategy);

        exec_key_ = GetKeyVector(exec_path_);

        if (!FLAGS_LI_FORCE_PROFILER_CACHE) {
            IfShouldBuildProfiler(exec_key_);
        }
        else {
            LOG(INFO) << "Forced to use cache, skip building profilers for " << op_name_;
            return {};
        }

        std::vector<std::vector<int64_t>> all_workloads;
        std::vector<GemmProblem>          gemm_problems;
        all_workloads.reserve(exec_key_.size());
        gemm_problems.reserve(exec_key_.size());
        std::for_each(exec_key_.begin(), exec_key_.end(), [&](const std::string& key) {
            std::vector<int64_t> k_m_n_values = InverseKeyFunc(key);
            all_workloads.emplace_back(k_m_n_values);
            GemmProblem gemm_problem(op_kind_,
                                     k_m_n_values[1],
                                     k_m_n_values[2],
                                     k_m_n_values[0],
                                     CppTypeToDataType<CppType>::Type(),
                                     CppTypeToDataType<CppType>::Type(),
                                     CppTypeToDataType<CppType>::Type(),
                                     DataType::FLOAT32,
                                     {},
                                     CppTypeToDataType<CppType>::Type(),
                                     layout_,
                                     {},
                                     epilogue_op_);
            gemm_problems.emplace_back(gemm_problem);
        });

        std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> generated_profilers;

        for (int i = 0; i < exec_key_.size(); i++) {
            Target::Instance()->GenerateKernel(GenOperationKind::Gemm, gemm_problems[i]);

            // init kernel instance map
            kernel_instance_map_ = register_kernel_ptr_->Init(op_kind_, epilogue_op_);
            if (kernel_instance_map_.size() == 0) {
                LI_THROW(Fatal("No GEMM op instances were generated for {}", op_name_));
            }

            // // run compile-time filter
            // std::map<std::string, std::shared_ptr<void>> new_kernel_instance_map;
            // for (const auto& [kernel_config_name, kernel_instance] : kernel_instance_map_) {
            //     if (register_kernel_ptr->FunctionFilter()) {
            //         new_kernel_instance_map[kernel_config_name] = kernel_instance;
            //     }
            // }

            // LOG(INFO) << "Filtered profiler kernels for " << op_name_ << " reduced the number of generated
            // kernels from"
            //           << new_kernel_instance_map.size() << "to" << kernel_instance_map_.size();

            // kernel_instance_map_ = new_kernel_instance_map;
            // if (kernel_instance_map_.size() == 0) {
            //     LI_THROW(Unavailable(
            //         "No GEMM op instances are left after filtering for {}, This is probably due to
            //         incompatible alignment requirements.", op_name_));
            // }

            if (exec_path_[exec_key_[i]]->algo_ == "") {
                // generate profiler
                generated_profilers = register_kernel_ptr_->GenKernelProfiler(context_ptr_->GetName(), GetAttrsMap());
            }
            else {
                LOG(INFO) << "op_name: " << op_name_ << ", " << "workload: " << exec_key_[i]
                          << " from cache, not profile";
            }
        }

        return generated_profilers;
    }

    std::vector<std::string>
    GenOpProfileCmd(const std::string&                                                 profiler_prefix,
                    const std::string&                                                 profiler_filename,
                    const std::string&                                                 exec_key,
                    const std::function<std::vector<std::string>(const std::string&)>& fbuild_cmd = nullptr)
    {
        std::filesystem::path exe_path = std::filesystem::path(profiler_prefix) / profiler_filename;

        if (!CheckExistWithRetries(exe_path, 3, 5)) {
            LI_THROW(Fatal("Profiler {} is not executable", exe_path.string()));
        }

        // k,m,n
        std::vector<std::string> cmd_args = fbuild_cmd(exec_key);

        std::vector<std::string> cmd = {exe_path.string()};

        cmd.insert(cmd.end(), cmd_args.begin(), cmd_args.end());

        // profiling gemm/bmm_permute with layout and shape for ROCM
        bool is_continue = false;
        if (permute_shape_.GetNumDim()) {
            for (const auto& x : permute_shape_.ToVector()) {
                // index 0 is seq dim, choose the second values
                if (is_continue) {
                    cmd.emplace_back(std::to_string(x.GetValues()[0]));
                }
                else {
                    cmd.emplace_back(std::to_string(x.GetValues()[1]));
                    is_continue = true;
                }
            }
        }
        return cmd;
    }

    static std::vector<int64_t> SplitKSearchSpace(const int64_t m, const int64_t n, const int64_t k)
    {
        // Get split_k search range = [1] by default
        std::vector<int64_t> space = {1};
        // split-k search
        int64_t factor     = k / std::max(m, n);
        int64_t low_range  = std::max((int64_t)1, factor / 4);
        int64_t high_range = std::min(factor, (int64_t)32);
        if (low_range == 1) {
            low_range += 1;
        }

        for (int64_t i = low_range; i <= high_range; i += 2) {
            space.emplace_back(i);
        }

        LOG(INFO) << "profiling split-k for gemm instance M=" << m << ", N=" << n << ", K=" << k;
        LOG(INFO) << "split-k search space" << space;
        return space;
    }

    void ProfileSingleWorkload(const std::string&                        profiler_prefix,
                               const std::string&                        workload,
                               const std::shared_ptr<GPUProfilerRunner>& profiler_runner_ptr,
                               bool                                      force_cache)
    {
        std::vector<std::string> kernel_instance_map_key = GetKeyVector(kernel_instance_map_);

        std::string exec_entry_sha1 = SHA1ToHexString(workload);
        auto        query           = GemmQueryEntry(DataTypeToShortString(CppTypeToDataType<CppType>::Type()),
                                    DataTypeToShortString(CppTypeToDataType<CppType>::Type()),
                                    DataTypeToShortString(CppTypeToDataType<CppType>::Type()),
                                    DataTypeToShortString(DataType::FLOAT32),
                                    DataLayoutToString(layout_),
                                    op_name_,
                                    Target::Instance()->GetTargetDeviceName(),
                                    g_short_tensor_operation_names_map.at(epilogue_op_),
                                    exec_entry_sha1,
                                    permute_shape_.ToString());

        auto cache_value = Target::Instance()->QueryProfileCache(GenOperationKind::Gemm, query);

        if (cache_value == std::make_tuple("null", -1) && force_cache) {
            LOG(WARNING) << "force_cache is enabled but we could not find the following cache available on device. "
                         << "op_name: " << op_name_ << " exec_entry_sha1: " << exec_entry_sha1;
        }

        for (const auto& kernel_config_name : kernel_instance_map_key) {
            auto GenCallback = [&](const int split_k_profile = 1) {
                auto process_result_callback =
                    [&](const std::vector<ProfileResult>&           result,
                        const std::shared_ptr<ProfilerPostprocess>& postprocessing_delegate_ptr) {
                        postprocessing_delegate_ptr->AddInstance(result,
                                                                 GenOperationKind::Gemm,
                                                                 GetAttrsMap(),
                                                                 kernel_config_name,
                                                                 workload,
                                                                 split_k_profile);
                    };
                return process_result_callback;
            };

            auto fbuild_cmd = static_cast<OpType*>(this)->GenBuildCmd();

            std::vector<std::string> command =
                GenOpProfileCmd(profiler_prefix, kernel_config_name, workload, fbuild_cmd);

            LOG(INFO) << "profile command: " << JoinToString(command);

            if (StartsWith(op_name_, "split_k")) {
                std::vector<int64_t> k_m_n_values = InverseKeyFunc(workload);  // {k, m, n} using std::map sort

                VLOG(1) << "InverseKeyFunc" << " M:" << k_m_n_values[1] << " N:" << k_m_n_values[2]
                        << " K:" << k_m_n_values[0];
                std::vector<int64_t> split_k_search_space;

                if (split_k_hints_) {
                    split_k_search_space = {split_k_hints_};
                }
                else {
                    split_k_search_space = SplitKSearchSpace(k_m_n_values[1], k_m_n_values[2], k_m_n_values[0]);
                }

                for (auto& split_k : split_k_search_space) {
                    command.emplace_back(std::to_string(split_k));
                    profiler_runner_ptr->Push(command, GenCallback(split_k));
                    command.pop_back();
                }
            }
            else {
                profiler_runner_ptr->Push(command, GenCallback());
            }
        }
    }

    void Profile(const std::shared_ptr<GPUProfilerRunner>& profiler_runner_ptr,
                 const std::string&                        folder_name = "kernel_profile") override
    {
        std::filesystem::path profiler_prefix =
            std::filesystem::path(FLAGS_LI_HOME_PATH) / folder_name / context_ptr_->GetName() / "profiler" / op_name_;

        for (const auto& workload : exec_key_) {
            if (exec_path_[workload]->algo_ == "") {
                if (kernel_instance_map_.size() == 0) {
                    kernel_instance_map_ = register_kernel_ptr_->Init(op_kind_, epilogue_op_);
                }

                ProfileSingleWorkload(profiler_prefix, workload, profiler_runner_ptr, FLAGS_LI_FORCE_PROFILER_CACHE);
            }
            else {
                LOG(INFO) << op_name_ << " from cache, not profile";
            }
        }
    }

    std::string GenOpFunction() override
    {
        return register_kernel_ptr_->GenKernelFunction(GetName(), context_ptr_->GetName(), GetAttrsMap());
    }

    void Forward() override
    {
        static_cast<OpType*>(this)->ForwardImpl();
    }

    std::unordered_map<std::string, std::any> GetAttrsMap()
    {
        std::unordered_map<std::string, std::any> op_attrs_map{{"op_name", op_name_},
                                                               {"kernel_instance_map", kernel_instance_map_},
                                                               {"dim_info_map", this->ExtractDims(true)},
                                                               {"a_dtype", CppTypeToDataType<CppType>::Type()},
                                                               {"b_dtype", CppTypeToDataType<CppType>::Type()},
                                                               {"c_dtype", CppTypeToDataType<CppType>::Type()},
                                                               {"acc_dtype", DataType::FLOAT32},
                                                               {"layout", layout_},
                                                               {"epilogue_op", epilogue_op_},
                                                               {"num_sources", num_sources_},
                                                               {"scale", scale_},
                                                               {"permute_shape", permute_shape_},
                                                               {"exec_path", exec_path_}};
        return op_attrs_map;
    }

    std::string       op_name_     = "gemm";
    GemmOperationKind op_kind_     = GemmOperationKind::Gemm;
    TensorOperation   epilogue_op_ = TensorOperation::PassThrough;
    DataLayout        layout_;

    bool has_profiler_ = true;

    int64_t split_k_       = 1;
    int64_t split_k_hints_ = 4;
    int     num_sources_   = 0;
    float   scale_         = 1.0f;
    Shape   permute_shape_;

    std::map<std::string, std::shared_ptr<ExecItem>> exec_path_;
    std::vector<std::string>                         exec_key_;

    std::shared_ptr<Kernel> register_kernel_ptr_;

    std::map<std::string, std::shared_ptr<void>> kernel_instance_map_;

    std::vector<Variable*> input_var_;
    std::vector<Variable*> output_var_;
};

}  // namespace flashck