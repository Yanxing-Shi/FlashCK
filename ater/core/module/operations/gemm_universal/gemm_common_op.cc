#include "ater/core/module/operations/gemm_universal/gemm_common_op.h"

#include <filesystem>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

#include "ater/core/utils/enforce.h"
#include "ater/core/utils/flags.h"
#include "ater/core/utils/log.h"
#include "ater/core/utils/string_utils.h"

#include "ater/core/module/kernels/kernel_factory.h"
#include "ater/core/module/operations/gemm_universal/gemm_common_utils.h"
#include "ater/core/profiler/alignment.h"
#include "ater/core/profiler/gemm_cache_entry.h"
#include "ater/core/utils/math_utils.h"

ATER_DECLARE_bool(ATER_FORCE_PROFILE);
ATER_DECLARE_bool(ATER_FORCE_PROFILER_CACHE);
ATER_DECLARE_string(ATER_HOME_PATH);

namespace ater {

/*
Base gemm operators
*/

/*
Extracts a mapping between dim names and a list of DimInfo.
    This function will be used in gemm shape inference, gemm padding graph
    transformation, gemm profiling, etc.

    All subclasses must implement this API.

    An example result from gemm_rcr:
    [M,K] * [N, K] = [M,N]
    {
        "M": [
            DimInfo(source=INPUT, tensor_idx=0, dim_idx=0),
            DimInfo(source=OUTPUT, tensor_idx=0, dim_idx=0),
        ],
        "K": [
            DimInfo(source=INPUT, tensor_idx=0, dim_idx=1),
            DimInfo(source=INPUT, tensor_idx=1, dim_idx=1),
        ],
        "N": [
            DimInfo(source=INPUT, tensor_idx=1, dim_idx=0),
            DimInfo(source=OUTPUT, tensor_idx=0, dim_idx=1),
        ],
    }

    Parameters
    ----------
    for_profiling: bool
        Whether this function is used for generating profiling source codes.
        If yes, some DimInfo are simplified. e.g. For gemm, we treat all tensors
        as 2d.
*/

/*

*/
template<typename T>
std::string GemmCommonOp<T>::GenExecKey(const std::map<std::string, std::vector<int>>& name_value_mapping)
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
            ATER_THROW(Fatal("Gemm input has empty dim values: {}", values[0]));
        }
    }

    return JoinToString(key_strs, " && ");
}

/*
Extracts profiling keys and execution conditions for a given dynamic_profiling_strategy.
    This function fills in self._attrs["exec_path"].
    Keys are "exec_key"s, and are used for profiling.
    Values are ItemValues, where "profiling_key" fields are the same as the corresponding keys,
    "exec_cond" fields specify dynamic ranges, and "algo" fields are empty for now.

    e.g. for gemm_rrr, input1=[m, k], input2=[k, n]
    m = 1, k = 128, n = 256.
    self._attrs["exec_path"] = {
        "M==1 && K==128 && N==256" : ItemValue(
            profiling_key="M==1 && K==128 && N==256",
            exec_cond="M==1 && K==128 && N==256",
            algo="",
        )
    }

    e.g. for gemm_rrr, input1=[dynamic_m, k], input2=[k, n]
    dynamic_m >= 1 and dynamic_m <= 1024, dynamic_profiling_strategy = MAX,
    k = 128, n = 256.
    self._attrs["exec_path"] = {
        "M==1024 && K==128 && N==256" : ItemValue(
            profiling_key="M==1024 && K==128 && N==256",
            exec_cond="M>=1 && M<=1024 && K==128 && N==256",
            algo="",
        )
    }

    Parameters
    ----------
    dynamic_profiling_strategy : DynamicProfileStrategy
        See comments for DynamicProfileStrategy.

*/

template<typename T>
void GemmCommonOp<T>::ExtractExecPath(const DynamicProfileStrategy& dynamic_profiling_strategy)
{
    // [M,K] * [N, K] = [M,N]
    // {
    //     "M": [
    //         DimInfo(source=INPUT, tensor_idx=0, dim_idx=0),
    //         DimInfo(source=OUTPUT, tensor_idx=0, dim_idx=0),
    //     ],
    //     "K": [
    //         DimInfo(source=INPUT, tensor_idx=0, dim_idx=1),
    //         DimInfo(source=INPUT, tensor_idx=1, dim_idx=1),
    //     ],
    //     "N": [
    //         DimInfo(source=INPUT, tensor_idx=1, dim_idx=0),
    //         DimInfo(source=OUTPUT, tensor_idx=0, dim_idx=1),
    //     ],
    // }
    std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> dim_info_map = ExtractDims();

    // dynamic shape M:{range_lower, range_upper}, K:{range_lower, range_upper}, N:{range_lower, range_upper}
    std::map<std::string, std::vector<DDim>> dim_map;

    for (auto& [name, dim_infos] : dim_info_map) {
        std::shared_ptr<DimInfo> dim_info = std::make_shared<DimInfo>();
        for (auto& d : dim_infos) {
            if (d->placeholder_) {
                continue;
            }

            if (!dim_info) {
                dim_info = d;
            }
            else if (d->source_ == TensorSource::Input) {
                // input should have priority.
                dim_info = d;
            }
        }

        std::vector<Variable*> var_vec   = dim_info->source_ == TensorSource::Input ? input_var_ : output_var_;
        std::vector<DDim>      dim_shape = var_vec[dim_info->tensor_idx_]->GetShape().ToVector();

        // std::vector<std::vector<int>> tensor_vec{{2, 2}, {2, 2}};

        // std::vector<int> dim_shape = input_tensor_shape[dim_info->tensor_idx_];
        // current_dim_map_[name].reserve(dim_info->dim_idx_.size());

        for (const auto& dim : dim_info->dim_idx_) {
            VLOG(1) << "name:" << name << " value: " << dim_shape[dim];
            dim_map[name].emplace_back(dim_shape[dim]);
        }
    }

    std::map<std::string, std::vector<int>> shape_values_map;

    int initial_product = 1;
    for (const auto& [name, dims] : dim_map) {
        std::vector<int> min_dims_value, max_dims_value;
        // dynamic shape
        for (const auto& dim : dims) {
            min_dims_value.emplace_back(dim.GetLowerBound());
            max_dims_value.emplace_back(dim.GetUpperBound());
        }

        int min_value =
            std::accumulate(min_dims_value.begin(), min_dims_value.end(), initial_product, std::multiplies<int>());
        int max_value =
            std::accumulate(max_dims_value.begin(), max_dims_value.end(), initial_product, std::multiplies<int>());

        std::vector<int> shape_values{min_value, max_value};
        std::sort(shape_values.begin(), shape_values.end());

        shape_values_map[name] = shape_values;

        VLOG(1) << "name: " << name << " min_dims_value: " << min_value;
        VLOG(1) << "name: " << name << " max_dims_value: " << max_value;
    }

    if (dynamic_profiling_strategy == DynamicProfileStrategy::MAX) {
        std::map<std::string, std::vector<int>> max_values;
        for (auto& [name, shape_values] : shape_values_map) {
            int max_shape_values = *max_element(shape_values.begin(), shape_values.end());
            max_values[name]     = {max_shape_values};
        }

        VLOG(1) << "profiling_key: " << GenExecKey(max_values);
        VLOG(1) << "exec_cond: " << GenExecKey(shape_values_map);

        std::shared_ptr<ExecItem> exec_item_ptr =
            std::make_shared<ExecItem>(GenExecKey(max_values), GenExecKey(shape_values_map), "");

        exec_item_ptr->SetGemmExecCondRange(shape_values_map);
        exec_path_[exec_item_ptr->profiling_key_] = exec_item_ptr;
    }
    else if (dynamic_profiling_strategy == DynamicProfileStrategy::MIN) {
        std::map<std::string, std::vector<int>> min_values;
        for (auto& [name, shape_values] : shape_values_map) {
            int min_shape_values = *min_element(shape_values.begin(), shape_values.end());
            min_values[name]     = {min_shape_values};
        }

        std::shared_ptr<ExecItem> exec_item_ptr =
            std::make_shared<ExecItem>(GenExecKey(min_values), GenExecKey(shape_values_map), "");
        exec_path_[exec_item_ptr->profiling_key_] = exec_item_ptr;
    }
    else {
        ATER_THROW(Unimplemented("{}", "Gemm only supports MIN or MAX dynamic profiling"));
    }
}

// check if profile cache exits If we have a cached
// entry for this gemm instance, we update this gemm op's
// relevant attributes with the cached result and return False.

template<typename T>
bool GemmCommonOp<T>::IfShouldBuildProfiler(const std::unordered_set<std::string>& workloads)
{
    // We are forced to use the cache, so we skip building profilers.
    if (FLAGS_ATER_FORCE_PROFILER_CACHE) {
        return false;
    }

    bool build_profiler = true;

    // Now, let's query if all of our workloads have cache entries. If that
    // is the case, it is safely to skip generating and building profilers.
    std::vector<std::string> kernel_instance_map_key;
    for (const auto& [kernel_config_name, _] : kernel_instance_map_) {
        kernel_instance_map_key.emplace_back(kernel_config_name);
    }
    static int kernel_instnace_idx = 0;
    auto       tmp_key             = *std::next(kernel_instance_map_key.begin(), kernel_instnace_idx++);
    auto       tmp_kernel_instance = std::static_pointer_cast<GemmOperation>(kernel_instance_map_[tmp_key]);
    for (const auto& wkl : workloads) {
        std::string exec_entry_sha1 = SHA1ToHexString(wkl);
        auto        query           = GemmQueryEntry(static_cast<int>(tmp_kernel_instance->a_tensor_desc_.element_),
                                    static_cast<int>(tmp_kernel_instance->b_tensor_desc_.element_),
                                    static_cast<int>(tmp_kernel_instance->c_tensor_desc_.element_),
                                    static_cast<int>(tmp_kernel_instance->accumulator_type_),
                                    static_cast<int>(tmp_kernel_instance->a_tensor_desc_.layout_),
                                    static_cast<int>(tmp_kernel_instance->b_tensor_desc_.layout_),
                                    static_cast<int>(tmp_kernel_instance->c_tensor_desc_.layout_),
                                    op_name_,                                   // op_type
                                    Target::Instance()->GetTargetDeviceName(),  // device_name
                                    static_cast<int>(tmp_kernel_instance->epilogue_functor_),  // epilogue functor
                                    exec_entry_sha1,
                                    permute_shape_);  // permute shape
        std::unordered_map<std::string, std::variant<int, std::string>> query_value_map = query.GetAttrsMap();
        std::tuple<std::string, int, int> cache_value = Target::Instance()->QueryProfileCache("gemm", query_value_map);

        if (cache_value != std::make_tuple("null", -1, -1) && !FLAGS_ATER_FORCE_PROFILE) {
            std::string best_algo = std::get<0>(cache_value);
            int         workspace = std::max(workspace_, std::get<1>(cache_value));
            int         split_k   = std::get<2>(cache_value);
            LOG(INFO) << "Load profiling result for" << op_name_ << "from cache, algo" << best_algo << "workspace"
                      << workspace << "split_k" << split_k;
        }
        else {
            // cache miss - we will have to generate and build profilers
            build_profiler = true;
        }
    }
    return build_profiler;
}

// Generate profilers for this gemm op
template<typename T>
std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
GemmCommonOp<T>::GenOpProfiler(const DynamicProfileStrategy& dynamic_profiling_strategy)
{
    // init exec path
    ExtractExecPath(dynamic_profiling_strategy);

    // init compile-time filter
    for (const auto& [exec_key, _] : exec_path_) {
        all_workloads_.emplace(exec_key);
    }

    // check if all workloads have the same ab_alignment
    std::vector<int> ab_alignments;
    std::transform(all_workloads_.begin(),
                   all_workloads_.end(),
                   std::back_inserter(ab_alignments),
                   [&](const std::string& exec_key) { return GetABAlignment(exec_key); });

    bool flag =
        std::all_of(ab_alignments.begin(), ab_alignments.end(), [&](const int& x) { return x == ab_alignments[0]; });

    ATER_ENFORCE_EQ(
        flag,
        true,
        Unavailable("ab_alignments should be the same among all workloads, got {}", JoinToString(ab_alignments)));

    // std::vector<std::vector<int>> k_m_n_values;
    // std::transform(all_workloads_.begin(),
    //                all_workloads_.end(),
    //                std::back_inserter(k_m_n_values),
    //                [&](const std::string& exec_key) { return GemmInverseKeyFunc(exec_key); });

    // generate kernel instances
    std::vector<int> inverse_res =
        GemmInverseKeyFunc(*all_workloads_.begin());  // current workloads not all workloads, k,m,n using std::map

    GemmProblem gemm_problem(inverse_res[1],
                             inverse_res[2],
                             inverse_res[0],
                             CppTypeToDataType<T>::Type(),
                             CppTypeToDataType<T>::Type(),
                             CppTypeToDataType<T>::Type(),
                             DataType::FLOAT32,
                             {},
                             CppTypeToDataType<T>::Type(),
                             layout_,
                             {},
                             epilogue_op_);
    Target::Instance()->GenerateKernel(GenOperationKind::Gemm, gemm_problem);

    // init candidate ops
    KernelKey kernel_key(SourceType::CK, layout_, CppTypeToDataType<T>::Type());
    auto      register_kernel_ptr = KernelFactory::Instance().SelectKernel(op_name_, kernel_key);

    // init kernel instance map
    kernel_instance_map_ = register_kernel_ptr->Init();
    if (kernel_instance_map_.size() == 0) {
        ATER_THROW(Fatal("No GEMM op instances were generated for {}", op_name_));
    }

    // run compile-time filter
    std::map<std::string, std::shared_ptr<void>> new_kernel_instance_map;
    for (const auto& [kernel_config_name, kernel_instance] : kernel_instance_map_) {
        if (register_kernel_ptr->FunctionFilter()) {
            new_kernel_instance_map[kernel_config_name] = kernel_instance;
        }
    }

    LOG(INFO) << "Filtered profiler kernels for " << op_name_ << " reduced the number of generated kernels from"
              << new_kernel_instance_map.size() << "to" << kernel_instance_map_.size();

    kernel_instance_map_ = new_kernel_instance_map;
    if (kernel_instance_map_.size() == 0) {
        ATER_THROW(Unavailable(
            "No GEMM op instances are left after filtering for {}, This is probably due to incompatible alignment requirements.",
            op_name_));
    }

    // generate profiler
    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> generated_profilers{};
    if (IfShouldBuildProfiler(all_workloads_)) {
        generated_profilers = register_kernel_ptr->GenKernelProfiler(
            op_name_, context_ptr_->GetName(), kernel_instance_map_, ExtractDims(true));
    }

    return generated_profilers;
}

template<typename T>
std::vector<std::string>
GemmCommonOp<T>::GenOpProfileCmd(const std::string&                                                 profiler_prefix,
                                 const std::string&                                                 profiler_filename,
                                 const std::string&                                                 exec_key,
                                 const std::function<std::vector<std::string>(const std::string&)>& fbuild_cmd)
{
    std::filesystem::path exe_path = std::filesystem::path(profiler_prefix) / profiler_filename;

    if (!CheckWithRetries(exe_path, 3, 5)) {
        ATER_THROW(Fatal("Profiler {} is not executable", exe_path.string()));
    }

    // k,m,n
    std::vector<std::string> cmd_args  = fbuild_cmd(exec_key);
    std::vector<std::string> cmd_args_ = {cmd_args[1], cmd_args[2], cmd_args[0]};

    std::vector<std::string> cmd = {exe_path.string()};

    // m n k
    cmd.insert(cmd.end(), cmd_args_.begin(), cmd_args_.end());

    // profiling gemm/bmm_permute with layout and shape for ROCM
    if (!permute_shape_.empty()) {
        for (const auto& x : permute_shape_) {
            cmd_args.emplace_back(std::to_string(x));
        }
    }
    return cmd;
}

template<typename T>
int GemmCommonOp<T>::GetABAlignment(const std::string& exec_key)
{
    // if (StartsWith(op_name_, "group_gemm")) {
    //     std::vector<int> all_m, all_n, all_k;
    //     std::tie(all_m, all_n, all_k) = GroupGemmInverseKeyFunc(exec_key);
    //     std::vector<int> all_ab_alignments;
    //     for (int i = 0; i < all_m.size(); i++) {
    //         int m = all_m[i];
    //         int n = all_n[i];
    //         int k = all_k[i];
    //         all_ab_alignments.emplace_back(ab_alignment_func_(m, n, k));
    //     }
    //     ab_alignment = std::max(all_ab_alignments);
    // }

    // exec_key may contain batch dimension, which we don't care here
    std::vector<int> inverse_res  = GemmInverseKeyFunc(exec_key);  // {k, m, n} using std::map
    int              ab_alignment = ab_alignment_func_(inverse_res[1], inverse_res[2], inverse_res[0]);
    if (!IsValidAlignment(ab_alignment, CppTypeToDataType<T>::Type())) {
        ATER_THROW(
            Unavailable("A / B {} is not valid! The last dimension of each input tensor needs to be divisible by 2.",
                        ab_alignment));
    }

    return ab_alignment;
}

template<typename T>
void GemmCommonOp<T>::ExtractEpilogueAlignment(const std::vector<int>&       ouput_shape,
                                               const DynamicProfileStrategy& dynamic_profiling_strategy)
{
    int epilogue_dim = ouput_shape.back();
    int shape        = epilogue_dim;

    // dynamic shape
    // if (dynamic_profiling_strategy == DynamicProfileStrategy::MAX) {
    //     shape = *std::max_element(epilogue_dim.begin(), epilogue_dim.end());
    // }
    // else if (dynamic_profiling_strategy == DynamicProfileStrategy::MIN) {
    // }
    // else {
    // }

    epilogue_alignment_ = FindMaxAlignment(shape, CppTypeToDataType<T>::Type());
}

template<typename T>
std::vector<int> GemmCommonOp<T>::SplitKSearchSpace(const int m, const int n, const int k)
{
    // Get split_k search range = [1] by default
    std::vector<int> space = {1};
    // split-k search
    int factor     = k / std::max(m, n);
    int low_range  = std::max(1, factor / 4);
    int high_range = std::min(factor, 32);
    if (low_range == 1) {
        low_range += 1;
    }

    for (int i = low_range; i <= high_range; i += 2) {
        space.emplace_back(i);
    }

    LOG(INFO) << Sprintf("profiling split-k for gemm instance M={}, N={}, K={}");
    // LOG(INFO) << "split-k search space" << space;
    return space;
}

template<typename T>
void GemmCommonOp<T>::ProfileSingleWorkload(const std::string&                        profiler_prefix,
                                            const std::string&                        exec_key,
                                            const std::shared_ptr<GPUProfilerRunner>& profiler_runner_ptr,
                                            bool                                      force_cache)
{
    std::vector<std::string> kernel_instance_map_key;
    for (const auto& [kernel_config_name, _] : kernel_instance_map_) {
        kernel_instance_map_key.emplace_back(kernel_config_name);
    }

    static int kernel_name_idx         = 0;
    auto       tmp_kernel_instance_key = *std::next(kernel_instance_map_key.begin(), kernel_name_idx++);

    VLOG(1) << "tmp_kernel_instance_key: " << tmp_kernel_instance_key;

    std::shared_ptr<GemmOperation> tmp_kernel_instance =
        std::static_pointer_cast<GemmOperation>(kernel_instance_map_[tmp_kernel_instance_key]);

    std::string exec_entry_sha1 = SHA1ToHexString(exec_key);
    auto        query           = GemmQueryEntry(static_cast<int>(tmp_kernel_instance->a_tensor_desc_.element_),
                                static_cast<int>(tmp_kernel_instance->b_tensor_desc_.element_),
                                static_cast<int>(tmp_kernel_instance->c_tensor_desc_.element_),
                                static_cast<int>(tmp_kernel_instance->accumulator_type_),
                                static_cast<int>(tmp_kernel_instance->a_tensor_desc_.layout_),
                                static_cast<int>(tmp_kernel_instance->b_tensor_desc_.layout_),
                                static_cast<int>(tmp_kernel_instance->c_tensor_desc_.layout_),
                                op_name_,
                                Target::Instance()->GetTargetDeviceName(),
                                static_cast<int>(tmp_kernel_instance->epilogue_functor_),
                                exec_entry_sha1,
                                permute_shape_);

    std::unordered_map<std::string, std::variant<int, std::string>> query_value_map = query.GetAttrsMap();

    std::tuple<std::string, int, int> cache_value = Target::Instance()->QueryProfileCache("gemm", query_value_map);

    if (cache_value != std::make_tuple("null", -1, -1) && !FLAGS_ATER_FORCE_PROFILE) {
        // fmt::print("Load profiling result for {op_name} from {cache_value}",
        //            fmt::arg("op_name", op_name_),
        //            fmt::arg("cache_value", JoinToString(cache_value)));
        exec_path_[exec_key]->algo_ = std::get<0>(cache_value);
        workspace_                  = std::max(workspace_, std::get<1>(cache_value));
        split_k_                    = std::get<2>(cache_value);
        return;
    }

    if (cache_value == std::make_tuple("null", -1, -1) && force_cache) {
        LOG(WARNING) << "force_cache is enabled but we could not find the following cache available on device. "
                     << "op_name: " << op_name_ << " exec_entry_sha1: " << exec_entry_sha1;
    }

    for (const auto& kernel_config_name : kernel_instance_map_key) {
        auto GenCallback = [&] {
            auto process_result_callback =
                [&](const std::vector<ProfileResult>&                          result,
                    const std::shared_ptr<GemmProfilerPostprocessingDelegate>& postprocessing_delegate_ptr) {
                    postprocessing_delegate_ptr->AddInstance(
                        result, GetAttrsMap(), kernel_config_name, exec_key, split_k_);
                };
            return process_result_callback;
        };

        std::vector<std::string> command = GenProfileCmd(profiler_prefix, kernel_config_name, exec_key);

        LOG(INFO) << "profile command: " << JoinToString(command);

        profiler_runner_ptr->Push(command, GenCallback());

        // if (StartsWith(op_name_, "group_gemm") || StartsWith(op_name_, "bmm")) {
        //     profiler_runner_ptr->Push(command, GenCallback());
        // }
        // else {
        //     std::vector<std::string> inverse_res = GemmInverseKeyFunc(exec_key);  // {k, m, n} using std::map
        //     sort

        //     VLOG(1) << "GemmInverseKeyFunc"
        //             << " M:" << inverse_res[1] << " N:" << inverse_res[2] << " K:" << inverse_res[0];
        //     std::vector<int> split_k_search_space;

        //     if (split_k_hints_) {
        //         split_k_search_space = {split_k_hints_};
        //     }
        //     else {
        //         split_k_search_space =
        //             SplitKSearchSpace(std::stoi(inverse_res[1]), std::stoi(inverse_res[2]),
        //             std::stoi(inverse_res[0]));
        //     }

        //     for (auto& split_k : split_k_search_space) {
        //         command.emplace_back(std::to_string(split_k));
        //         profiler_runner_ptr->Push(command, GenCallback(split_k));
        //     }
        // }
    }
}

template<typename T>
void GemmCommonOp<T>::Profile(const std::shared_ptr<GPUProfilerRunner>& profiler_runner_ptr,
                              const std::string&                        folder_name)
{
    std::vector<std::string> workloads;
    for (const auto& [profiling_key, _] : exec_path_) {
        workloads.emplace_back(profiling_key);
    }

    std::filesystem::path profiler_prefix =
        std::filesystem::path(FLAGS_ATER_HOME_PATH) / folder_name / context_ptr_->GetName() / "profiler" / op_name_;
    // if (kernel_instance_map_.size() == 0) {
    //     auto kernel_ptr = KernelFactory::Get().ProduceShared(op_name_);
    //     kernel_ptr->Init(target_ptr);
    // }

    for (const auto& workload : workloads) {
        LOG(INFO) << "Profile: " << op_name_ << " : " << workload;
        // we have cached best algo
        if (exec_path_[workload]->algo_ != "") {
            LOG(WARNING) << "Profile: " << op_name_ << ":" << workload << " already has algo"
                         << exec_path_[workload]->algo_ << ", skip profiling";
            return;
        }
        else {
            ProfileSingleWorkload(profiler_prefix, workload, profiler_runner_ptr, FLAGS_ATER_FORCE_PROFILER_CACHE);
        }
    }
}

template<typename T>
void GemmCommonOp<T>::SanityCheck(Variable* a, Variable* b)
{
    Shape a_shape = a->GetShape();
    Shape b_shape = b->GetShape();
    if (a_shape.ToVector().size() < 2) {
        ATER_THROW(Unavailable("Input A shape should be at least 2D, got {}", a_shape.ToString()));
    }

    if (b_shape.ToVector().size() < 2) {
        ATER_THROW(Unavailable("Input B shape should be at least 2D, got {}", b_shape.ToString()));
    }

    if (a->GetDtype() != b->GetDtype()) {
        ATER_THROW(Unavailable("Input A and B should have the same data type, got {} and {}",
                               DataTypeToString(a->GetDtype()),
                               DataTypeToString(b->GetDtype())));
    }
}

template<typename T>
Variable* GemmCommonOp<T>::operator()(Variable* a, Variable* b)
{
    AlignAB(a, b);
    SanityCheck(a, b);
    input_var_         = {a, b};
    Shape output_shape = InferShape(a, b);
    VLOG(1) << "output_shape: " << output_shape.ToString();
    // ExtractEpilogueAlignment(output_shape_vec);
    int max_output_size = std::get<1>(output_shape.GetElementSizeTuple());
    output_var_ = {new Variable(op_name_ + std::string("_output"), max_output_size, CppTypeToDataType<T>::Type())};
    output_var_[0]->SetShape(output_shape);

    SetParentsNode({a, b});
    this->SetChildrenNode({static_cast<Node*>(output_var_[0])});

    return output_var_[0];
}

template<typename T>
std::string GemmCommonOp<T>::GenOpFunction()
{
    // init candidate ops
    KernelKey kernel_key(SourceType::CK, layout_, CppTypeToDataType<T>::Type());
    auto      register_kernel_ptr = KernelFactory::Instance().SelectKernel(op_name_, kernel_key);

    std::string best_func;
    if (IfShouldBuildProfiler(all_workloads_)) {
        best_func = register_kernel_ptr->GenKernelFunction(
            GetName(), kernel_instance_map_, exec_path_, permute_shape_, ExtractDims(false));
    }

    return best_func;
}

template<typename T>
void GemmCommonOp<T>::Forward()
{
    T* a_ptr = (T*)GetParentNode(0)->GetValue();
    // int a_dim0 = GetParentNode(0)->GetShape().GetDim(0);
    // int a_dim1 = GetParentNode(0)->GetShape().GetDim(1);

    T* b_ptr = (T*)GetParentNode(1)->GetValue();
    // int b_dim0 = GetParentNode(1)->GetShape().GetDim(0);
    // int b_dim1 = GetParentNode(1)->GetShape().GetDim(1);

    T* c_ptr = (T*)GetChildNode(0)->GetValue();
    // int c_dim0 = GetChildNode(0)->GetShape().GetDim(0);
    // int c_dim1 = GetChildNode(0)->GetShape().GetDim(1);

    hipStream_t stream = context_ptr_->GetStream();
    if (!context_ptr_->IsBuilt()) {
        return;
    }

    // extract actual shape
    Shape a_shape = GetParentNode(0)->GetShape();
    Shape b_shape = GetParentNode(1)->GetShape();
    Shape c_shape = InferShape(GetParentNode(0), GetParentNode(1));
    output_var_[0]->SetShape(c_shape);  // must update actual output shape

    std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> dim_info_map = ExtractDims();
    std::map<std::string, int>                                   dim_map;

    for (auto& [name, dim_infos] : dim_info_map) {
        std::shared_ptr<DimInfo> dim_info = std::make_shared<DimInfo>();
        for (auto& d : dim_infos) {
            if (d->placeholder_) {
                continue;
            }

            if (!dim_info) {
                dim_info = d;
            }
            else if (d->source_ == TensorSource::Input) {
                // input should have priority.
                dim_info = d;
            }
        }

        std::vector<Variable*> var_vec   = dim_info->source_ == TensorSource::Input ? input_var_ : output_var_;
        std::vector<DDim>      dim_shape = var_vec[dim_info->tensor_idx_]->GetShape().ToVector();

        for (const auto& dim : dim_info->dim_idx_) {
            VLOG(1) << "name:" << name << " value: " << dim_shape[dim];
            dim_map[name] = dim_shape[dim].GetValues()[0];
        }
    }

    int m = dim_map["M"];
    int n = dim_map["N"];
    int k = dim_map["K"];

    std::string algo = "";
    // need to fix, random choose one algo now
    for (const auto& [_, value] : exec_path_) {
        if (value->GetGemmAlgo(m, n, k) != "") {
            algo = value->GetGemmAlgo(m, n, k);
        }
    }

    ATER_ENFORCE_NE(algo, "", Unavailable("No algo found for gemm instance M={}, N={}, K={}", m, n, k));

    std::string config;
    std::string config_name;
    auto        kernel_instance      = kernel_instance_map_.at(algo);
    auto        gemm_kernel_instance = std::static_pointer_cast<GemmOperation>(kernel_instance);
    int         block_size           = gemm_kernel_instance->tile_desc_.block_size_;
    int         m_per_block          = gemm_kernel_instance->tile_desc_.m_per_block_;
    int         n_per_block          = gemm_kernel_instance->tile_desc_.n_per_block_;
    auto        grid_size            = IntegerDivideCeil(m, m_per_block) * IntegerDivideCeil(n, n_per_block);

    VLOG(1) << "grid_size: " << grid_size << " block_size: " << block_size;

    KernelKey kernel_key(SourceType::CK, layout_, CppTypeToDataType<T>::Type());
    auto      register_kernel_ptr = KernelFactory::Instance().SelectKernel(op_name_, kernel_key);

    register_kernel_ptr->LaunchKernel(
        GetName(), (size_t)(grid_size * block_size), (size_t)block_size, stream, context_ptr_->GetName())(
        a_ptr, b_ptr, c_ptr, m, n, k);
}

template<typename T>
std::unordered_map<std::string, std::any> GemmCommonOp<T>::GetAttrsMap()
{
    std::unordered_map<std::string, std::any> op_attrs_map{{"op_name", op_name_},
                                                           {"workspace", workspace_},
                                                           {"split_k", split_k_},
                                                           {"split_k_hints", split_k_hints_},
                                                           {"num_sources", num_sources_},
                                                           {"alpha", alpha_},
                                                           {"permute_shape", permute_shape_},
                                                           {"exec_path", exec_path_},
                                                           {"kernel_instance_map", kernel_instance_map_}};
    return op_attrs_map;
}

template class GemmCommonOp<float>;
template class GemmCommonOp<_Float16>;
}  // namespace ater
