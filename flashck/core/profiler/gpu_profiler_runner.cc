
#include "flashck/core/profiler/gpu_profiler_runner.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <functional>

#include "flashck/core/utils/enforce.h"
#include "flashck/core/utils/flags.h"
#include "flashck/core/utils/log.h"
#include "flashck/core/utils/printf.h"
#include "flashck/core/utils/string_utils.h"

#include "flashck/core/graph/shape.h"
#include "flashck/core/profiler/base.h"
#include "flashck/core/profiler/target.h"

namespace flashck {

struct InstanceData {
    std::vector<ProfileResult>                results;
    GenOperationKind                          op_kind;
    std::unordered_map<std::string, std::any> op_attrs;
    std::string                               kernel_name;
    std::string                               exec_key;
    int64_t                                   split_k;
};

void ProfilerPostprocess::AddInstance(std::vector<ProfileResult>                       result,
                                      GenOperationKind                                 op_kind,
                                      const std::unordered_map<std::string, std::any>& op_attrs,
                                      std::string                                      kernel_name,
                                      std::string                                      exec_key,
                                      int64_t                                          split_k)
{
    instances_.emplace_back(std::move(result), op_kind, op_attrs, std::move(kernel_name), std::move(exec_key), split_k);
}

struct ProfileResultGroupByKey {
    using KeyTuple = std::tuple<GenOperationKind, std::string, std::string>;

    KeyTuple operator()(const InstanceData& instance) const
    {
        return {instance.op_kind, std::any_cast<std::string>(instance.op_attrs.at("op_kind")), instance.exec_key};
    }
};

void ProfilerPostprocess::PostProcessResults()
{
    using InstanceIter = decltype(instances_)::iterator;
    using GroupType    = GroupedResults<InstanceIter, ProfileResultGroupByKey>;

    const auto instance_groups = GroupByFunc(instances_.begin(), instances_.end(), ProfileResultGroupByKey{});

    for (const auto& group : instance_groups) {
        if (group.range.empty())
            continue;

        const auto& [best_results, op_kind, op_attrs, kernel_name, exec_key, split_k] =
            *std::min_element(group.range.begin(), group.range.end(), [](const auto& a, const auto& b) {
                return a.results.duration < b.results.duration;
            });

        for (const auto& instance : group.range) {
            auto& exec_item     = std::any_cast<ExecItemMap&>(instance.op_attrs.at("exec_path"))[exec_key];
            exec_item->algo_    = best_results.kernel_config;
            exec_item->split_k_ = split_k;
        }

        LOG(INFO) << fmt::format("Profiler {} selected kernel {} ({} ms) split_k:{}",
                                 kernel_name,
                                 best_results.kernel_config,
                                 best_results.duration,
                                 split_k);

        if (op_kind == GenOperationKind::Gemm) {
            try {
                const auto cache_record = CreateGemmCacheRecord(op_attrs, exec_key, best_results, split_k);
                Target::Instance().InsertProfileCache(op_kind, cache_record);
            }
            catch (const std::exception& e) {
                LI_THROW(Unavailable("Cache update failed for {}: {}", exec_key, e.what()));
            }
        }
    }
}

GemmRecordEntry ProfilerPostprocess::CreateGemmCacheRecord(const AttrMap&       op_attrs,
                                                           const std::string&   exec_key,
                                                           const ProfileResult& result,
                                                           int64_t              split_k) const
{
    const auto get_attr = [&](auto key) { return std::any_cast<std::decay_t<decltype(key)>>(op_attrs.at(key)); };

    return {exec_key,
            SHA1ToHexString(exec_key),
            DataTypeToShortString(get_attr("a_dtype")),
            DataTypeToShortString(get_attr("b_dtype")),
            DataTypeToShortString(get_attr("c_dtype")),
            DataTypeToShortString(get_attr("acc_dtype")),
            DataLayoutToString(get_attr("layout")),
            get_attr("op_name"),
            g_short_tensor_operation_names_map.at(get_attr("epilogue_op")),
            get_attr("permute_shape").ToString(),
            Target::Instance()->GetTargetDeviceName(),
            result.kernel_config,
            split_k};
}

GPUProfilerRunner::GPUProfilerRunner(const int                                   num_gpu,
                                     const std::shared_ptr<ProfilerPostprocess>& postprocessing_delegate_ptr,
                                     const int                                   timeout):
    num_gpu_(num_gpu), timeout_(timeout), postprocessing_ptr_(postprocessing_delegate_ptr)
{
}

void GPUProfilerRunner::Join()
{
    postprocessor_->PostProcessResults();
}

}  // namespace flashck