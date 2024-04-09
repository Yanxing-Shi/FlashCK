
#include "ater/core/profiler/gemm_gpu_profiler_runner.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <functional>

#include "ater/core/utils/flags.h"
#include "ater/core/utils/log.h"
#include "ater/core/utils/printf.h"

#include "ater/core/profiler/base.h"
#include "ater/core/profiler/gemm_cache_entry.h"
#include "ater/core/profiler/gemm_operation.h"
#include "ater/core/profiler/target.h"
#include "ater/core/utils/enforce.h"
#include "ater/core/utils/string_utils.h"

ATER_DECLARE_int32(dist_threadpool_size);

namespace ater {

/*
Object which collects profiler results after profiler executables complete,
updates profiler results cache and the gemm nodes' attrs after all profilers complete.
*/

// Initialize storage for profiler results
// Instance=(
//     ProfileResult=(best_algo, elapsed_runtime, workspace),
//     func_attrs,
//     profiler_filename,
//     exec_key,
//     split_k,
// )

// As a profiler executable completes, collect the result
void GemmProfilerPostprocessingDelegate::AddInstance(const std::vector<ProfileResult>&                result,
                                                     const std::unordered_map<std::string, std::any>& op_attrs_map,
                                                     const std::string&                               kernel_name,
                                                     const std::string&                               exec_key,
                                                     const int                                        split_k)
{
    instances_.emplace_back(std::make_tuple(result, op_attrs_map, kernel_name, exec_key, split_k));
}

/*
When all profiler executables complete, find the best instance
(min runtime per op name, profiler executable and exec_key (i.e. gemm shape mnk)
across multiple split_k values)
The best instance is cached, and written into corresponding gemm nodes in the graph
*/

struct ProfileResultGroupByKey {
    typedef std::tuple<std::string, std::string> value_type;
    value_type                                   operator()(
        const std::
            tuple<std::vector<ProfileResult>, std::unordered_map<std::string, std::any>, std::string, std::string, int>&
                instance) const
    {
        return std::make_tuple(std::any_cast<std::string>(std::get<1>(instance).at("op_name")),  // unique op name
                               std::get<3>(instance));  // profiler key (gemm shape)
    }
};

void GemmProfilerPostprocessingDelegate::PostProcessResults()
{
    auto instance_group_by_key = GroupByFunc(instances_.begin(), instances_.end(), ProfileResultGroupByKey());

    for (auto group : instance_group_by_key) {
        // select minimal duration algo
        auto min_runtime_results = *std::min_element(
            group.range.begin(),
            group.range.end(),
            [](const std::tuple<std::vector<ProfileResult>,
                                std::unordered_map<std::string, std::any>,
                                std::string,
                                std::string,
                                int>& a,
               const std::tuple<std::vector<ProfileResult>,
                                std::unordered_map<std::string, std::any>,
                                std::string,
                                std::string,
                                int>& b) { return std::get<0>(a)[0].duration < std::get<0>(b)[0].duration; });
        // only gemm profile result
        ProfileResult best_gemm_profile_result = std::get<0>(min_runtime_results)[0];

        std::unordered_map<std::string, std::any> op_attrs_map      = std::get<1>(min_runtime_results);
        std::string                               profiler_filename = std::get<2>(min_runtime_results);
        std::string                               exec_key          = std::get<3>(min_runtime_results);
        int                                       split_k           = std::get<4>(min_runtime_results);

        for (auto profile_instance : group.range) {
            auto op_attrs_map = std::get<1>(profile_instance);
            std::any_cast<std::map<std::string, std::shared_ptr<ExecItem>>>(op_attrs_map["exec_path"])[exec_key]
                ->algo_ = best_gemm_profile_result.kernel_config;
            op_attrs_map["workspace"] =
                std::max(std::any_cast<int>(op_attrs_map["workspace"]), best_gemm_profile_result.workspace);
            op_attrs_map["split_k"] = split_k;
        }

        LOG(INFO) << "Profiler " << profiler_filename << " selected kernel " << best_gemm_profile_result.kernel_config
                  << " with runtime " << best_gemm_profile_result.duration << " ms and workspace "
                  << best_gemm_profile_result.workspace << " bytes";

        auto kernel_instance_map =
            std::any_cast<std::map<std::string, std::shared_ptr<void>>>(op_attrs_map["kernel_instance_map"]);
        std::vector<std::shared_ptr<void>> kernel_instance_map_values;
        for (const auto& [_, kernel_instance] : kernel_instance_map) {
            kernel_instance_map_values.emplace_back(kernel_instance);
        }

        static int kernel_instance_map_idx = 0;
        auto       tmp_kernel_instance     = std::static_pointer_cast<GemmOperation>(
            *std::next(kernel_instance_map_values.begin(), kernel_instance_map_idx++));

        std::string exec_entry_sha1 = SHA1ToHexString(exec_key);
        auto        cache_record    = GemmRecordEntry(exec_key,
                                            exec_entry_sha1,
                                            static_cast<int>(tmp_kernel_instance->a_tensor_desc_.element_),
                                            static_cast<int>(tmp_kernel_instance->b_tensor_desc_.element_),
                                            static_cast<int>(tmp_kernel_instance->c_tensor_desc_.element_),
                                            static_cast<int>(tmp_kernel_instance->accumulator_type_),
                                            static_cast<int>(tmp_kernel_instance->a_tensor_desc_.layout_),
                                            static_cast<int>(tmp_kernel_instance->b_tensor_desc_.layout_),
                                            static_cast<int>(tmp_kernel_instance->c_tensor_desc_.layout_),
                                            std::any_cast<std::string>(op_attrs_map["op_name"]),
                                            static_cast<int>(tmp_kernel_instance->epilogue_functor_),
                                            std::any_cast<std::string>(op_attrs_map["permute_shape"]),
                                            Target::Instance()->GetTargetDeviceName(),
                                            best_gemm_profile_result.kernel_config,
                                            best_gemm_profile_result.workspace,
                                            split_k);
        try {
            Target::Instance()->InsertProfileCache("gemm", cache_record.GetAttrsMap());
        }
        catch (const std::exception& e) {
            ATER_THROW(Unavailable("Failed to update gemm record cache:{}", e.what()));
        }
    }
}

/*
Parameters
----------
devices : List[str]
    device identifiers (contents of HIP_VISIBLE_DEVICES)
postprocessing_delegate :
    object responsible for postprocessing results after futures completion
timeout : int
    timeout to wait for all profilers completion in seconds
*/
GPUProfilerRunner::GPUProfilerRunner(
    const int                                                  num_gpu,
    const std::shared_ptr<GemmProfilerPostprocessingDelegate>& postprocessing_delegate_ptr,
    const int                                                  timeout):
    num_gpu_(num_gpu), timeout_(timeout), postprocessing_delegate_ptr_(postprocessing_delegate_ptr)
{

    // Get the current list of visible devices
    // num_gpu_ = Target::Instance()->GetTaregtGPUDeviceCount();  // pool_size_ = num_gpu

    for (int i = 0; i < num_gpu_; i++) {
        device_queue_.push(std::to_string(i));
    }
}

/*
Schedule the profiler for execution in a separate process,
    Call the callback after subprocess completion

    Parameters
    ----------
    cmds : List[str]
        argv for the launched profiler
    process_result_callback : Callable
        Called after subprocess completion in the main process
        (but possibly not main thread).
        Currently used to aggregate profiler results,
        so the callable takes `result` and `postprocessing_delegate` parameters
        It is also used to propagate the profiler launch context to the aggregation point,
        namely, split_k value for the gemm profilers
*/
void GPUProfilerRunner::Push(
    const std::vector<std::string>&                                                        cmds,
    const std::function<void(const std::vector<ProfileResult>&,
                             const std::shared_ptr<GemmProfilerPostprocessingDelegate>&)>& process_result_callback)
{
    auto run_task = [this, &cmds, &process_result_callback]() {
        // std::string device_id = device_queue_.front();
        // VLOG(1) << "device_id:" << device_id;
        // device_queue_.pop();
        LOG(INFO) << "running profiler " << JoinToString(cmds);
        int attempts = 0;
        try {
            subprocess::Popen popen(cmds, subprocess::output{subprocess::PIPE}, subprocess::error{subprocess::PIPE});

            std::string stdout_str = popen.communicate().first.buf.data();
            std::string stderr_str = popen.communicate().second.buf.data();
            if (stdout_str.empty() && stderr_str.empty()) {
                LOG(ERROR) << "Profiler failed to run, stdout or stderr is empty";
            }
            else {
                LOG(INFO) << "Profiler stdout: " << stdout_str;
                // collect profiler results for postprocessing
                std::vector<ProfileResult> profile_result;
                bool                       is_error = false;
                std::tie(profile_result, is_error)  = ExtractProfileResult(stdout_str);
                if (is_error) {
                    LOG(ERROR) << "Profiler failure!\nProfiler stdout: " << stdout_str
                               << "\nProfiler stderr: " << stderr_str;
                }
                process_result_callback(profile_result, postprocessing_delegate_ptr_);
            }
        }
        catch (std::exception& e) {
            attempts += 1;
            if (attempts >= g_profiler_run_max_attempts) {
                ATER_THROW(ExecutionTimeout(
                    "[{} / {}] Failed to run profiler {} due to exception: {}. Will retry in {} seconds.",
                    attempts,
                    g_profiler_run_max_attempts,
                    g_profiler_run_retry_delay_seconds,
                    JoinToString(cmds),
                    e.what()));
            }
            std::this_thread::sleep_for(std::chrono::seconds(g_profiler_run_retry_delay_seconds));
        }
    };

    run_task();
}

/*
Wait for subprocesses completion or timeout; postprocess the profiler results with delegate(s)
*/
void GPUProfilerRunner::Join()
{
    // Wait for all tasks to complete
    // VLOG(1) << "Wait for all tasks to complete";

    // for (size_t i = 0; i < num_gpu_; i++) {
    //     VLOG(1) << "waiting for future " << i;
    //     std::future_status status = futures_[i].wait_for(std::chrono::seconds(timeout_));
    //     if (status == std::future_status::timeout) {
    //         ATER_THROW(
    //             ExecutionTimeout("Profiler timed out after {} sec. Try increasing the timeout.Cancelled
    //             profilers:{}",
    //                              timeout_,
    //                              JoinToString(cmds_)));
    //     }
    //     else {
    //         threads_[i].join();
    //     }
    // }

    // Postprocess profiler results
    postprocessing_delegate_ptr_->PostProcessResults();
}

}  // namespace ater