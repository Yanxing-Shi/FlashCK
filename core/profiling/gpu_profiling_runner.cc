
#include "core/profiling/gpu_profiling_runner.h"

#include <any>

#include "core/profiling/profiling_engine.h"

// Flag declarations for profiling configuration
FC_DECLARE_int32(FC_BUILDING_MAX_ATTEMPTS);  ///< Maximum retry attempts for profiling execution
FC_DECLARE_int32(FC_BUILDING_TIMEOUT);       ///< Timeout between retry attempts (seconds)
FC_DECLARE_int32(FC_TUNING_METRIC);          ///< Metric used for selecting best instances

namespace flashck {

void Postprocesser::AddInstance(InstanceData& instance_data, std::map<std::string, RunningItem>& running_info)
{
    // Store instance data with reference to running information for later processing
    instances_.emplace_back(std::make_tuple(instance_data, std::ref(running_info)));

    VLOG(2) << "Added instance for post-processing: " << instance_data.instance_name_
            << " (code_gen_kind: " << static_cast<int>(instance_data.code_gen_kind_) << ")";
}

/**
 * @struct ProfileResultGroupByKey
 * @brief Functor for grouping profiling results by code generation kind
 *
 * Groups profiling instances by their code generation type to enable
 * separate optimization for different kernel categories (e.g., GEMM, Norm).
 */
struct ProfileResultGroupByKey {
    using KeyTuple = std::tuple<CodeGenKind>;

    KeyTuple operator()(const std::tuple<InstanceData, std::reference_wrapper<std::map<std::string, RunningItem>>>&
                            profiling_instance_tuple)
    {
        const auto& instance_data = std::get<0>(profiling_instance_tuple);
        return {instance_data.code_gen_kind_};
    }
};

void Postprocesser::PostProcessResults()
{
    VLOG(1) << "Starting post-processing of " << instances_.size() << " profiled instances";

    // Group instances by code generation kind for separate optimization
    auto instance_groups = GroupByFunc(instances_.begin(), instances_.end(), ProfileResultGroupByKey{});

    VLOG(1) << "Grouped instances into " << std::distance(instance_groups.begin(), instance_groups.end())
            << " code generation categories";

    for (const auto& group : instance_groups) {
        // Get the configured comparison metric
        auto metric = static_cast<Metric>(FLAGS_FC_TUNING_METRIC);

        // Find the best performing instance in this group
        // PerfResult::compare(a, b, metric) returns true if 'a' is better than 'b'
        // We want to find the element that is better than all others
        auto best_instance = *std::max_element(group.begin(), group.end(), [metric](const auto& a, const auto& b) {
            const auto& instance_data_a = std::get<0>(a);
            const auto& instance_data_b = std::get<0>(b);
            return PerfResult::compare(instance_data_b.perf_result_, instance_data_a.perf_result_, metric);
        });

        auto& best_instance_data = std::get<0>(best_instance);

        VLOG(1) << "According to given metrics: " << MetricToString(metric) << "\n"
                << "Profiling engine selected the best instance is: " << best_instance_data.Serialize() << std::endl;

        // Update all running items in this group with the best instance information
        for (auto& [instance_data, running_info_ref] : group) {
            auto& running_info = running_info_ref.get();
            for (auto& [key, running_item] : running_info) {
                running_item.instance_name_        = best_instance_data.instance_name_;
                running_item.perf_result_.split_k_ = best_instance_data.perf_result_.split_k_;
            }

            VLOG(2) << "Updated " << running_info.size()
                    << " running items with best instance: " << best_instance_data.instance_name_;
        }

        // Cache optimal instances for normalization operations
        if (best_instance_data.code_gen_kind_ == CodeGenKind::Norm) {
            try {
                ProfilingEngine::GetInstance()->GetProfilingDB()->Insert(best_instance_data);
                VLOG(1) << "Cached optimal normalization instance: " << best_instance_data.instance_name_;
            }
            catch (const std::exception& e) {
                FC_THROW(Unavailable("Cache update failed for {}", e.what()));
            }
        }
    }

    VLOG(1) << "Post-processing completed successfully";
}

GPUProfilingRunner::GPUProfilingRunner(const Postprocesser& postprocesser): postprocesser_(postprocesser)
{
    VLOG(2) << "Created GPUProfilingRunner with post-processor";
}

void GPUProfilingRunner::Push(const std::vector<std::string>&                  cmds,
                              std::function<void(PerfResult&, Postprocesser&)> process_result_callback)
{
    VLOG(1) << "Executing profiling command: " << JoinStrings(cmds, " ");

    int attempts = 0;

    while (attempts < FLAGS_FC_BUILDING_MAX_ATTEMPTS) {
        try {
            // Execute profiling command using subprocess
            subprocess::Popen popen(cmds, subprocess::output{subprocess::PIPE}, subprocess::error{subprocess::PIPE});

            // Capture stdout and stderr from profiling execution
            std::string stdout_str = popen.communicate().first.buf.data();
            std::string stderr_str = popen.communicate().second.buf.data();

            VLOG(2) << "Profiling execution completed with stdout size: " << stdout_str.size()
                    << ", stderr size: " << stderr_str.size();

            if (stdout_str.empty() && stderr_str.empty()) {
                LOG(ERROR) << "Profiling engine failed to run, stdout and stderr are both empty";
                return;
            }

            // Extract and validate profiling results
            auto [perf_result, is_error] = ExtractProfilingResult(stdout_str);
            if (is_error) {
                LOG(ERROR) << "Profiling engine failure!" << "\n"
                           << "stdout: " << stdout_str << "\n"
                           << "stderr: " << stderr_str;
                return;
            }

            VLOG(1) << "Successfully extracted profiling results - latency: " << perf_result.latency_
                    << "ms, tflops: " << perf_result.tflops_ << ", bandwidth: " << perf_result.bandwidth_ << "GB/s";

            // Process results through callback
            process_result_callback(perf_result, postprocesser_);
            return;  // Success, exit retry loop
        }
        catch (const std::exception& e) {
            attempts++;
            LOG(WARNING) << "[" << attempts << "/" << FLAGS_FC_BUILDING_MAX_ATTEMPTS
                         << "] Profiling execution failed: " << e.what();

            if (attempts >= FLAGS_FC_BUILDING_MAX_ATTEMPTS) {
                FC_THROW(ExecutionTimeout(
                    "[{} / {}] Failed to run profiling engine due to exception: {}. Maximum attempts reached.",
                    attempts,
                    FLAGS_FC_BUILDING_MAX_ATTEMPTS,
                    e.what()));
            }

            VLOG(1) << "Retrying in " << FLAGS_FC_BUILDING_TIMEOUT << " seconds...";
            std::this_thread::sleep_for(std::chrono::seconds(FLAGS_FC_BUILDING_TIMEOUT));
        }
    }
}

void GPUProfilingRunner::Join()
{
    VLOG(1) << "Joining profiling runner and starting post-processing";
    postprocesser_.PostProcessResults();
}

}  // namespace flashck