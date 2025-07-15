
#include "flashck/core/profiling/gpu_profiling_runner.h"

#include <any>

#include "flashck/core/profiling/profiling_engine.h"

FC_DECLARE_int32(FC_BUILDING_MAX_ATTEMPTS);
FC_DECLARE_int32(FC_BUILDING_TIMEOUT);
FC_DECLARE_int32(FC_TUNING_METRIC);

namespace flashck {

void Postprocesser::AddInstance(InstanceData& instance_data, std::map<std::string, RunningItem>& running_info)
{
    instances_.emplace_back(std::make_tuple(instance_data, std::ref(running_info)));
}

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
    auto instance_groups = GroupByFunc(instances_.begin(), instances_.end(), ProfileResultGroupByKey{});

    for (const auto& group : instance_groups) {
        auto metric        = static_cast<Metric>(FLAGS_FC_TUNING_METRIC);
        auto best_instance = *std::max_element(group.begin(), group.end(), [metric](const auto& a, const auto& b) {
            const auto& instance_data_a = std::get<0>(a);
            const auto& instance_data_b = std::get<0>(b);
            return PerfResult::compare(instance_data_a.perf_result_, instance_data_b.perf_result_, metric);
        });

        auto& best_instance_data = std::get<0>(best_instance);

        LOG(INFO) << "According to given metrics: " << MetricToString(metric) << "\n"
                  << "Profiling engine selected the best instance is: " << best_instance_data.Serialize() << std::endl;

        for (auto& [instance_data, running_info_ref] : group) {
            auto& running_info = running_info_ref.get();
            for (auto& [key, running_item] : running_info) {
                running_item.instance_name_        = best_instance_data.instance_name_;
                running_item.perf_result_.split_k_ = best_instance_data.perf_result_.split_k_;
            }
        }

        if (best_instance_data.code_gen_kind_ == CodeGenKind::Norm) {
            try {
                ProfilingEngine::GetInstance()->GetProfilingDB()->Insert(best_instance_data);
            }
            catch (const std::exception& e) {
                FC_THROW(Unavailable("Cache update failed for {}", e.what()));
            }
        }
    }
}

GPUProfilingRunner::GPUProfilingRunner(const Postprocesser& postprocesser): postprocesser_(postprocesser) {}

void GPUProfilingRunner::Push(const std::vector<std::string>&                  cmds,
                              std::function<void(PerfResult&, Postprocesser&)> process_result_callback)
{
    LOG(INFO) << "running profiler engine with command: " << cmds;
    int attempts = 0;
    try {
        subprocess::Popen popen(cmds, subprocess::output{subprocess::PIPE}, subprocess::error{subprocess::PIPE});

        std::string stdout_str = popen.communicate().first.buf.data();
        std::string stderr_str = popen.communicate().second.buf.data();

        if (stdout_str.empty() && stderr_str.empty()) {
            LOG(ERROR) << "profiling engine failed to run, stdout or stderr is empty";
        }
        else {
            // collect profiler results for postprocessing
            auto [perf_result, is_error] = ExtractProfilingResult(stdout_str);
            if (is_error) {
                LOG(ERROR) << "profiling engine failure!" << "\n"
                           << "profiling engine stdout: " << stdout_str << "\n"
                           << "profiling engine stderr: " << stderr_str;
            }

            process_result_callback(perf_result, postprocesser_);
        }
    }
    catch (std::exception& e) {
        attempts += 1;
        if (attempts >= FLAGS_FC_BUILDING_MAX_ATTEMPTS) {
            FC_THROW(ExecutionTimeout(
                "[{} / {}] Failed to run profiling engine due to exception: {}. Will retry in {} seconds.",
                attempts,
                FLAGS_FC_BUILDING_MAX_ATTEMPTS,
                e.what(),
                FLAGS_FC_BUILDING_TIMEOUT));
        }
        std::this_thread::sleep_for(std::chrono::seconds(FLAGS_FC_BUILDING_TIMEOUT));
    }
}

void GPUProfilingRunner::Join()
{
    postprocesser_.PostProcessResults();
}

}  // namespace flashck