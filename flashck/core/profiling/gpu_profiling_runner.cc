
#include "flashck/core/profiling/gpu_profiling_runner.h"

#include <any>

#include "flashck/core/profiling/profiling_engine.h"

FC_DECLARE_int32(FC_BUILDING_MAX_ATTEMPTS);
FC_DECLARE_int32(FC_BUILDING_TIMEOUT);
FC_DECLARE_int32(FC_TUNING_METRIC);

namespace flashck {

void Postprocesser::AddInstance(InstanceData instance_data)
{
    instances_.emplace_back(std::move(instance_data));
}

struct ProfileResultGroupByKey {
    using KeyTuple = std::tuple<CodeGenKind>;

    KeyTuple operator()(const InstanceData& instance_data)
    {
        return {instance_data.code_gen_kind_};
    }
};

void Postprocesser::PostProcessResults()
{
    auto instance_groups = GroupByFunc(instances_.begin(), instances_.end(), ProfileResultGroupByKey{});

    for (const auto& group : instance_groups) {
        auto metric        = static_cast<Metric>(FLAGS_FC_TUNING_METRIC);
        auto best_instance = *std::max_element(group.begin(), group.end(), [metric](const auto& a, const auto& b) {
            return PerfResult::compare(b.perf_result_, a.perf_result_, metric);
        });

        LOG(INFO) << "According to given metrics: " << MetricToString(metric) << "\n"
                  << "Profiling engine selected the best instance is: " << best_instance.Serialize() << std::endl;

        // for (const auto& instance : group) {
        //     auto& exec_item =
        //         std::any_cast<std::map<std::string, RunningItem>>(instance.op_attrs.at("exec_path"))[exec_cond];
        //     exec_item.instance_name_        = best_instance.instance_name_;
        //     exec_item.perf_result_.split_k_ = best_instance.perf_result_.split_k_;
        // }

        if (best_instance.code_gen_kind_ == CodeGenKind::Norm) {
            try {
                ProfilingEngine::GetInstance()->GetProfilingDB()->Insert(best_instance);
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