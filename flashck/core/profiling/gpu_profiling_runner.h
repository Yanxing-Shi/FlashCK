#pragma once

#include "flashck/core/profiling/gpu_profiling_runner_utils.h"

#include "flashck/core/utils/common.h"

namespace flashck {

class Postprocesser {
public:
    void AddInstance(InstanceData& instance_data, std::map<std::string, RunningItem>& running_info);

    void PostProcessResults();

private:
    std::vector<std::tuple<InstanceData, std::reference_wrapper<std::map<std::string, RunningItem>>>> instances_;
};

class GPUProfilingRunner {
public:
    explicit GPUProfilingRunner(const Postprocesser& postprocesser);

    void Push(const std::vector<std::string>&                  cmds,
              std::function<void(PerfResult&, Postprocesser&)> process_result_callback);

    void Join();

private:
    Postprocesser postprocesser_;
};
}  // namespace flashck