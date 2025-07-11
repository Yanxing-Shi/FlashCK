#pragma once

#include "flashck/core/profiling/gpu_profiling_runner_utils.h"

#include "flashck/core/utils/common.h"

namespace flashck {

class Postprocesser {
public:
    void AddInstance(InstanceData instance_data);

    void PostProcessResults();

private:
    std::vector<InstanceData> instances_;
};

class GPUProfilerRunner {
public:
    explicit GPUProfilerRunner(const Postprocesser& postprocesser);

    void Push(const std::vector<std::string>&                                           cmds,
              std::function<void(const std::vector<PerfResult>&, const Postprocesser&)> process_result_callback);

    void Join();

private:
    Postprocesser postprocesser_;
};
}  // namespace flashck