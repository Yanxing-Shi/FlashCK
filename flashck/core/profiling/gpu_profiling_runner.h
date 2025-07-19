#pragma once

#include "flashck/core/profiling/gpu_profiling_runner_utils.h"
#include "flashck/core/utils/common.h"

namespace flashck {

/**
 * @class Postprocesser
 * @brief Processes profiling results and selects optimal kernel instances
 *
 * The Postprocesser collects profiling results from multiple kernel instances,
 * groups them by code generation kind, and selects the best performing instance
 * based on the configured metric (latency, throughput, etc.). It also handles
 * caching of optimal instances for future use.
 */
class Postprocesser {
public:
    /**
     * @brief Add a profiled instance for post-processing
     * @param instance_data Profiling data for the kernel instance
     * @param running_info Reference to the running information map
     *
     * Stores the instance data and associated running information for later
     * processing. The running information will be updated with the best
     * instance details during post-processing.
     */
    void AddInstance(InstanceData& instance_data, std::map<std::string, RunningItem>& running_info);

    /**
     * @brief Process all collected instances and select optimal configurations
     *
     * Groups instances by code generation kind, compares performance metrics,
     * selects the best instance for each group, and updates running information.
     * Also caches optimal instances for normalization operations.
     */
    void PostProcessResults();

private:
    /// Collection of profiled instances with their associated running information
    std::vector<std::tuple<InstanceData, std::reference_wrapper<std::map<std::string, RunningItem>>>> instances_;
};

/**
 * @class GPUProfilingRunner
 * @brief Executes GPU kernel profiling commands and manages result processing
 *
 * The GPUProfilingRunner provides an interface for executing profiling commands
 * asynchronously, collecting performance results, and coordinating with the
 * post-processor to analyze and select optimal kernel configurations.
 */
class GPUProfilingRunner {
public:
    /**
     * @brief Construct runner with a post-processor
     * @param postprocesser Post-processor instance for result analysis
     */
    explicit GPUProfilingRunner(const Postprocesser& postprocesser);

    /**
     * @brief Execute profiling command and process results
     * @param cmds Command vector to execute (profiling binary and arguments)
     * @param process_result_callback Callback to process performance results
     *
     * Executes the profiling command using subprocess, captures stdout/stderr,
     * extracts performance metrics, and invokes the callback for result processing.
     * Includes retry logic for handling transient execution failures.
     */
    void Push(const std::vector<std::string>&                  cmds,
              std::function<void(PerfResult&, Postprocesser&)> process_result_callback);

    /**
     * @brief Wait for all profiling tasks and process final results
     *
     * Triggers post-processing of all collected profiling instances to select
     * optimal configurations and update caches. Should be called after all
     * profiling tasks have been pushed.
     */
    void Join();

private:
    /// Post-processor instance for analyzing profiling results
    Postprocesser postprocesser_;
};

}  // namespace flashck