#pragma once

#include <any>
#include <future>
#include <map>
#include <memory>
#include <queue>
#include <regex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "flashck/core/profiler/gpu_profiler_runner_utils.h"
#include "flashck/core/profiler/target.h"
#include "flashck/core/utils/named_tuple_utils.h"
#include "flashck/core/utils/subprocess_utils.h"

namespace flashck {

class ProfilerPostprocess {
public:
    void AddInstance(const std::vector<ProfileResult>&                result,
                     const GenOperationKind&                          op_kind,
                     const std::unordered_map<std::string, std::any>& op_attrs_map,
                     const std::string&                               kernel_name,
                     const std::string&                               exec_key,
                     const int64_t                                    split_k = -1);

    void PostProcessResults();

private:
    std::vector<std::tuple<std::vector<ProfileResult>,
                           GenOperationKind,
                           std::unordered_map<std::string, std::any>,
                           std::string,
                           std::string,
                           int64_t>>
        instances_;
};

/// @file gpu_profiler_runner.h
/// @brief GPU profiling executor with retry mechanism
class GPUProfilerRunner {
public:
    explicit GPUProfilerRunner(const int                                   num_gpu,
                               const std::shared_ptr<ProfilerPostprocess>& postprocessing_delegate_ptr,
                               const int                                   timeout = 300);

    void Push(const std::vector<std::string>& cmds, std::function<void(const auto&, const auto&)> result_callback);

    void Join();

private:
    int num_gpu_;
    int timeout_;

    std::shared_ptr<ProfilerPostprocess> postprocessing_ptr_;

    static constexpr int  MAX_ATTEMPTS = 3;
    static constexpr auto RETRY_DELAY  = std::chrono::seconds(5);
};
}  // namespace flashck