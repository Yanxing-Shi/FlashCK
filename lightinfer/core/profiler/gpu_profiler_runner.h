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

#include "lightinfer/core/profiler/gpu_profiler_runner_utils.h"
#include "lightinfer/core/profiler/target.h"
#include "lightinfer/core/utils/named_tuple_utils.h"
#include "lightinfer/core/utils/subprocess_utils.h"

namespace lightinfer {

class ProfilerPostprocess {
public:
    ProfilerPostprocess() = default;

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

class GPUProfilerRunner {
public:
    explicit GPUProfilerRunner(const int                                   num_gpu,
                               const std::shared_ptr<ProfilerPostprocess>& postprocessing_delegate_ptr,
                               const int                                   timeout = 300);

    void Push(const std::vector<std::string>& cmds,
              const std::function<void(const std::vector<ProfileResult>&, const std::shared_ptr<ProfilerPostprocess>&)>&
                  process_result_callback);

    void Join();

private:
    int num_gpu_;
    int timeout_;

    std::shared_ptr<ProfilerPostprocess> postprocessing_ptr_;

    std::vector<std::string> cmds_;
};
}  // namespace lightinfer