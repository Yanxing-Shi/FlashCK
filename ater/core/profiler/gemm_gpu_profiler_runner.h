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

#include "ater/core/profiler/profiler_runner_utils.h"
#include "ater/core/profiler/target.h"
#include "ater/core/utils/named_tuple_utils.h"
#include "ater/core/utils/subprocess_utils.h"

namespace ater {

/*
Object which collects profiler results after profiler executables complete,
updates profiler results cache and the gemm nodes' attrs after all profilers complete.
*/
class GemmProfilerPostprocessingDelegate {
public:
    // Initialize storage for profiler results
    // Instance=(
    //     ProfileResult=(best_algo, elapsed_runtime, workspace),
    //     func_attrs,
    //     profiler_filename,
    //     exec_key,
    //     split_k,
    // )
    GemmProfilerPostprocessingDelegate() = default;

    // As a profiler executable completes, collect the result
    void AddInstance(const std::vector<ProfileResult>&                result,
                     const std::unordered_map<std::string, std::any>& op_attrs_map,
                     const std::string&                               kernel_name,
                     const std::string&                               exec_key,
                     const int                                        split_k);
    /*
    When all profiler executables complete, find the best instance
    (min runtime per op name, profiler executable and exec_key (i.e. gemm shape mnk)
    across multiple split_k values)
    The best instance is cached, and written into corresponding gemm nodes in the graph
    */
    void PostProcessResults();

private:
    std::vector<
        std::
            tuple<std::vector<ProfileResult>, std::unordered_map<std::string, std::any>, std::string, std::string, int>>
        instances_;
};

/*
Another parallel runner to execute profilers on multiple GPUs in parallel
It uses a process pool for implementation, avoiding process creation overhead
The size of the process pool is equal to the number of provided GPUs,
so ~ideally~ each process should execute a profiler on its dedicated GPU.
This property hasn't been properly verified yet, however, the results are
empirically better compared to the previous runner.
*/
class GPUProfilerRunner {
public:
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
    explicit GPUProfilerRunner(const int                                                  num_gpu,
                               const std::shared_ptr<GemmProfilerPostprocessingDelegate>& postprocessing_delegate_ptr,
                               const int                                                  timeout = 300);

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
    void Push(
        const std::vector<std::string>&                                                        cmds,
        const std::function<void(const std::vector<ProfileResult>&,
                                 const std::shared_ptr<GemmProfilerPostprocessingDelegate>&)>& process_result_callback);
    /*
    Wait for subprocesses completion or timeout;
    postprocess the profiler results with delegate(s) */
    void Join();

private:
    int num_gpu_;
    int timeout_;
    // This queue is used to ensure only one task is executed on a device at a time
    std::queue<std::string> device_queue_;

    std::vector<std::future<void>> futures_;
    std::vector<std::thread>       threads_;

    std::shared_ptr<GemmProfilerPostprocessingDelegate> postprocessing_delegate_ptr_;

    std::vector<std::string> cmds_;
};
}  // namespace ater