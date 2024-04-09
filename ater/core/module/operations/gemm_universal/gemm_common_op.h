#pragma once

#include <any>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

#include "ater/core/utils/dtype.h"
#include "ater/core/utils/layout.h"

#include "ater/core/profiler/base.h"
#include "ater/core/profiler/gemm_gpu_profiler_runner.h"

#include "ater/core/graph/node.h"

namespace ater {

/*v
Base gemm operators
*/
template<typename T>
class GemmCommonOp: public Operation {
public:
    GemmCommonOp() = default;
    GemmCommonOp(std::string op_name): Operation(op_name) {}
    virtual ~GemmCommonOp() {}

    virtual Shape InferShape(Variable* A, Variable* B)
    {
        ATER_THROW(Unimplemented("{}", "Not implemented"));
    }

    void Forward() override;
    /*
    Extracts a mapping between dim names and a list of DimInfo.
        This function will be used in gemm shape inference, gemm padding graph
        transformation, gemm profiling, etc.

        All subclasses must implement this API.

        An example result from gemm_rcr:
        [M,K] * [N, K] = [M,N]
        {
            "M": [
                DimInfo(source=INPUT, tensor_idx=0, dim_idx=0),
                DimInfo(source=OUTPUT, tensor_idx=0, dim_idx=0),
            ],
            "K": [
                DimInfo(source=INPUT, tensor_idx=0, dim_idx=1),
                DimInfo(source=INPUT, tensor_idx=1, dim_idx=1),
            ],
            "N": [
                DimInfo(source=INPUT, tensor_idx=1, dim_idx=0),
                DimInfo(source=OUTPUT, tensor_idx=0, dim_idx=1),
            ],
        }

        Parameters
        ----------
        for_profiling: bool
            Whether this function is used for generating profiling source codes.
            If yes, some DimInfo are simplified. e.g. For gemm, we treat all tensors
            as 2d.
    */

    virtual std::map<std::string, std::vector<std::shared_ptr<DimInfo>>> ExtractDims(bool for_profiling = false)
    {
        ATER_THROW(Unimplemented("{}", "Not implemented"));
    }

    /*

    */
    std::string GenExecKey(const std::map<std::string, std::vector<int>>& name_value_mapping);

    /*
    Extracts profiling keys and execution conditions for a given dynamic_profiling_strategy.
        This function fills in self._attrs["exec_path"].
        Keys are "exec_key"s, and are used for profiling.
        Values are ItemValues, where "profiling_key" fields are the same as the corresponding keys,
        "exec_cond" fields specify dynamic ranges, and "algo" fields are empty for now.

        e.g. for gemm_rrr, input1=[m, k], input2=[k, n]
        m = 1, k = 128, n = 256.
        self._attrs["exec_path"] = {
            "M==1 && K==128 && N==256" : ItemValue(
                profiling_key="M==1 && K==128 && N==256",
                exec_cond="M==1 && K==128 && N==256",
                algo="",
            )
        }

        e.g. for gemm_rrr, input1=[dynamic_m, k], input2=[k, n]
        dynamic_m >= 1 and dynamic_m <= 1024, dynamic_profiling_strategy = MAX,
        k = 128, n = 256.
        self._attrs["exec_path"] = {
            "M==1024 && K==128 && N==256" : ItemValue(
                profiling_key="M==1024 && K==128 && N==256",
                exec_cond="M>=1 && M<=1024 && K==128 && N==256",
                algo="",
            )
        }

        Parameters
        ----------
        dynamic_profiling_strategy : DynamicProfileStrategy
            See comments for DynamicProfileStrategy.

    */

    void ExtractExecPath(const DynamicProfileStrategy& dynamic_profiling_strategy);

    // check if profile cache exits
    // If we have a cached
    // entry for this gemm instance, we update this gemm op's
    // relevant attributes with the cached result and return False.
    bool IfShouldBuildProfiler(const std::unordered_set<std::string>& workloads);

    // Generate profilers for this gemm op
    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenOpProfiler(const DynamicProfileStrategy& dynamic_profiling_strategy = DynamicProfileStrategy::MAX) override;

    std::vector<std::string>
    GenOpProfileCmd(const std::string&                                                 profiler_prefix,
                    const std::string&                                                 profiler_filename,
                    const std::string&                                                 exec_key,
                    const std::function<std::vector<std::string>(const std::string&)>& fbuild_cmd = nullptr);

    virtual std::vector<std::string>
    GenProfileCmd(const std::string& profiler_prefix, const std::string& profiler_filename, const std::string& exec_key)
    {
        ATER_THROW(Unimplemented("{}", "Not implemented"));
    }

    int GetABAlignment(const std::string& exec_key);

    void ExtractEpilogueAlignment(const std::vector<int>&       ouput_shape,
                                  const DynamicProfileStrategy& dynamic_profiling_strategy);

    std::vector<int> SplitKSearchSpace(const int m, const int n, const int k);

    void ProfileSingleWorkload(const std::string&                        profiler_prefix,
                               const std::string&                        exec_key,
                               const std::shared_ptr<GPUProfilerRunner>& profiler_runner_ptr,
                               bool                                      force_cache);

    // select the best kernel configurations for this gemm op
    /*
    Parameters
        ----------
        profiler_runner: ProfilerRunner
            Profiler runner to schedule async profiler jobs,
        workdir : str
            Base dir to keep profiling source codes, by default "./"running on separate GPU devices concurrently
    */
    void Profile(const std::shared_ptr<GPUProfilerRunner>& profiler_runner_ptr,
                 const std::string&                        folder_name = "kernel_profile") override;

    virtual void AlignAB(Variable* a, Variable* b)
    {
        ATER_THROW(Unimplemented("{}", "Not implemented"));
    }

    void SanityCheck(Variable* a, Variable* b);

    // call this op
    Variable* operator()(Variable* a, Variable* b);

    std::string GenOpFunction() override;

    std::unordered_map<std::string, std::any> GetAttrsMap();

    std::string     op_name_ = "gemm";
    DataLayout      layout_;
    TensorOperation epilogue_op_;

    bool                              has_profiler_       = true;
    std::function<int(int, int, int)> ab_alignment_func_  = nullptr;
    int                               epilogue_alignment_ = 1;
    std::string                       epilogue_;

    int         workspace_     = 0;
    int         split_k_       = 1;
    int         split_k_hints_ = 0;
    size_t      num_sources_   = 0;
    float       alpha_         = 1.0;
    std::string permute_shape_ = "";

    std::map<std::string, std::shared_ptr<ExecItem>> exec_path_;
    std::unordered_set<std::string>                  all_workloads_;
    std::map<std::string, std::vector<int>>          current_dim_map_;

    std::map<std::string, std::shared_ptr<void>> kernel_instance_map_;

    std::vector<Variable*> input_var_;
    std::vector<Variable*> output_var_;
};

}  // namespace ater