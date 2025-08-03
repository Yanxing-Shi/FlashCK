#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/profiling/moe/moe_library.h"
#include "core/profiling/moe/moe_problem.h"
#include "core/profiling/moe/topk_softmax/topk_softmax_codegen.h"
#include "core/utils/json_config.h"

namespace flashck {

/**
 * @class TopKSoftmaxEmitter
 * @brief Manages norm operation code generation and tile descriptor selection
 *
 * This class provides functionality to generate norm operation instances based on
 * different strategies (heuristic, autotuning, or hybrid) and manages tile
 * descriptor validation and filtering.
 */
class TopKSoftmaxEmitter {
public:
    TopKSoftmaxEmitter()  = default;
    ~TopKSoftmaxEmitter() = default;

    TopKSoftmaxEmitter(const TopKSoftmaxEmitter&)            = delete;
    TopKSoftmaxEmitter& operator=(const TopKSoftmaxEmitter&) = delete;

    /**
     * @brief Get singleton instance of TopKSoftmaxEmitter
     * @return Pointer to the singleton instance
     */
    static TopKSoftmaxEmitter* GetInstance()
    {
        static TopKSoftmaxEmitter instance;
        return &instance;
    }

    bool IsValidInstance(const TopKSoftmaxCodeGen& instance);

    std::vector<TopKSoftmaxCodeGen> CreateInstanceForConfig(const flashck::TopKSoftmaxConfig& config, const MoeProblem& moe_problem);

    void GenerateInstances(TopKSoftmaxProblem& gemm_problem);

    /**
     * @brief Gets the total number of generated instances
     * @return Number of generated instances
     */
    int64_t GetNumInstances() const;

    // get profiling instance map for the given norm kind
    std::map<std::string, TopKSoftmaxCodeGen>& GetInstanceMap(TopKSoftmaxProblem gemm_problem)
    {
        GenerateInstances(gemm_problem);
        return instance_map_[gemm_problem.kind_];
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:

    std::map<TopKSoftmaxKind, std::map<std::string, TopKSoftmaxCodeGen>> instance_map_;
    int64_t                                                num_instances_ = 0;
};


}  // namespace flashck