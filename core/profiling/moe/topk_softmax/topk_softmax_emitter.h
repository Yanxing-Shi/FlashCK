#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/profiling/moe/moe_library.h"
#include "core/profiling/moe/topk_softmax/topk_softmax_codegen.h"

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

    std::vector<TopKSoftmaxCodeGen> HeuristicFilter(
                const std::vector<TopKSoftmaxCodeGen>& instances,
                const TopKSoftmaxProblem& topk_softmax_problem);

    std::vector<TopKSoftmaxCodeGen> CreateInstanceForConfig(const TopKSoftmaxConfig& config, const TopKSoftmaxProblem& topk_softmax_problem);

    void GenerateInstances(TopKSoftmaxProblem& topk_softmax_problem);

    /**
     * @brief Gets the total number of generated instances
     * @return Number of generated instances
     */
    int64_t GetNumInstances() const;

    // get profiling instance map for the given norm kind
    std::map<std::string, TopKSoftmaxCodeGen>& GetInstanceMap(TopKSoftmaxProblem topk_softmax_problem)
    {
        GenerateInstances(topk_softmax_problem);
        return instance_map_;
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:

    std::map<std::string, TopKSoftmaxCodeGen> instance_map_;
    int64_t                                                num_instances_ = 0;
};


}  // namespace flashck