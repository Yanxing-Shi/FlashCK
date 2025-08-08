#pragma once

#include "core/profiling/moe/moe_library.h"
#include "core/profiling/moe/topk_softmax/topk_softmax_problem.h"

#include "core/utils/common.h"

namespace flashck {

class TopKSoftmaxCodeGen {
public:
    /**
     * @brief Generate a unique instance name for this configuration
     * @return String identifier combining operation type and parameters
     */
    std::string GetInstanceName();

    /**
     * @brief Generate the complete kernel code for this configuration
     * @return String containing the generated GPU kernel code
     */
    std::string Emit();

    // ====================== Operation Configuration ======================
    TopKSoftmaxProblem problem_;

    // ====================== Trait Configuration ======================
    int64_t issues_per_col_;
    int64_t bytes_per_issue_;
    int64_t launch_type_;
    int64_t block_size_;
    int64_t expert_tile_;

    // ====================== Launch Configuration ======================
    int64_t max_thread_per_block_;
    int64_t min_block_per_cu_;

};

}  // namespace flashck