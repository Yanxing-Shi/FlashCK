#pragma once

#include "core/profiling/moe/moe_library.h"
#include "core/profiling/moe/moe_problem.h"

#include "core/utils/common.h"

namespace flashck {

class TopKSoftmaxCodeGen {
public:
    /**
     * @brief Generate a unique instance name for this configuration
     * @return String identifier combining operation type and parameters
     */
    std::string GetInstanceName() const;

    /**
     * @brief Generate the complete kernel code for this configuration
     * @return String containing the generated GPU kernel code
     */
    std::string Emit() const;

    // ====================== Operation Configuration ======================
    TopKSoftmaxProblem problem_;

    int64_t num_experts_;

    int issues_pre_col_;
    int bytes_per_issue_;

    int launch_type_;

    int64_t block_size_;

    int min_block_pre_cu_;

};

}  // namespace flashck