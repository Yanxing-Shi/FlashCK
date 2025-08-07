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
    std::string GetInstanceName();

    /**
     * @brief Generate the complete kernel code for this configuration
     * @return String containing the generated GPU kernel code
     */
    std::string Emit();

    // ====================== Operation Configuration ======================
    MoeProblem problem_;

    int issues_per_col_;
    int bytes_per_issue_;

    int launch_type_;

    int64_t block_size_;

    int min_block_per_cu_;

    int64_t expert_tile_;

};

}  // namespace flashck