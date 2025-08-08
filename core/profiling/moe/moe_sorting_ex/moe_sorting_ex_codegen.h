#pragma once

#include "core/profiling/moe/moe_library.h"
#include "core/profiling/moe/moe_sorting_ex/moe_sorting_ex_problem.h"

#include "core/utils/common.h"

namespace flashck {

class MoeSortingExCodeGen {
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
    MoeSortingExProblem problem_;

    // ====================== Trait Configuration ======================
    int64_t sub_token_tile_;
    bool sub_token_one_shot_;
    bool local_token_expert_masking_;
    bool local_token_;
    bool skip_expert_with_zero_token_;
    int64_t expert_tile_;

    // ====================== Launch Configuration ======================
    int64_t max_thread_per_block_;
    int64_t min_block_per_cu_;

};

}  // namespace flashck