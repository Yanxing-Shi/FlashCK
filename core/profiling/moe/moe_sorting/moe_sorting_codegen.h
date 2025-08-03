#pragma once

#include "core/profiling/moe/moe_library.h"
#include "core/profiling/moe/moe_problem.h"

#include "core/utils/common.h"

namespace flashck {

class MoeSortingCodeGen {
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

    int64_t internal_load_unroll_;
    int64_t expert_tile_;

    int min_block_per_cu_;

};

}  // namespace flashck