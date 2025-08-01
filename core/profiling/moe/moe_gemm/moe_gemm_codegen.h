#pragma once

#include "core/profiling/gemm/gemm_library.h"
#include "core/profiling/gemm/gemm_problem.h"

#include "core/utils/common.h"

namespace flashck {

/**
 * @class GemmTileDesc
 * @brief Describes the tiling configuration for GEMM operations
 *
 */
class MoeGemmTileDesc {
public:
    /**
     * @brief Generate a unique name for this tile configuration
     * @return String identifier based on tile parameters
     */
    std::string GetInstanceName() const;

    /**
     * @brief Generate code template parameters for this tile
     * @return String representation for code generation
     */
    std::string Emit() const;

    // ====================== Tile Configuration Parameters ======================

    int64_t m_block_;
    int64_t n_block_;
    int64_t k_block_;

    int64_t m_warp_;
    int64_t n_warp_;
    int64_t k_warp_;

    int64_t m_warp_tile_;
    int64_t n_warp_tile_;
    int64_t k_warp_tile_;

};

class MoeGemmCodeGen {
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