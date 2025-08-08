#pragma once

#include "core/profiling/moe/moe_library.h"
#include "core/profiling/moe/moe_smooth_quant/moe_smooth_quant_problem.h"

#include "core/utils/common.h"

namespace flashck {
class MoeSmoothQuantTileDesc {
public:
    /**
     * @brief Generate a unique name for this tile configuration
     * @return String identifier based on tile parameters
     */
    std::string GetInstanceName();

    /**
     * @brief Generate code template parameters for this tile
     * @return String representation for code generation
     */
    std::string Emit();

    // ====================== Tile Configuration Parameters ======================

    int64_t m_repeat_;            ///< Number of M-dimension repeats per thread
    int64_t n_repeat_;            ///< Number of N-dimension repeats per thread
    int64_t m_thread_per_block_;  ///< Number of threads along M dimension in a block
    int64_t n_thread_per_block_;  ///< Number of threads along N dimension in a block
    int64_t n_vector_;            ///< Vector size along N dimension for memory coalescing
};

class MoeSmoothQuantCodeGen {
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
    MoeSmoothQuantProblem problem_;

    // ====================== Tile Configuration ======================
    MoeSmoothQuantTileDesc tile_desc_;

    // ====================== Trait Configuration ======================
    bool is_pad_n_;

    // ====================== Strategy Configuration ======================
    bool is_two_pass_;

    // ====================== Launch Configuration ======================
    int64_t max_thread_per_block_;
    int64_t min_block_per_cu_;

};

}  // namespace flashck