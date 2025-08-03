#pragma once

#include "core/profiling/moe/moe_library.h"
#include "core/profiling/moe/moe_problem.h"

#include "core/utils/common.h"

namespace flashck {
class MoeSmoothQuantTileDesc {
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
    std::string GetInstanceName() const;

    /**
     * @brief Generate the complete kernel code for this configuration
     * @return String containing the generated GPU kernel code
     */
    std::string Emit() const;

    // ====================== Operation Configuration ======================
    MoeProblem problem_;

    MoeSmoothQuantTileDesc tile_desc_;

    bool is_pad_n_;

    bool is_two_pass_;

    int min_block_per_cu_;

};

}  // namespace flashck