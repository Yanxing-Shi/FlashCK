#pragma once

#include "core/profiling/norm/norm_library.h"
#include "core/profiling/norm/rms_norm/rms_norm_problem.h"

#include "core/utils/common.h"

namespace flashck {

/**
 * @class RmsNormTileDesc
 * @brief Describes the tiling configuration for normalization operations
 *
 * This class defines how the normalization computation is divided across
 * thread blocks and individual threads. It specifies the work distribution
 * and vectorization strategy for optimal GPU performance.
 */
class RmsNormTileDesc {
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
/**
 * @class RmsNormCodeGen
 * @brief Code generator for normalization operations
 *
 * This class encapsulates all the parameters and configuration needed to generate
 * optimized GPU kernels for normalization operations. It combines problem
 * specifications with tiling strategies to produce efficient implementations.
 */
class RmsNormCodeGen {
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

    RmsNormProblem problem_;

    // ====================== tile Configuration ======================
    RmsNormTileDesc tile_desc_;  ///< Tile configuration for this operation

    // ====================== Trait Configuration ======================
    bool is_pad_n_;  ///< Whether to pad the N dimension for alignment
    bool is_two_pass_;

    // ====================== Launch Configuration ======================
    int64_t max_thread_per_block_;  ///< Maximum threads per block for kernel launch
    int64_t min_block_per_cu_ = -1;  ///< Minimum blocks per compute unit for occupancy control

};

}  // namespace flashck