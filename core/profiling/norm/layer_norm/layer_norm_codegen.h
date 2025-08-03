#pragma once

#include "core/profiling/norm/norm_library.h"
#include "core/profiling/norm/norm_problem.h"

#include "core/utils/common.h"

namespace flashck {

/**
 * @class LayerNormTileDesc
 * @brief Describes the tiling configuration for normalization operations
 *
 * This class defines how the normalization computation is divided across
 * thread blocks and individual threads. It specifies the work distribution
 * and vectorization strategy for optimal GPU performance.
 */
class LayerNormTileDesc {
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

    int64_t m_repeat;            ///< Number of M-dimension repeats per thread
    int64_t n_repeat;            ///< Number of N-dimension repeats per thread
    int64_t m_thread_per_block;  ///< Number of threads along M dimension in a block
    int64_t n_thread_per_block;  ///< Number of threads along N dimension in a block
    int64_t n_vector;            ///< Vector size along N dimension for memory coalescing
};

/**
 * @class NormCodeGen
 * @brief Code generator for normalization operations
 *
 * This class encapsulates all the parameters and configuration needed to generate
 * optimized GPU kernels for normalization operations. It combines problem
 * specifications with tiling strategies to produce efficient implementations.
 */
class NormCodeGen {
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

    NormProblem problem_;

    // Tiling strategy
    LayerNormTileDesc tile_desc_;  ///< Tile configuration for this operation

    bool is_two_pass_;

    bool is_pad_n_;

};

}  // namespace flashck