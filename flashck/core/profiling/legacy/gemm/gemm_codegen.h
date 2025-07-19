#pragma once

#include <string>
#include <vector>

#include "flashck/core/profiling/legacy/gemm/gemm_library.h"
#include "flashck/core/utils/dtype.h"

namespace flashck {

/**
 * @class GemmTileDesc
 * @brief Describes the tiling configuration for GEMM operations
 *
 * This class defines how the GEMM computation is divided across thread blocks
 * and XDL (eXtended Data Layout) units. It specifies the work distribution
 * strategy for optimal GPU performance on AMD GPUs.
 */
class GemmTileDesc {
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

    int64_t block_size_;  ///< The number of threads within workgroup. Defines the total thread count per block for GEMM
                          ///< computation.

    // Block-level tiling dimensions
    int64_t m_per_block_;  ///< The input/output data tile size in the M dimension. Determines how much work each block
                           ///< processes in M direction.
    int64_t n_per_block_;  ///< The input/output data tile size in the N dimension. Determines how much work each block
                           ///< processes in N direction.
    int64_t k_per_block_;  ///< The input/output data tile size in the K dimension. Determines how much work each block
                           ///< processes in K direction.

    int64_t a_k1_;  ///< The vector load size from global memory for A tensor. Determines vectorization granularity for
                    ///< matrix A reads.
    int64_t b_k1_;  ///< The vector load size from global memory for B tensor. Determines vectorization granularity for
                    ///< matrix B reads.

    int64_t m_per_xdl_;  ///< M size of matrix-fused-multiply-add instruction. Defines the M dimension processed per MMA
                         ///< operation.
    int64_t n_per_xdl_;  ///< N size of matrix-fused-multiply-add instruction. Defines the N dimension processed per MMA
                         ///< operation.

    // Wave-level XDL distribution
    int64_t m_xdl_per_wave_;  ///< The number of iterations in the M dimension over output tile per wavefront. Controls
                              ///< MMA instruction distribution per wave.
    int64_t n_xdl_per_wave_;  ///< The number of iterations in the N dimension over output tile per wavefront. Controls
                              ///< MMA instruction distribution per wave.
};

/**
 * @class BlockTransferDesc
 * @brief Describes data transfer configuration for GEMM block operations
 *
 * This class specifies how data is transferred from global memory to
 * local data store (LDS) and how threads are organized for efficient
 * memory access patterns.
 */
class BlockTransferDesc {
public:
    /**
     * @brief Default constructor
     */
    BlockTransferDesc() = default;

    /**
     * @brief Constructor for aggregate initialization
     * @param thread_cluster_length Thread cluster lengths
     * @param thread_cluster_arrange_order Thread cluster arrangement order
     * @param src_access_order Source access order
     * @param src_vector_dim Source vector dimension
     * @param src_scalar_per_vector Source scalars per vector
     * @param dst_scalar_per_vector Destination scalars per vector
     * @param add_extra_dim Add extra dimension flag
     */
    BlockTransferDesc(const std::vector<int64_t>& thread_cluster_length,
                      const std::vector<int64_t>& thread_cluster_arrange_order,
                      const std::vector<int64_t>& src_access_order,
                      int64_t                     src_vector_dim,
                      int64_t                     src_scalar_per_vector,
                      int64_t                     dst_scalar_per_vector,
                      int64_t                     add_extra_dim);

    /**
     * @brief Generate a unique name for this transfer configuration
     * @return String identifier based on transfer parameters
     */
    std::string GetInstanceName() const;

    /**
     * @brief Generate code template parameters for this transfer
     * @return String representation for code generation
     */
    std::string Emit() const;

    // ====================== Thread Cluster Configuration ======================

    std::vector<int64_t> thread_cluster_length_;  ///< Spatial thread distribution over the input data. Can be
                                                  ///< interpreted as the answer to the question: "How many threads to
                                                  ///< arrange on each input data axis?"
    std::vector<std::string>
        thread_cluster_length_vec_;  ///< String representation of thread cluster lengths for template generation.

    std::vector<int64_t>
        thread_cluster_arrange_order_;  ///< The order of thread spatial distribution over the input
                                        ///< tensor dimension. Can be interpreted as the answer to the
                                        ///< question: "In which order to spread threads through tensor axes?".
    std::vector<std::string> thread_cluster_arrange_order_vec_;  ///< String representation of thread arrangement order
                                                                 ///< for template generation.

    std::vector<int64_t>
        src_access_order_;  ///< The order of accessing input tensor axes. Can be interpreted as the
                            ///< answer to the question: "Which dimension to read first? And which next?" etc.
    std::vector<std::string>
        src_access_order_vec_;  ///< String representation of source access order for template generation.

    // ====================== Vectorization Configuration ======================

    int64_t src_vector_dim_;         ///< The index of axis on which we could do vectorized memory access - the one with
                                     ///< contiguous memory.
    int64_t src_scalar_per_vector_;  ///< The size of vector access instruction - the number of elements accessed per
                                     ///< thread per instruction.
    int64_t dst_scalar_per_vector_;  ///< The size of vectorized store into LDS memory.

    // ====================== Additional Dimension Handling ======================
    int64_t add_extra_dim_;  ///< Whether to use padding for LDS or not. With universal GEMM there's no need for
                             ///< padding. Controls LDS memory layout optimization.
};

/**
 * @class CBlockTransferDesc
 * @brief Describes output matrix C block transfer configuration
 *
 * This class specifies how the computed results are transferred from
 * registers to global memory for the output matrix C.
 */
class CBlockTransferDesc {
public:
    /**
     * @brief Default constructor
     */
    CBlockTransferDesc() = default;

    /**
     * @brief Construct with full configuration
     * @param m_xdl_per_wave XDL units along M per wavefront
     * @param n_xdl_per_wave XDL units along N per wavefront
     * @param m_n_block_wave_per_xdl Block/wave distribution per XDL
     * @param scalar_per_vector Number of scalars per vector for output
     */
    CBlockTransferDesc(int64_t              m_xdl_per_wave,
                       int64_t              n_xdl_per_wave,
                       std::vector<int64_t> m_n_block_wave_per_xdl,
                       int64_t              scalar_per_vector);

    /**
     * @brief Generate a unique name for this C-block transfer configuration
     * @return String identifier based on transfer parameters
     */
    std::string GetInstanceName() const;

    /**
     * @brief Generate code template parameters for this C-block transfer
     * @return String representation for code generation
     */
    std::string Emit() const;

    // ====================== C-Block Transfer Configuration ======================

    int64_t m_xdl_per_wave_;  ///< The number of matrix-multiplication instructions results to process per wave per
                              ///< iteration of CShuffle in M dimension.
    int64_t n_xdl_per_wave_;  ///< The number of matrix-multiplication instructions results to process per wave per
                              ///< iteration of CShuffle in N dimension.
    std::vector<int64_t> m_n_block_wave_per_xdl_;  ///< The spatial thread distribution used for storing data into
                                                   ///< output tensor across output data layout dimensions.
    std::vector<std::string>
            m_n_block_wave_per_xdl_vec_;  ///< String representation of block wave distribution for template generation.
    int64_t scalar_per_vector_;  ///< The size of vectorized memory access. Used when storing data to output tensor.
                                 ///< Controls write vectorization granularity.
};

/**
 * @class GemmCodegen
 * @brief Code generator for GEMM operations
 *
 * This class encapsulates all the parameters and configuration needed to generate
 * optimized GPU kernels for GEMM operations. It combines problem specifications
 * with tiling strategies and data transfer configurations to produce efficient
 * implementations.
 */
class GemmCodegen {
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

    // ====================== Layout Configuration ======================

    LayoutType
        a_layout_;  ///< Matrix A memory layout (row/column major). Determines how matrix A data is arranged in memory.
    LayoutType
        b_layout_;  ///< Matrix B memory layout (row/column major). Determines how matrix B data is arranged in memory.
    LayoutType
        c_layout_;  ///< Matrix C memory layout (row/column major). Determines how matrix C data is arranged in memory.
    std::vector<LayoutType> ds_layout_;  ///< Data store (DS) memory layouts. Specifies how multiple data tensors are
                                         ///< organized for GemmMultipleD operations.

    // ====================== Data Type Configuration ======================

    DataType              a_dtype_;   ///< Matrix A data type. Specifies the precision and format of matrix A elements.
    DataType              b_dtype_;   ///< Matrix B data type. Specifies the precision and format of matrix B elements.
    DataType              c_dtype_;   ///< Matrix C data type. Specifies the precision and format of matrix C elements.
    std::vector<DataType> ds_dtype_;  ///< Data types for data store (DS) tensors. Specifies the precision and format of
                                      ///< multiple DS tensor elements for GemmMultipleD operations.

    // ====================== Epilogue Configuration ======================

    GemmKind kind_;  ///< Type of GEMM operation (standard, multiple D, batched). Determines the device implementation
                     ///< template to use.
    EpilogueType
        c_element_op_;  ///< Type of epilogue operation applied after GEMM computation (e.g., ReLU, addition, GELU).

    // ====================== Specialization Configuration ======================

    GemmSpecialization
        gemm_spec_;  ///< GEMM specialization for M/N/K padding optimization. Determines which padding version to use.
    BlockGemmPipelineVersion
        pipeline_version_;  ///< The version of blockwise-gemm pipeline determining computation strategy.
    BlockGemmPipelineScheduler
        pipeline_scheduler_;  ///< The scheduler type for blockwise-gemm pipeline controlling wave execution.

    // ====================== Tiling and Transfer Configuration ======================

    GemmTileDesc tile_desc_;  ///< Tile configuration for this GEMM operation. Defines how computation is divided across
                              ///< blocks and matrix multiplication units.

    BlockTransferDesc a_block_transfer_desc_;  ///< Data transfer configuration for matrix A. Specifies how A data moves
                                               ///< from global memory to local data store.
    BlockTransferDesc b_block_transfer_desc_;  ///< Data transfer configuration for matrix B. Specifies how B data moves
                                               ///< from global memory to local data store.

    CBlockTransferDesc c_block_transfer_desc_;  ///< Output transfer configuration for matrix C. Controls how
                                                ///< computation results are written from registers to global memory.
};

}  // namespace flashck