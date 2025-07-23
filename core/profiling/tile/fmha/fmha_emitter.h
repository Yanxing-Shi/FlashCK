
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/profiling/tile/fmha/fmha_fwd_appendkv_codegen.h"
#include "core/profiling/tile/fmha/fmha_fwd_codegen.h"
#include "core/profiling/tile/fmha/fmha_fwd_splitkv_codegen.h"
#include "core/profiling/tile/fmha/fmha_fwd_splitkv_combine_codegen.h"

#include "core/profiling/tile/fmha/fmha_library.h"
#include "core/profiling/tile/fmha/fmha_problem.h"

namespace flashck {

// Global tile descriptor configurations for different FMHA operations
const std::vector<FmhaTileDesc> g_fmha_tile_descriptions = {
    // clang-format off
    //  bm0, bn0, bn1, bk0_max, bk1, bm1, bm1_per_wave, bn1_per_wave, bk1_per_wave, bm1_repeat, bn1_repeat, bk1_repeat, c_shuffle_block_transfer_cluster_lengths_mblock_mperblock_nblock_nperblock, c_shuffle_block_transfer_scalar_per_vector_nperblock
    {128, 64,  16,  32,  32,  32, 2, 1, 1, 2, 1, 1, 32, 32, 16, 32, 32, 16},
    {128, 128, 32,  256, 32,  256, 4, 1, 1, 4, 1, 1, 32, 32, 16, 32, 32, 16},
    {128, 64,  32,  64,  32,  64, 4, 1, 1, 4, 1, 1, 32, 32, 16, 32, 32, 16},
    {128, 128, 32,  128, 32,  128, 4, 1, 1, 4, 1, 1, 32, 32, 16, 32, 32, 16},
    {128, 128, 32,  256, 32,  256, 4, 1, 1, 4, 1, 1, 32, 32, 16, 32, 32, 16}
    // clang-format on
};

const std::vector<FmhaAppendKVTileDesc> g_fmha_appendkv_tile_descriptions = {
    // clang-format off
    //  bs,  bsk, bd,  bdv
    {64,  64,  32,  32},
    {128, 64,  32,  32},
    {64,  128, 32,  32},
    {128, 128, 32,  32}
    // clang-format on
};

const std::vector<FmhaSplitKVCombineTileDesc> g_fmha_splitkv_combine_tile_descriptions = {
    // clang-format off
    //  bm0, bn1
    {128, 32},
    {64,  32},
    {128, 64},
    {64,  64}
    // clang-format on
};

/**
 * @class FmhaEmitter
 * @brief Manages FMHA operation code generation and tile descriptor selection
 *
 * This class provides functionality to generate FMHA operation instances based on
 * different strategies (heuristic, autotuning, or hybrid) and manages tile
 * descriptor validation and filtering. Interface is designed to be consistent with GemmEmitter.
 */
class FmhaEmitter {
public:
    FmhaEmitter()  = default;
    ~FmhaEmitter() = default;

    // Delete copy constructor and assignment operator to maintain singleton pattern
    FmhaEmitter(const FmhaEmitter&)            = delete;
    FmhaEmitter& operator=(const FmhaEmitter&) = delete;

    /**
     * @brief Get singleton instance of FmhaEmitter
     * @return Pointer to the singleton instance
     */
    static FmhaEmitter* GetInstance()
    {
        static FmhaEmitter instance;
        return &instance;
    }

    /**
     * @brief Validates if a tile descriptor is valid for the given problem
     * @param tile_desc The tile descriptor to validate
     * @param fmha_problem The FMHA problem configuration
     * @return true if tile descriptor is valid, false otherwise
     */
    bool IsValidTile(const FmhaTileDesc& tile_desc, const FmhaProblem& fmha_problem) const;

    /**
     * @brief Validates if an AppendKV tile descriptor is valid for the given problem
     * @param tile_desc The AppendKV tile descriptor to validate
     * @param fmha_problem The FMHA problem configuration
     * @return true if tile descriptor is valid, false otherwise
     */
    bool IsValidTile(const FmhaAppendKVTileDesc& tile_desc, const FmhaProblem& fmha_problem) const;

    /**
     * @brief Validates if a SplitKV Combine tile descriptor is valid for the given problem
     * @param tile_desc The SplitKV Combine tile descriptor to validate
     * @param fmha_problem The FMHA problem configuration
     * @return true if tile descriptor is valid, false otherwise
     */
    bool IsValidTile(const FmhaSplitKVCombineTileDesc& tile_desc, const FmhaProblem& fmha_problem) const;

    /**
     * @brief Applies heuristic filtering to tile descriptors
     * @param fmha_tile_desc Vector of tile descriptors to filter
     * @param fmha_problem The FMHA problem configuration
     * @return Vector of filtered tile descriptors
     */
    std::vector<FmhaTileDesc> HeuristicFilter(const std::vector<FmhaTileDesc>& fmha_tile_desc,
                                              const FmhaProblem&               fmha_problem) const;

    /**
     * @brief Applies heuristic filtering to AppendKV tile descriptors
     * @param fmha_tile_desc Vector of AppendKV tile descriptors to filter
     * @param fmha_problem The FMHA problem configuration
     * @return Vector of filtered tile descriptors
     */
    std::vector<FmhaAppendKVTileDesc> HeuristicFilter(const std::vector<FmhaAppendKVTileDesc>& fmha_tile_desc,
                                                      const FmhaProblem&                       fmha_problem) const;

    /**
     * @brief Applies heuristic filtering to SplitKV Combine tile descriptors
     * @param fmha_tile_desc Vector of SplitKV Combine tile descriptors to filter
     * @param fmha_problem The FMHA problem configuration
     * @return Vector of filtered tile descriptors
     */
    std::vector<FmhaSplitKVCombineTileDesc>
    HeuristicFilter(const std::vector<FmhaSplitKVCombineTileDesc>& fmha_tile_desc,
                    const FmhaProblem&                             fmha_problem) const;

    /**
     * @brief Generates FMHA operation instances based on the problem specification
     * @param fmha_problem The FMHA problem configuration
     */
    void GenerateInstances(FmhaProblem& fmha_problem);

    /**
     * @brief Gets the total number of generated instances
     * @return Number of generated instances
     */
    int64_t GetNumInstances() const
    {
        return num_instances_;
    }

    /**
     * @brief Get profiling instance map for the given FMHA kind
     * @param fmha_problem The FMHA problem configuration
     * @return Reference to the instance map for the specific FMHA kind
     */
    std::map<std::string, std::string>& GetInstanceMap(FmhaProblem fmha_problem)
    {
        GenerateInstances(fmha_problem);
        return instance_map_[fmha_problem.kind_];
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:
    /**
     * @brief Validates mode parameter and throws if invalid
     * @param mode The mode to validate
     */
    void ValidateMode(int mode) const;

    /**
     * @brief Creates FMHA operation instances for Forward operations
     * @param fmha_problem The FMHA problem configuration
     * @param tile_descriptors Vector of tile descriptors to use
     */
    void CreateFwdInstances(const FmhaProblem& fmha_problem, const std::vector<FmhaTileDesc>& tile_descriptors);

    /**
     * @brief Creates FMHA operation instances for Split-KV operations
     * @param fmha_problem The FMHA problem configuration
     * @param tile_descriptors Vector of tile descriptors to use
     */
    void CreateSplitKVInstances(const FmhaProblem& fmha_problem, const std::vector<FmhaTileDesc>& tile_descriptors);

    /**
     * @brief Creates FMHA operation instances for Split-KV Combine operations
     * @param fmha_problem The FMHA problem configuration
     * @param tile_descriptors Vector of tile descriptors to use
     */
    void CreateSplitKVCombineInstances(const FmhaProblem&                             fmha_problem,
                                       const std::vector<FmhaSplitKVCombineTileDesc>& tile_descriptors);

    /**
     * @brief Creates FMHA operation instances for Append-KV operations
     * @param fmha_problem The FMHA problem configuration
     * @param tile_descriptors Vector of tile descriptors to use
     */
    void CreateAppendKVInstances(const FmhaProblem&                       fmha_problem,
                                 const std::vector<FmhaAppendKVTileDesc>& tile_descriptors);

    /**
     * @brief Creates a single Forward FMHA operation instance
     * @param fmha_problem The FMHA problem configuration
     * @param tile_desc The tile descriptor
     * @return Generated FMHA operation instance
     */
    FmhaFwdCodeGen GenFmhaFwdInstance(const FmhaProblem& fmha_problem, const FmhaTileDesc& tile_desc) const;

    /**
     * @brief Creates a single Split-KV FMHA operation instance
     * @param fmha_problem The FMHA problem configuration
     * @param tile_desc The tile descriptor
     * @return Generated FMHA operation instance
     */
    FmhaFwdSplitKVCodeGen GenFmhaFwdSplitKVInstance(const FmhaProblem&  fmha_problem,
                                                    const FmhaTileDesc& tile_desc) const;

    /**
     * @brief Creates a single Split-KV Combine FMHA operation instance
     * @param fmha_problem The FMHA problem configuration
     * @param tile_desc The tile descriptor
     * @return Generated FMHA operation instance
     */
    FmhaFwdSplitKVCombineCodeGen GenFmhaFwdSplitKVCombineInstance(const FmhaProblem&                fmha_problem,
                                                                  const FmhaSplitKVCombineTileDesc& tile_desc) const;

    /**
     * @brief Creates a single Append-KV FMHA operation instance
     * @param fmha_problem The FMHA problem configuration
     * @param tile_desc The tile descriptor
     * @return Generated FMHA operation instance
     */
    FmhaFwdAppendKVCodeGen GenFmhaFwdAppendKVInstance(const FmhaProblem&          fmha_problem,
                                                      const FmhaAppendKVTileDesc& tile_desc) const;

    /**
     * @brief Determines appropriate padding configuration based on problem and tile
     * @param problem The FMHA problem configuration
     * @param operation_mode The operation mode (Batch/Group)
     * @param pipeline The pipeline type
     * @return Padding configuration flags
     */
    struct PaddingConfig {
        bool is_pad_q_seq_len    = false;
        bool is_pad_kv_seq_len   = false;
        bool is_pad_qk_head_dim  = false;
        bool is_pad_v_head_dim   = false;
        bool is_pad_qkv_head_dim = false;
    };

    PaddingConfig DetermineFwdPaddingConfig(const FmhaProblem&    problem,
                                            const FmhaTileDesc&   tile_desc,
                                            FmhaMode              operation_mode,
                                            BlockFmhaPipelineEnum pipeline = BlockFmhaPipelineEnum::QRKSVS) const;

    PaddingConfig DetermineSplitKVPaddingConfig(const FmhaProblem&  problem,
                                                const FmhaTileDesc& tile_desc,
                                                FmhaMode            operation_mode) const;

    PaddingConfig DetermineSplitKVCombinePaddingConfig(const FmhaProblem&                problem,
                                                       const FmhaSplitKVCombineTileDesc& tile_desc,
                                                       FmhaMode                          operation_mode) const;

    PaddingConfig DetermineAppendKVPaddingConfig(const FmhaProblem&          problem,
                                                 const FmhaAppendKVTileDesc& tile_desc,
                                                 FmhaMode                    operation_mode) const;

    /**
     * @brief Calculates log2 of maximum splits for Split-KV Combine operations
     * @param num_splits Number of splits
     * @return Log2 of maximum splits
     */
    int CalculateLogMaxSplits(int num_splits) const;

    std::map<FmhaKind, std::map<std::string, std::string>> instance_map_;
    int64_t                                                num_instances_ = 0;
};

}  // namespace flashck