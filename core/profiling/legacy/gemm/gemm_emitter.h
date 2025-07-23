#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/profiling/legacy/gemm/gemm_codegen.h"
#include "core/profiling/legacy/gemm/gemm_library.h"
#include "core/profiling/legacy/gemm/gemm_problem.h"
#include "core/utils/common.h"

namespace flashck {

const std::vector<GemmTileDesc> g_gemm_tile_descriptions = {
    // clang-format off
//  Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl| 
//   Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per| 
//       |      |      |      |    |    |     |     | Wave| Wave|
//       |      |      |      |    |    |     |     |     |     |
{   256,   256,   128,    32,   8,   8,   32,   32,    4,    2},
{   256,   128,   256,    32,   8,   8,   32,   32,    2,    4},
{   128,   128,   128,    32,   8,   8,   32,   32,    4,    2},
{   256,   128,   128,    32,   8,   8,   32,   32,    2,    2},
{   128,   128,    64,    32,   8,   8,   32,   32,    2,    2},
{   128,    64,   128,    32,   8,   8,   32,   32,    2,    2},
{   256,   128,    64,    32,   8,   8,   32,   32,    2,    1},
{   256,    64,   128,    32,   8,   8,   32,   32,    1,    2},
    // clang-format on
};

const std::vector<BlockTransferDesc> g_a_block_descriptions_rowmajor = {
    // clang-format off
//  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|
// Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
{     {4, 64, 1},      {1, 0, 2},      {1, 0, 2},              2,              8,              8,         1},
{     {4, 64, 1},      {1, 0, 2},      {1, 0, 2},              2,              8,              8,         1},
{     {4, 32, 1},      {1, 0, 2},      {1, 0, 2},              2,              8,              8,         1},
{     {4, 64, 1},      {1, 0, 2},      {1, 0, 2},              2,              8,              8,         1},
{     {4, 32, 1},      {1, 0, 2},      {1, 0, 2},              2,              8,              8,         1},
{     {4, 32, 1},      {1, 0, 2},      {1, 0, 2},              2,              8,              8,         1},
{     {4, 64, 1},      {1, 0, 2},      {1, 0, 2},              2,              8,              8,         1},
{     {4, 64, 1},      {1, 0, 2},      {1, 0, 2},              2,              8,              8,         1},
    // clang-format on
};

const std::vector<BlockTransferDesc> g_b_block_descriptions_colmajor = {
    // clang-format off
//  BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|
// Lengths_K0_N_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
{    {4, 64, 1},     {1, 0, 2},     {1, 0, 2},              2,              8,              8,         1},
{    {4, 64, 1},     {1, 0, 2},     {1, 0, 2},              2,              8,              8,         1},
{    {4, 32, 1},     {1, 0, 2},     {1, 0, 2},              2,              8,              8,         1},
{    {4, 64, 1},     {1, 0, 2},     {1, 0, 2},              2,              8,              8,         1},
{    {4, 32, 1},     {1, 0, 2},     {1, 0, 2},              2,              8,              8,         1},
{    {4, 32, 1},     {1, 0, 2},     {1, 0, 2},              2,              8,              8,         1},
{    {4, 64, 1},     {1, 0, 2},     {1, 0, 2},              2,              8,              8,         1},
{    {4, 64, 1},     {1, 0, 2},     {1, 0, 2},              2,              8,              8,         1},
    // clang-format on
};

const std::vector<BlockTransferDesc> g_a_block_descriptions_colmajor = {
    // clang-format off
//  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|
// Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
{     {4, 64, 1},      {2, 0, 1},      {2, 0, 1},              1,              8,              8,         1},
{     {4, 64, 1},      {2, 0, 1},      {2, 0, 1},              1,              8,              8,         1},
{     {4, 32, 1},      {2, 0, 1},      {2, 0, 1},              1,              8,              8,         1},
{     {4, 64, 1},      {2, 0, 1},      {2, 0, 1},              1,              8,              8,         1},
{     {4, 32, 1},      {2, 0, 1},      {2, 0, 1},              1,              8,              8,         1},
{     {4, 32, 1},      {2, 0, 1},      {2, 0, 1},              1,              8,              8,         1},
{     {4, 64, 1},      {2, 0, 1},      {2, 0, 1},              1,              8,              8,         1},
{     {4, 64, 1},      {2, 0, 1},      {2, 0, 1},              1,              8,              8,         1},
    // clang-format on
};

const std::vector<BlockTransferDesc> g_b_block_descriptions_rowmajor = {
    // clang-format off
//  BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|
// Lengths_K0_N_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
{    {4, 64, 1},     {2, 0, 1},     {2, 0, 1},              1,              8,              8,         1},
{    {4, 64, 1},     {2, 0, 1},     {2, 0, 1},              1,              8,              8,         1},
{    {4, 32, 1},     {2, 0, 1},     {2, 0, 1},              1,              8,              8,         1},
{    {4, 64, 1},     {2, 0, 1},     {2, 0, 1},              1,              8,              8,         1},
{    {4, 32, 1},     {2, 0, 1},     {2, 0, 1},              1,              8,              8,         1},
{    {4, 32, 1},     {2, 0, 1},     {2, 0, 1},              1,              8,              8,         1},
{    {4, 64, 1},     {2, 0, 1},     {2, 0, 1},              1,              8,              8,         1},
{    {4, 64, 1},     {2, 0, 1},     {2, 0, 1},              1,              8,              8,         1},
    // clang-format on
};

const std::vector<CBlockTransferDesc> g_c_block_descriptions = {
    // clang-format off
// CBlockTransferClusterLengths|  CBlockTransfer
//         _MBlock_MWaveMPerXdl| ScalarPerVector
//         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl
//                             |                
{1, 1,              {1, 32, 1, 8},               8},
{1, 1,              {1, 32, 1, 8},               8},
{1, 1,              {1, 16, 1, 8},               8},
{1, 1,              {1, 32, 1, 8},               8},
{1, 1,              {1, 32, 1, 4},               8},
{1, 1,              {1, 16, 1, 8},               8},
{1, 1,              {1, 32, 1, 8},               8},
{1, 1,              {1, 32, 1, 8},               8},
    // clang-format on
};

// GemmSpecialization name mapping for backward compatibility
const std::map<std::string, GemmSpecialization> g_gemm_spec_names = {{"", GemmSpecialization::Default},
                                                                     {"M", GemmSpecialization::MPadding},
                                                                     {"N", GemmSpecialization::NPadding},
                                                                     {"K", GemmSpecialization::KPadding},
                                                                     {"MN", GemmSpecialization::MNPadding},
                                                                     {"MK", GemmSpecialization::MKPadding},
                                                                     {"NK", GemmSpecialization::NKPadding},
                                                                     {"MNK", GemmSpecialization::MNKPadding}};

/**
 * @class GemmEmitter
 * @brief Manages GEMM operation code generation and tile descriptor selection
 *
 * This class provides functionality to generate GEMM operation instances based on
 * different strategies (heuristic, autotuning, or hybrid) and manages tile
 * descriptor validation and filtering.
 */
class GemmEmitter {
public:
    GemmEmitter()  = default;
    ~GemmEmitter() = default;

    // Delete copy constructor and assignment operator to maintain singleton pattern
    GemmEmitter(const GemmEmitter&)            = delete;
    GemmEmitter& operator=(const GemmEmitter&) = delete;

    /**
     * @brief Get singleton instance of GemmEmitter
     * @return Pointer to the singleton instance
     */
    static GemmEmitter* GetInstance()
    {
        static GemmEmitter instance;
        return &instance;
    }

    /**
     * @brief Validates if a tile descriptor is valid for the given problem
     * @param tile_desc The tile descriptor to validate
     * @param gemm_problem The GEMM problem configuration
     * @return true if tile descriptor is valid, false otherwise
     */
    bool IsValidTile(const GemmTileDesc& tile_desc, const GemmProblem& gemm_problem) const;

    /**
     * @brief Applies heuristic filtering to tile descriptors
     * @param gemm_tile_desc Vector of tile descriptors to filter
     * @param gemm_problem The GEMM problem configuration
     * @return Vector of filtered tile descriptors
     */
    std::vector<GemmTileDesc> HeuristicFilter(const std::vector<GemmTileDesc>& gemm_tile_desc,
                                              const GemmProblem&               gemm_problem) const;

    /**
     * @brief Generates GEMM operation instances based on the problem specification
     * @param gemm_problem The GEMM problem configuration
     */
    void GenerateInstances(GemmProblem& gemm_problem);

    /**
     * @brief Gets the total number of generated instances
     * @return Number of generated instances
     */
    int64_t GetNumInstances() const;

    /**
     * @brief Get profiling instance map for the given GEMM kind
     * @param gemm_problem The GEMM problem configuration
     * @return Reference to the instance map for the specific GEMM kind
     */
    std::map<std::string, GemmCodegen>& GetInstanceMap(GemmProblem gemm_problem)
    {
        GenerateInstances(gemm_problem);
        return instance_map_[gemm_problem.kind_];
    }

    /**
     * @brief Clears all generated instances and resets counters
     */
    void ClearInstances();

private:
    /**
     * @brief Creates a GemmCodegen instance from problem and tile descriptor
     * @param gemm_problem The GEMM problem configuration
     * @param tile_desc The tile descriptor
     * @return Configured GemmCodegen instance
     */
    GemmCodegen CreateGemmCodegen(const GemmProblem& gemm_problem, const GemmTileDesc& tile_desc) const;

    /**
     * @brief Finds the index of a tile descriptor in the global tile descriptions array
     * @param tile_desc The tile descriptor to find
     * @return The index in g_gemm_tile_descriptions, or 0 if not found
     */
    size_t FindTileDescriptorIndex(const GemmTileDesc& tile_desc) const;

    /**
     * @brief Validates mode parameter and throws if invalid
     * @param mode The mode to validate
     */
    void ValidateMode(int mode) const;

    /**
     * @brief Determines the appropriate GEMM specialization based on problem dimensions and tile configuration
     * @param gemm_problem The GEMM problem configuration
     * @param tile_desc The tile descriptor
     * @return The appropriate GemmSpecialization
     */
    GemmSpecialization DetermineGemmSpecialization(const GemmProblem&  gemm_problem,
                                                   const GemmTileDesc& tile_desc) const;

    /**
     * @brief Determines the appropriate pipeline version and scheduler based on problem characteristics
     * @param gemm_problem The GEMM problem configuration
     * @return A pair containing the pipeline version and scheduler
     */
    std::pair<BlockGemmPipelineVersion, BlockGemmPipelineScheduler>
    DeterminePipelineConfiguration(const GemmProblem& gemm_problem) const;

    /**
     * @brief Configures block transfer descriptors for A, B, and C matrices based on layout
     * @param codegen The GemmCodegen instance to configure
     * @param gemm_problem The GEMM problem configuration
     */
    void
    ConfigureBlockTransferDescriptors(GemmCodegen& codegen, const GemmProblem& gemm_problem, size_t tile_index) const;

    std::map<GemmKind, std::map<std::string, GemmCodegen>> instance_map_;
    int64_t                                                num_instances_ = 0;
};

}  // namespace flashck