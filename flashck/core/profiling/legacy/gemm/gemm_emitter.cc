#include "flashck/core/profiling/legacy/gemm/gemm_emitter.h"

#include <algorithm>
#include <stdexcept>

#include "flashck/core/utils/common.h"

FC_DECLARE_int32(FC_TUNING_MODE);  // Mode for GEMM operation: 0 - heuristic, 1 - autotuning, 2 - hybrid

namespace flashck {

bool GemmEmitter::IsValidTile(const GemmTileDesc& tile_desc, const GemmProblem& gemm_problem) const
{
    // Validate tile descriptor parameters
    if (tile_desc.block_size_ <= 0 || tile_desc.m_per_block_ <= 0 || tile_desc.n_per_block_ <= 0
        || tile_desc.k_per_block_ <= 0) {
        VLOG(3) << "Invalid tile descriptor: negative or zero values not allowed";
        return false;
    }

    // Validate block size constraints
    if (tile_desc.block_size_ > 1024) {  // Common GPU thread block limit
        VLOG(3) << "Invalid tile descriptor: block size " << tile_desc.block_size_ << " exceeds limit (1024)";
        return false;
    }

    // Validate XDL dimensions
    if (tile_desc.m_per_xdl_ <= 0 || tile_desc.n_per_xdl_ <= 0 || tile_desc.m_xdl_per_wave_ <= 0
        || tile_desc.n_xdl_per_wave_ <= 0) {
        VLOG(3) << "Invalid tile descriptor: XDL dimensions must be positive";
        return false;
    }

    // Validate K1 dimensions
    if (tile_desc.a_k1_ <= 0 || tile_desc.b_k1_ <= 0) {
        VLOG(3) << "Invalid tile descriptor: K1 dimensions must be positive";
        return false;
    }

    // Validate against problem dimensions
    if (tile_desc.m_per_block_ > gemm_problem.m_ || tile_desc.n_per_block_ > gemm_problem.n_
        || tile_desc.k_per_block_ > gemm_problem.k_) {
        VLOG(3) << "Invalid tile descriptor: block dimensions (" << tile_desc.m_per_block_ << "x"
                << tile_desc.n_per_block_ << "x" << tile_desc.k_per_block_ << ") exceed problem dimensions ("
                << gemm_problem.m_ << "x" << gemm_problem.n_ << "x" << gemm_problem.k_ << ")";
        return false;
    }

    // Validate XDL configuration consistency
    const int total_xdl_per_block_m = tile_desc.m_xdl_per_wave_ * tile_desc.m_per_xdl_;
    const int total_xdl_per_block_n = tile_desc.n_xdl_per_wave_ * tile_desc.n_per_xdl_;

    if (total_xdl_per_block_m > tile_desc.m_per_block_ || total_xdl_per_block_n > tile_desc.n_per_block_) {
        VLOG(3) << "Invalid tile descriptor: XDL configuration exceeds block dimensions";
        return false;
    }

    return true;
}

std::vector<GemmTileDesc> GemmEmitter::HeuristicFilter(const std::vector<GemmTileDesc>& gemm_tile_desc,
                                                       const GemmProblem&               gemm_problem) const
{
    std::vector<GemmTileDesc> filtered_tile_desc;

    for (const auto& tile_desc : gemm_tile_desc) {
        // Enhanced heuristic based on problem characteristics
        bool should_include = false;

        // For small problems, prefer smaller tile sizes
        if (gemm_problem.m_ <= 128 && gemm_problem.n_ <= 128 && gemm_problem.k_ <= 64) {
            if (tile_desc.m_per_block_ <= 128 && tile_desc.n_per_block_ <= 128 && tile_desc.k_per_block_ <= 32) {
                should_include = true;
            }
        }
        // For medium problems, prefer balanced tiles
        else if (gemm_problem.m_ <= 512 && gemm_problem.n_ <= 512 && gemm_problem.k_ <= 256) {
            if (tile_desc.m_per_block_ == 128 && tile_desc.n_per_block_ == 128 && tile_desc.k_per_block_ == 32) {
                should_include = true;
            }
        }
        // For large problems, prefer larger tiles
        else {
            if (tile_desc.m_per_block_ >= 128 && tile_desc.n_per_block_ >= 128 && tile_desc.k_per_block_ >= 32) {
                should_include = true;
            }
        }

        // Additional heuristics based on GEMM type
        switch (gemm_problem.kind_) {
            case GemmKind::Gemm:
                // Standard GEMM: prefer balanced configurations
                should_include = should_include && (tile_desc.m_xdl_per_wave_ * tile_desc.n_xdl_per_wave_ >= 4);
                break;
            case GemmKind::GemmMultipleD:
                // Multiple D: may need more memory bandwidth
                should_include = should_include && (tile_desc.block_size_ <= 256);
                break;
            case GemmKind::BatchGemm:
                // Batched GEMM: prefer larger tiles for better parallelization
                should_include = should_include && (tile_desc.m_per_block_ >= 128 || tile_desc.n_per_block_ >= 128);
                break;
            default:
                break;
        }

        if (should_include) {
            filtered_tile_desc.push_back(tile_desc);
            VLOG(2) << "Selected tile descriptor: " << tile_desc.GetInstanceName();
        }
        else {
            VLOG(3) << "Filtered out tile descriptor: " << tile_desc.GetInstanceName();
        }
    }

    // Ensure we have at least one tile descriptor
    if (filtered_tile_desc.empty() && !gemm_tile_desc.empty()) {
        LOG(WARNING) << "No tile descriptors passed heuristic filter, using first valid tile";
        for (const auto& tile_desc : gemm_tile_desc) {
            if (IsValidTile(tile_desc, gemm_problem)) {
                filtered_tile_desc.push_back(tile_desc);
                break;
            }
        }
    }

    return filtered_tile_desc;
}

void GemmEmitter::ValidateMode(int mode) const
{
    FC_ENFORCE_EQ(mode == 0 || mode == 1 || mode == 2,
                  true,
                  Unavailable("Unsupported mode: {}, valid modes are 0 (heuristic), 1 (autotuning), 2 (hybrid)", mode));
}

void GemmEmitter::GenerateInstances(GemmProblem& gemm_problem)
{
    // Check if instances already exist for this GEMM kind
    if (instance_map_.find(gemm_problem.kind_) != instance_map_.end() && !instance_map_[gemm_problem.kind_].empty()) {
        VLOG(2) << "Instances already generated for GEMM kind: " << GetGemmKindName(gemm_problem.kind_);
        return;
    }

    ValidateMode(FLAGS_FC_TUNING_MODE);

    std::vector<GemmTileDesc> tile_descriptors;

    switch (FLAGS_FC_TUNING_MODE) {
        case 0:  // Heuristic mode
            VLOG(1) << "Generating instances using heuristic mode for GEMM kind: "
                    << GetGemmKindName(gemm_problem.kind_);
            tile_descriptors = HeuristicFilter(g_gemm_tile_descriptions, gemm_problem);
            break;

        case 1:  // Autotuning mode
            VLOG(1) << "Generating instances using autotuning mode for GEMM kind: "
                    << GetGemmKindName(gemm_problem.kind_);
            // Use all valid tile descriptors for comprehensive search
            for (const auto& tile_desc : g_gemm_tile_descriptions) {
                if (IsValidTile(tile_desc, gemm_problem)) {
                    tile_descriptors.push_back(tile_desc);
                }
            }
            break;

        case 2:  // Hybrid mode
            VLOG(1) << "Generating instances using hybrid mode for GEMM kind: " << GetGemmKindName(gemm_problem.kind_);
            // Start with heuristic filter, then expand if needed
            tile_descriptors = HeuristicFilter(g_gemm_tile_descriptions, gemm_problem);
            if (tile_descriptors.size() < 3) {  // Expand if too few options
                for (const auto& tile_desc : g_gemm_tile_descriptions) {
                    if (IsValidTile(tile_desc, gemm_problem)) {
                        auto it = std::find_if(tile_descriptors.begin(),
                                               tile_descriptors.end(),
                                               [&tile_desc](const GemmTileDesc& existing) {
                                                   return existing.GetInstanceName() == tile_desc.GetInstanceName();
                                               });
                        if (it == tile_descriptors.end()) {
                            tile_descriptors.push_back(tile_desc);
                        }
                    }
                }
            }
            break;

        default:
            FC_THROW(Unavailable("Invalid tuning mode: {}", FLAGS_FC_TUNING_MODE));
    }

    if (tile_descriptors.empty()) {
        FC_THROW(Unavailable("No valid tile descriptors found for GEMM problem"));
    }

    // Generate instances
    std::map<std::string, GemmCodegen>& kind_instance_map = instance_map_[gemm_problem.kind_];
    int64_t                             generated_count   = 0;

    for (const auto& tile_desc : tile_descriptors) {
        try {
            GemmCodegen codegen       = CreateGemmCodegen(gemm_problem, tile_desc);
            std::string instance_name = codegen.GetInstanceName();

            // Avoid duplicates
            if (kind_instance_map.find(instance_name) == kind_instance_map.end()) {
                kind_instance_map[instance_name] = std::move(codegen);
                generated_count++;
                VLOG(2) << "Generated GEMM instance: " << instance_name;
            }
            else {
                VLOG(3) << "Skipped duplicate GEMM instance: " << instance_name;
            }
        }
        catch (const std::exception& e) {
            LOG(WARNING) << "Failed to create GEMM codegen for tile: " << tile_desc.GetInstanceName()
                         << ", error: " << e.what();
        }
    }

    num_instances_ += generated_count;
    VLOG(1) << "Generated " << generated_count << " GEMM instances for kind: " << GetGemmKindName(gemm_problem.kind_)
            << " (total: " << num_instances_ << ")";
}

GemmCodegen GemmEmitter::CreateGemmCodegen(const GemmProblem& gemm_problem, const GemmTileDesc& tile_desc) const
{
    GemmCodegen codegen;

    // Set basic configuration
    codegen.tile_desc_ = tile_desc;

    // Set GEMM operation kind and epilogue
    codegen.kind_         = gemm_problem.kind_;
    codegen.c_element_op_ = gemm_problem.epilogue_;

    // Set data types from problem
    codegen.a_dtype_ = gemm_problem.a_dtype_;
    codegen.b_dtype_ = gemm_problem.b_dtype_;
    codegen.c_dtype_ = gemm_problem.c_dtype_;

    // Set DS data types for GemmMultipleD operations
    if (gemm_problem.kind_ == GemmKind::GemmMultipleD) {
        codegen.ds_dtype_  = gemm_problem.ds_dtype_;
        codegen.ds_layout_ = gemm_problem.ds_layout_;
    }

    // Set layouts from problem
    codegen.a_layout_ = gemm_problem.a_layout_;
    codegen.b_layout_ = gemm_problem.b_layout_;
    codegen.c_layout_ = gemm_problem.c_layout_;

    // Set GEMM specialization based on problem dimensions and requirements
    codegen.gemm_spec_ = DetermineGemmSpecialization(gemm_problem, tile_desc);

    // Set pipeline configuration based on problem size and GEMM kind
    std::tie(codegen.pipeline_version_, codegen.pipeline_scheduler_) = DeterminePipelineConfiguration(gemm_problem);

    // Configure block transfer descriptors based on tile index
    size_t tile_index = FindTileDescriptorIndex(tile_desc);
    ConfigureBlockTransferDescriptors(codegen, gemm_problem, tile_index);

    VLOG(3) << "Created GEMM codegen with spec: " << GetGemmSpecializationName(codegen.gemm_spec_)
            << ", pipeline: " << GetPipelineVersionName(codegen.pipeline_version_)
            << ", scheduler: " << GetSchedulerName(codegen.pipeline_scheduler_);

    return codegen;
}

void GemmEmitter::ConfigureBlockTransferDescriptors(GemmCodegen&       codegen,
                                                    const GemmProblem& gemm_problem,
                                                    size_t             tile_index) const
{
    // Simple 1:1 mapping - each tile descriptor corresponds to the same index in block transfer descriptors
    // This ensures consistent pairing between tile configurations and their corresponding block transfer settings
    //
    // IMPORTANT: All block transfer descriptor arrays must have the same size as g_gemm_tile_descriptions
    // Currently all arrays have 8 elements to match the 8 tile descriptors

    // A matrix block transfer - use corresponding index from appropriate layout array
    if (gemm_problem.a_layout_ == LayoutType::RowMajor && tile_index < g_a_block_descriptions_rowmajor.size()) {
        codegen.a_block_transfer_desc_ = g_a_block_descriptions_rowmajor[tile_index];
    }
    else if (gemm_problem.a_layout_ == LayoutType::ColumnMajor && tile_index < g_a_block_descriptions_colmajor.size()) {
        codegen.a_block_transfer_desc_ = g_a_block_descriptions_colmajor[tile_index];
    }
    else if (!g_a_block_descriptions_rowmajor.empty()) {
        // Fallback to index 0 if tile_index is out of bounds
        codegen.a_block_transfer_desc_ = g_a_block_descriptions_rowmajor[0];
    }

    // B matrix block transfer - use corresponding index from appropriate layout array
    if (gemm_problem.b_layout_ == LayoutType::ColumnMajor && tile_index < g_b_block_descriptions_colmajor.size()) {
        codegen.b_block_transfer_desc_ = g_b_block_descriptions_colmajor[tile_index];
    }
    else if (gemm_problem.b_layout_ == LayoutType::RowMajor && tile_index < g_b_block_descriptions_rowmajor.size()) {
        codegen.b_block_transfer_desc_ = g_b_block_descriptions_rowmajor[tile_index];
    }
    else if (!g_b_block_descriptions_colmajor.empty()) {
        // Fallback to index 0 if tile_index is out of bounds
        codegen.b_block_transfer_desc_ = g_b_block_descriptions_colmajor[0];
    }

    // C matrix block transfer - use corresponding index
    if (tile_index < g_c_block_descriptions.size()) {
        codegen.c_block_transfer_desc_ = g_c_block_descriptions[tile_index];
    }
    else if (!g_c_block_descriptions.empty()) {
        // Fallback to index 0 if tile_index is out of bounds
        codegen.c_block_transfer_desc_ = g_c_block_descriptions[0];
    }
}

size_t GemmEmitter::FindTileDescriptorIndex(const GemmTileDesc& tile_desc) const
{
    // Find the index of the tile descriptor in the global g_gemm_tile_descriptions array
    for (size_t i = 0; i < g_gemm_tile_descriptions.size(); ++i) {
        const auto& global_tile = g_gemm_tile_descriptions[i];
        // Compare all fields to find exact match
        if (global_tile.block_size_ == tile_desc.block_size_ && global_tile.m_per_block_ == tile_desc.m_per_block_
            && global_tile.n_per_block_ == tile_desc.n_per_block_ && global_tile.k_per_block_ == tile_desc.k_per_block_
            && global_tile.a_k1_ == tile_desc.a_k1_ && global_tile.b_k1_ == tile_desc.b_k1_
            && global_tile.m_per_xdl_ == tile_desc.m_per_xdl_ && global_tile.n_per_xdl_ == tile_desc.n_xdl_per_wave_
            && global_tile.m_xdl_per_wave_ == tile_desc.m_xdl_per_wave_
            && global_tile.n_xdl_per_wave_ == tile_desc.n_xdl_per_wave_) {
            return i;
        }
    }

    // If not found, return 0 as fallback index
    VLOG(3) << "Tile descriptor not found in global array, using index 0 as fallback";
    return 0;
}

int64_t GemmEmitter::GetNumInstances() const
{
    return num_instances_;
}

void GemmEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
    VLOG(1) << "Cleared all GEMM instances";
}

GemmSpecialization GemmEmitter::DetermineGemmSpecialization(const GemmProblem&  gemm_problem,
                                                            const GemmTileDesc& tile_desc) const
{
    // Lambda for integer division with ceiling
    auto IntegerDivideCeil = [](int64_t dividend, int64_t divisor) -> int64_t {
        return (dividend + divisor - 1) / divisor;
    };

    if (gemm_problem.m_ % tile_desc.m_per_block_ != 0 && gemm_problem.n_ % tile_desc.n_per_block_ != 0
        && gemm_problem.k_ % tile_desc.k_per_block_ != 0) {
        return GemmSpecialization::MNKPadding;
    }
    else if (gemm_problem.m_ % tile_desc.m_per_block_ != 0 && gemm_problem.n_ % tile_desc.n_per_block_ != 0) {
        return GemmSpecialization::MNPadding;
    }
    else if (gemm_problem.m_ % tile_desc.m_per_block_ != 0 && gemm_problem.k_ % tile_desc.k_per_block_ != 0) {
        return GemmSpecialization::MKPadding;
    }
    else if (gemm_problem.n_ % tile_desc.n_per_block_ != 0 && gemm_problem.k_ % tile_desc.k_per_block_ != 0) {
        return GemmSpecialization::NKPadding;
    }
    else if (gemm_problem.m_ % tile_desc.m_per_block_ != 0) {
        return GemmSpecialization::MPadding;
    }
    else if (gemm_problem.n_ % tile_desc.n_per_block_ != 0) {
        return GemmSpecialization::NPadding;
    }
    else if (gemm_problem.k_ % tile_desc.k_per_block_ != 0) {
        return GemmSpecialization::KPadding;
    }
    else {
        return GemmSpecialization::Default;
    }
}

std::pair<BlockGemmPipelineVersion, BlockGemmPipelineScheduler>
GemmEmitter::DeterminePipelineConfiguration(const GemmProblem& gemm_problem) const
{
    // Calculate problem complexity metric
    int64_t complexity = gemm_problem.m_ * gemm_problem.n_ * gemm_problem.k_;

    if (complexity > 1024 * 1024 * 1024) {
        // Large problems: use compute optimized pipeline
        return {BlockGemmPipelineVersion::V3, BlockGemmPipelineScheduler::Interwave};
    }
    else if (gemm_problem.kind_ == GemmKind::GemmMultipleD) {
        // Multiple D operations: use memory optimized pipeline
        return {BlockGemmPipelineVersion::V2, BlockGemmPipelineScheduler::Intrawave};
    }
    else {
        // Default configuration
        return {BlockGemmPipelineVersion::V1, BlockGemmPipelineScheduler::Intrawave};
    }
}

}  // namespace flashck