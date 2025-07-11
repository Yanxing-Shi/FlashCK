#include "flashck/core/profiling/tile/norm/norm_emitter.h"

#include <algorithm>
#include <stdexcept>

#include "flashck/core/utils/common.h"

namespace flashck {

bool NormEmitter::IsValidTile(const NormTileDesc& tile_desc, const NormProblem& norm_problem) const
{
    // Validate tile descriptor parameters
    if (tile_desc.repeat_m <= 0 || tile_desc.repeat_n <= 0 || tile_desc.thread_per_block_m <= 0
        || tile_desc.thread_per_block_n <= 0 || tile_desc.vector_n <= 0) {
        VLOG(3) << "Invalid tile descriptor: negative or zero values not allowed";
        return false;
    }

    // Validate thread block dimensions
    const int total_threads = tile_desc.thread_per_block_m * tile_desc.thread_per_block_n;
    if (total_threads > 1024) {  // Common GPU thread block limit
        VLOG(3) << "Invalid tile descriptor: thread block size " << total_threads << " exceeds limit (1024)";
        return false;
    }

    // Validate vector size alignment
    if (tile_desc.vector_n > tile_desc.thread_per_block_n) {
        VLOG(3) << "Invalid tile descriptor: vector_n (" << tile_desc.vector_n << ") cannot exceed thread_per_block_n ("
                << tile_desc.thread_per_block_n << ")";
        return false;
    }

    // Validate against problem dimensions
    const int effective_m = tile_desc.repeat_m * tile_desc.thread_per_block_m;
    const int effective_n = tile_desc.repeat_n * tile_desc.thread_per_block_n;

    if (effective_m > norm_problem.m || effective_n > norm_problem.n) {
        VLOG(3) << "Invalid tile descriptor: effective dimensions (" << effective_m << "x" << effective_n
                << ") exceed problem dimensions (" << norm_problem.m << "x" << norm_problem.n << ")";
        return false;
    }

    return true;
}

std::vector<NormTileDesc> NormEmitter::HeuristicFilter(const std::vector<NormTileDesc>& norm_tile_desc,
                                                       const NormProblem&               norm_problem) const
{
    std::vector<NormTileDesc> filtered_tile_desc;

    for (const auto& tile_desc : norm_tile_desc) {
        // Enhanced heuristic based on problem characteristics
        bool should_include = false;

        // For small problems, prefer smaller tile sizes
        if (norm_problem.m <= 64 && norm_problem.n <= 64) {
            if (tile_desc.repeat_m == 1 && tile_desc.repeat_n == 1 && tile_desc.thread_per_block_m <= 8
                && tile_desc.thread_per_block_n <= 8) {
                should_include = true;
            }
        }
        // For medium problems, prefer balanced tiles
        else if (norm_problem.m <= 256 && norm_problem.n <= 256) {
            if (tile_desc.thread_per_block_m == 4 && tile_desc.thread_per_block_n == 16) {
                should_include = true;
            }
        }
        // For large problems, prefer larger tiles with higher vectorization
        else {
            if (tile_desc.thread_per_block_m == 4 && tile_desc.thread_per_block_n == 64 && tile_desc.vector_n >= 2) {
                should_include = true;
            }
        }

        if (should_include) {
            filtered_tile_desc.push_back(tile_desc);
            VLOG(2) << "Selected tile descriptor: " << tile_desc.ToString();
        }
        else {
            VLOG(3) << "Filtered out tile descriptor: " << tile_desc.ToString();
        }
    }

    // Ensure we have at least one tile descriptor
    if (filtered_tile_desc.empty() && !norm_tile_desc.empty()) {
        LOG(WARNING) << "No tile descriptors passed heuristic filter, using first valid tile";
        filtered_tile_desc.push_back(norm_tile_desc[0]);
    }

    return filtered_tile_desc;
}

void NormEmitter::ValidateMode(int mode) const
{
    if (mode < 0 || mode > 2) {
        FC_THROW(Unavailable("Unsupported mode: {}, valid modes are 0 (heuristic), 1 (autotuning), 2 (hybrid)", mode));
    }
}

NormCodeGen NormEmitter::CreateNormCodeGen(const NormProblem& norm_problem, const NormTileDesc& tile_desc) const
{
    NormCodeGen norm;

    norm.kind      = norm_problem.kind;
    norm.tile_desc = tile_desc;

    // Copy data type information
    norm.x_dtype_            = norm_problem.x_dtype;
    norm.y_dtype_            = norm_problem.y_dtype;
    norm.smooth_scale_dtype_ = norm_problem.smooth_scale_dtype_;
    norm.y_scale_dtype_      = norm_problem.y_scale_dtype_;

    // Copy operation configuration
    norm.is_add_bias_ = norm_problem.is_add_bias_;
    norm.fused_add_   = norm_problem.fused_add_;
    norm.fused_quant_ = norm_problem.fused_quant_;

    return norm;
}

std::map<std::string, std::unique_ptr<NormCodeGen>> NormEmitter::GenerateInstances(const NormProblem& norm_problem)
{
    ValidateMode(FLAGS_FC_mode);

    // Clear previous instances
    instance_map_.clear();
    num_instances_ = 0;

    // Filter valid tile descriptors
    std::vector<NormTileDesc> valid_tile_desc;
    for (const auto& tile_desc : g_default_norm_tile_desc) {
        if (IsValidTile(tile_desc, norm_problem)) {
            valid_tile_desc.push_back(tile_desc);
        }
    }

    if (valid_tile_desc.empty()) {
        LOG(ERROR) << "No valid tile descriptors found for problem: " << norm_problem.ToString();
        return instance_map_;
    }

    VLOG(1) << "Found " << valid_tile_desc.size() << " valid tile descriptors";

    // Generate operation instances based on mode
    std::vector<NormTileDesc> selected_tile_desc;

    switch (FLAGS_FC_mode) {
        case 0: {
            // Heuristic mode: select single best tile
            auto filtered_desc = HeuristicFilter(valid_tile_desc, norm_problem);
            if (filtered_desc.size() != 1) {
                LOG(WARNING) << "Heuristic mode expected 1 tile descriptor, got " << filtered_desc.size()
                             << ". Using first descriptor.";
            }
            selected_tile_desc = {filtered_desc.empty() ? valid_tile_desc[0] : filtered_desc[0]};
            break;
        }
        case 1: {
            // Autotuning mode: use all valid tiles
            selected_tile_desc = valid_tile_desc;
            break;
        }
        case 2: {
            // Hybrid mode: use heuristic filter but allow multiple tiles
            selected_tile_desc = HeuristicFilter(valid_tile_desc, norm_problem);
            if (selected_tile_desc.empty()) {
                LOG(WARNING) << "Hybrid mode heuristic returned no tiles, falling back to all valid tiles";
                selected_tile_desc = valid_tile_desc;
            }
            break;
        }
        default:
            FC_THROW(std::invalid_argument("Invalid mode: " + std::to_string(FLAGS_FC_mode)));
    }

    // Generate code instances
    for (const auto& tile_desc : selected_tile_desc) {
        NormCodeGen norm = CreateNormCodeGen(norm_problem, tile_desc);

        std::string config_name               = norm.GetConfigName();
        instance_map_[norm.kind][config_name] = std::make_unique<NormCodeGen>(std::move(norm));
        num_instances_++;

        VLOG(2) << "Generated norm instance: " << config_name;
    }

    LOG(INFO) << "Generated " << selected_tile_desc.size() << " norm operation instances for mode " << FLAGS_FC_mode;

    return instance_map_;
}

int64_t NormEmitter::GetNumInstances() const
{
    return num_instances_;
}

void NormEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
}

}  // namespace flashck