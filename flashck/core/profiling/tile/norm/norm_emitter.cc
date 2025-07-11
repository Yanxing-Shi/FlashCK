#pragma once

#include "flashck/core/profiling/tile/norm/norm_codegen.h"
#include "flashck/core/profiling/tile/norm/norm_problem.h"

FC_DECLARE_int32(mode);

namespace flashck {

// check if the tile is valid
bool NormEmitter::IsValidTile(const NormTileDesc& tile_desc)
{
    // To Do: Add more validation logic if needed
    return true;
}

// Append a tile descriptor to the emitter
std::vector<NormTileDesc> NormEmitter::HeuristicFilter(const std::vector<NormTileDesc>& norm_tile_desc)
{
    std::vector<NormTileDesc> filter_tile_desc;
    for (const auto& tile_desc : norm_tile_desc) {
        // Hard code to select the tile descriptor
        if (norm_problem.m == 64 && norm_problem.n == 64 && tile_desc.m == 1 && tile_desc.n == 1 && tile_desc.k == 8
            && tile_desc.h == 8 && tile_desc.w == 8) {
            filter_tile_desc.push_back(tile_desc);
        }
        else {
            LOG(WARNING) << "Tile descriptor does not match the heuristic condition: " << tile_desc.ToString();
        }
    }
    return filter_tile_desc;
}

// Generate the operation for the tile descriptor
std::map<NormKind, std::map<std::string, NormCodeGen>> NormEmitter::GenerateInstances(const NormProblem& norm_problem)
{
    // filter invalid tile descriptor
    std::vector<NormTileDesc> valid_tile_desc;
    for (const auto& tile_desc : g_default_norm_tile_desc) {
        if (IsValidTile(tile_desc)) {
            valid_tile_desc.push_back(tile_desc);
        }
    }

    // generate the operation based on the valid tile descriptors according to the different strategies
    std::vector<NormTileDesc> filtered_tile_desc;
    if (FLAGS_FC_mode == 0) {
        // 1. Heuristic generate (filtered_tile_desc size must be 1)
        auto filtered_tile_desc = HeuristicFilter(valid_tile_desc);
        if (filtered_tile_desc.size() != 1) {
            LOG(ERROR) << "Heuristic generate failed, expected 1 tile descriptor, got " << filtered_tile_desc.size();
            return;
        }
    }
    else if (FLAGS_FC_mode == 1) {
        // 2. Autotuning
        filtered_tile_desc = valid_tile_desc;  // Use all valid tile descriptors for autotuning
    }
    else if (FLAGS_FC_mode == 2) {
        // 3. Heuristic(filtered_tile_desc size greated than 1) -> autotuning
        if (valid_tile_desc.size() == 1) {
            LOG(WARNING) << "Heuristic generate returned only one tile descriptor, continuing without autotuning.";
        }
        filtered_tile_desc = HeuristicFilter(valid_tile_desc);
    }
    else {
        FC_THROW("Unsupported mode: " + std::to_string(FLAGS_FC_mode));
    }

    // append the generated operation to the instance map
    for (const auto& tile_desc : filtered_tile_desc) {
        NormCodeGen norm;

        norm.kind = norm_problem.kind;

        norm.x_dtype_            = norm_problem.x_dtype;
        norm.y_dtype_            = norm_problem.y_dtype;
        norm.smooth_scale_dtype_ = norm_problem.smooth_scale_dtype_;
        norm.y_scale_dtype_      = norm_problem.y_scale_dtype_;

        norm.is_add_bias_ = norm_problem.is_add_bias_;
        norm.fused_add_   = norm_problem.fused_add_;
        norm.fused_quant_ = norm_problem.fused_quant_;

        instance_map_[norm.kind][norm.GetConfigName()] = norm;
        num_instances_++;
    }
    LOG(INFO) << "Generated " << filtered_tile_desc.size() << " Norm operations.";
}

int64_t NormEmitter::GetNumInstances()
{
    return num_instances_;
}

}  // namespace flashck