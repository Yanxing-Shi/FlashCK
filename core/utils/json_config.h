#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <nlohmann/json.hpp>

namespace flashck {

using json = nlohmann::json;

// ========== Common Parameter Structs ==========

struct IntEnumConfigParam {
    json data;  // Store raw JSON data to support both modes
    
    // Get all values, supporting both "values" and "min/max/step/exclude" modes
    std::vector<int> GetAllValues() const {
        if (data.contains("values") && data["values"].is_array()) {
            return data["values"].get<std::vector<int>>();
        }
        
        // Generate from min/max/step/exclude
        std::vector<int> result;
        if (data.contains("min") && data.contains("max")) {
            int min_val = data["min"].get<int>();
            int max_val = data["max"].get<int>();
            int step_val = data.contains("step") ? data["step"].get<int>() : 1;
            std::vector<int> exclude_vals;
            if (data.contains("exclude")) {
                exclude_vals = data["exclude"].get<std::vector<int>>();
            }
            
            for (int v = min_val; v <= max_val; v += step_val) {
                if (std::find(exclude_vals.begin(), exclude_vals.end(), v) == exclude_vals.end()) {
                    result.push_back(v);
                }
            }
        }
        return result;
    }
    
    // Legacy compatibility - access values directly
    std::vector<int> values;
    
    // Constructor to maintain backward compatibility
    IntEnumConfigParam() = default;
    IntEnumConfigParam(const std::vector<int>& vals) : values(vals) {
        data["values"] = vals;
    }
};

// Custom JSON serialization for IntEnumConfigParam
inline void to_json(json& j, const IntEnumConfigParam& p) {
    j = p.data;
}

inline void from_json(const json& j, IntEnumConfigParam& p) {
    p.data = j;
    // For backward compatibility, populate values if present
    if (j.contains("values")) {
        p.GetAllValues() = j["values"].get<std::vector<int>>();
    } else {
        p.GetAllValues() = p.GetAllValues();
    }
}

struct BoolEnumConfigParam {
    json data;  // Store raw JSON data to support both modes
    
    // Get all values, supporting both "values" and "min/max/exclude" modes
    std::vector<bool> GetAllValues() const {
        if (data.contains("values") && data["values"].is_array()) {
            return data["values"].get<std::vector<bool>>();
        }
        
        // Generate from min/max/exclude (for bool, typically just [false, true])
        std::vector<bool> result;
        if (data.contains("min") && data.contains("max")) {
            bool min_val = data["min"].get<bool>();
            bool max_val = data["max"].get<bool>();
            std::vector<bool> exclude_vals;
            if (data.contains("exclude")) {
                exclude_vals = data["exclude"].get<std::vector<bool>>();
            }
            
            std::vector<bool> candidates = {false, true};
            for (bool v : candidates) {
                if (v >= min_val && v <= max_val) {
                    if (std::find(exclude_vals.begin(), exclude_vals.end(), v) == exclude_vals.end()) {
                        result.push_back(v);
                    }
                }
            }
        }
        return result;
    }
    
    // Legacy compatibility - access values directly
    std::vector<bool> values;
    
    // Constructor to maintain backward compatibility
    BoolEnumConfigParam() = default;
    BoolEnumConfigParam(const std::vector<bool>& vals) : values(vals) {
        data["values"] = vals;
    }
};

// Custom JSON serialization for BoolEnumConfigParam
inline void to_json(json& j, const BoolEnumConfigParam& p) {
    j = p.data;
}

inline void from_json(const json& j, BoolEnumConfigParam& p) {
    p.data = j;
    // For backward compatibility, populate values if present
    if (j.contains("values")) {
        p.GetAllValues() = j["values"].get<std::vector<bool>>();
    } else {
        p.GetAllValues() = p.GetAllValues();
    }
}

struct StrEnumConfigParam {
    json data;  // Store raw JSON data to support both modes
    
    // Get all values, supporting both "values" and "exclude" modes
    std::vector<std::string> GetAllValues() const {
        if (data.contains("values") && data["values"].is_array()) {
            return data["values"].get<std::vector<std::string>>();
        }
        
        // For strings, min/max/step typically not meaningful, so just return empty
        // Could be extended if needed for specific use cases
        return {};
    }
    
    // Legacy compatibility - access values directly
    std::vector<std::string> values;
    
    // Constructor to maintain backward compatibility
    StrEnumConfigParam() = default;
    StrEnumConfigParam(const std::vector<std::string>& vals) : values(vals) {
        data["values"] = vals;
    }
};

// Custom JSON serialization for StrEnumConfigParam
inline void to_json(json& j, const StrEnumConfigParam& p) {
    j = p.data;
}

inline void from_json(const json& j, StrEnumConfigParam& p) {
    p.data = j;
    // For backward compatibility, populate values if present
    if (j.contains("values")) {
        p.GetAllValues() = j["values"].get<std::vector<std::string>>();
    } else {
        p.GetAllValues() = p.GetAllValues();
    }
}


// ========== Tile-based GEMM Structs ==========
struct GemmBlockConfig {
    IntEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmBlockConfig, m, n, k)

struct GemmWarpConfig {
    IntEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmWarpConfig, m, n, k)

struct GemmWarpTileConfig {
    IntEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmWarpTileConfig, m, n, k)

struct GemmTileConfig {
    GemmBlockConfig block_tile;
    GemmWarpConfig block_warps;
    GemmWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmTileConfig, block_tile, block_warps, warp_tile)

struct GemmPaddingConfig {
    BoolEnumConfigParam m, n, k;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmPaddingConfig, m, n, k)

struct GemmTraitConfig {
    GemmPaddingConfig padding;
    IntEnumConfigParam num_wave_groups;
    BoolEnumConfigParam persistent;
    BoolEnumConfigParam preshuffle;
    BoolEnumConfigParam use_sparsity;
};

struct GemmStrategyConfig{
    StrEnumConfigParam pipeline, scheduler, epilogue;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmStrategyConfig, pipeline, scheduler, epilogue)

struct GemmPartitionConfig {
    IntEnumConfigParam tile_partitioner_group_num, tile_partitioner_m01;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmPartitionConfig, tile_partitioner_group_num, tile_partitioner_m01)

struct GemmLaunchConfig {
    IntEnumConfigParam max_thread_per_block;
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmLaunchConfig, max_thread_per_block, min_block_per_cu)

struct GemmConfig {
    GemmTileConfig tile_shape;
    GemmTraitConfig trait;
    GemmStrategyConfig strategy;
    GemmPartitionConfig partition;
    GemmLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GemmConfig, tile_shape, trait, strategy, partition, launch)

// ========== FMHA Fwd Structs ==========
struct FmhaFwdBlockConfig {
    IntEnumConfigParam m0, n0, k0, n1, k1, k0_max;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBlockConfig, m0, n0, k0, n1, k1, k0_max)

struct FmhaFwdWarpConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdWarpConfig, m0, n0, k0, m1, n1, k1)

struct FmhaFwdWarpTileConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdWarpTileConfig, m0, n0, k0, m1, n1, k1)

struct FmhaFwdTileConfig {
    FmhaFwdBlockConfig block_tile;
    FmhaFwdWarpConfig block_warps;
    FmhaFwdWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdTileConfig, block_tile, block_warps, warp_tile)

struct FmhaFwdPaddingConfig {
    BoolEnumConfigParam s, sk, d, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPaddingConfig, s, sk, d, dv)

struct FmhaFwdtraitConfig {
    FmhaFwdPaddingConfig padding;
    BoolEnumConfigParam skip_min_q_seq_len;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdtraitConfig, padding, skip_min_q_seq_len)

struct FmhaFwdStrategyConfig {
    StrEnumConfigParam pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdStrategyConfig, pipeline)

struct FmhaFwdLaunchConfig {
    IntEnumConfigParam max_thread_per_block;
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdLaunchConfig, max_thread_per_block, min_block_per_cu)

struct FmhaFwdConfig {
    FmhaFwdTileConfig tile_shape;
    FmhaFwdTraitConfig trait;
    FmhaFwdStrategyConfig strategy;
    FmhaFwdLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdConfig, tile_shape, trait, strategy, launch)

// ========== FMHA Fwd append kv Structs ==========
struct FmhaFwdAppendKVBlockConfig {
    IntEnumConfigParam s, sk, d, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVBlockConfig, s, sk, d, dv)

struct FmhaFwdAppendKVTileConfig {
    FmhaFwdAppendKVBlockConfig block_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVTileConfig, block_tile)

struct FmhaFwdAppendKVPaddingConfig {
    BoolEnumConfigParam s, sk, d, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVPaddingConfig, s, sk, d, dv)

struct FmhaFwdAppendKVtraitConfig {
    FmhaFwdAppendKVPaddingConfig padding;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVtraitConfig, padding)

struct FmhaFwdAppendKVLaunchConfig {
    IntEnumConfigParam max_thread_per_block;
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVLaunchConfig, max_thread_per_block, min_block_per_cu)

struct FmhaFwdAppendKVConfig {
    FmhaFwdAppendKVTileConfig tile_shape;
    FmhaFwdAppendKVtraitConfig trait;
    FmhaFwdAppendKVLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdAppendKVConfig, tile_shape, trait, launch)

// ========== FMHA Fwd split kv Structs ==========
struct FmhaFwdSplitKVBlockConfig {
    IntEnumConfigParam m0, n0, k0, n1, k1, k0_max;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVBlockConfig, m0, n0, k0, n1, k1, k0_max)

struct FmhaFwdSplitKVWarpConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVWarpConfig, m0, n0, k0, m1, n1, k1)

struct FmhaFwdSplitKVWarpTileConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVWarpTileConfig, m0, n0, k0, m1, n1, k1)
struct FmhaFwdSplitKVTileConfig {
    FmhaFwdSplitKVBlockConfig block_tile;
    FmhaFwdSplitKVWarpConfig block_warps;
    FmhaFwdSplitKVWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVTileConfig, block_tile, block_warps, warp_tile)

struct FmhaFwdSplitKVPaddingConfig {
    BoolEnumConfigParam s, sk, d, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVPaddingConfig, s, sk, d, dv)

struct FmhaFwdSplitKVTraitConfig {
    FmhaFwdSplitKVPaddingConfig padding;
    BoolEnumConfigParam has_uneven_splits;
    BoolEnumConfigParam merge_groups_num_head_q_seq_len;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVTraitConfig, padding, has_uneven_splits, merge_groups_num_head_q_seq_len)

struct FmhaFwdSplitKVStrategyConfig {
    StrEnumConfigParam pipeline;
    IntEnumConfigParam num_splits;
};

struct FmhaFwdSplitKVLaunchConfig {
    IntEnumConfigParam max_thread_per_block;
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVLaunchConfig, max_thread_per_block, min_block_per_cu)

struct FmhaFwdSplitKVConfig {
    FmhaFwdSplitKVTileConfig tile_shape;
    FmhaFwdSplitKVTraitConfig trait;
    FmhaFwdSplitKVStrategyConfig strategy;
    FmhaFwdSplitKVLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVConfig, tile_shape, trait, strategy, launch)

// ========== FMHA Fwd split kv combine Structs ==========
struct FmhaFwdSplitKVCombineBlockConfig {
    IntEnumConfigParam n1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineBlockConfig, n1)

struct FmhaFwdSplitKVCombineTileConfig {
    FmhaFwdSplitKVCombineBlockConfig block_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineTileConfig, block_tile)

struct FmhaFwdSplitKVCombinePaddingConfig {
    BoolEnumConfigParam s, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombinePaddingConfig, s, dv)

struct FmhaFwdSplitKVCombineTraitConfig {
    FmhaFwdSplitKVCombinePaddingConfig padding;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineTraitConfig, padding)

struct FmhaFwdSplitKVCombineLaunchConfig {
    IntEnumConfigParam max_thread_per_block;
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineLaunchConfig, min_block_per_cu)

struct FmhaFwdSplitKVCombineConfig {
    FmhaFwdSplitKVCombineTileConfig tile_shape;
    FmhaFwdSplitKVCombineTraitConfig trait;
    FmhaFwdSplitKVCombineLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdSplitKVCombineConfig, tile_shape, trait, launch)

// ========== FMHA batch prefill Structs ==========
struct FmhaFwdBatchPrefillBlockConfig {
    IntEnumConfigParam m0, n0, k0, n1, k1, k0_max;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBatchPrefillBlockConfig, m0, n0, k0, n1, k1, k0_max)

struct FmhaFwdBatchPrefillWarpConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBatchPrefillWarpConfig, m0, n0, k0, m1, n1, k1)

struct FmhaFwdBatchPrefillWarpTileConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBatchPrefillWarpTileConfig, m0, n0, k0, m1, n1, k1)

struct FmhaFwdBatchPrefillTileConfig {
    FmhaFwdBatchPrefillBlockConfig block_tile;
    FmhaFwdBatchPrefillWarpConfig block_warps;
    FmhaFwdBatchPrefillWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBatchPrefillTileConfig, block_tile, block_warps, warp_tile)

struct FmhaFwdBatchPrefillPaddingConfig {
    BoolEnumConfigParam s, sk, d, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBatchPrefillPaddingConfig, s, sk, d, dv)

struct FmhaFwdBatchPrefillTraitConfig {
    FmhaFwdBatchPrefillPaddingConfig padding;
    BoolEnumConfigParam skip_min_q_seq_len;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBatchPrefillTraitConfig, padding, skip_min_q_seq_len)
struct FmhaFwdBatchPrefillStrategyConfig {
    StrEnumConfigParam pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBatchPrefillStrategyConfig, pipeline)

struct FmhaFwdBatchPrefillLaunchConfig {
    IntEnumConfigParam max_thread_per_block;
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBatchPrefillLaunchConfig, min_block_per_cu)


struct FmhaFwdBatchPrefillConfig {
    FmhaFwdBatchPrefillTileConfig tile_shape;
    FmhaFwdBatchPrefillTraitConfig trait;
    FmhaFwdBatchPrefillStrategyConfig strategy;
    FmhaFwdBatchPrefillLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdBatchPrefillConfig, tile_shape, trait, strategy, launch)

// ========== FMHA paged kv prefill Structs ==========
struct FmhaFwdPagedKVPrefillBlockConfig {
    IntEnumConfigParam m0, n0, k0, n1, k1, k0_max;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPagedKVPrefillBlockConfig, m0, n0, k0, n1, k1, k0_max)

struct FmhaFwdPagedKVPrefillWarpConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPagedKVPrefillWarpConfig, m0, n0, k0, m1, n1, k1)

struct FmhaFwdPagedKVPrefillWarpTileConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPagedKVPrefillWarpTileConfig, m0, n0, k0, m1, n1, k1)

struct FmhaFwdPagedKVPrefillTileConfig {
    FmhaFwdPagedKVPrefillBlockConfig block_tile;
    FmhaFwdPagedKVPrefillWarpConfig block_warps;
    FmhaFwdPagedKVPrefillWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPagedKVPrefillTileConfig, block_tile, block_warps, warp_tile)

struct FmhaFwdPagedKVPrefillPaddingConfig {
    BoolEnumConfigParam s, sk, d, dv;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPagedKVPrefillPaddingConfig, s, sk, d, dv)

struct FmhaFwdPagedKVPrefillTraitConfig {
    FmhaFwdPagedKVPrefillPaddingConfig padding;
    BoolEnumConfigParam skip_min_q_seq_len;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPagedKVPrefillTraitConfig, padding, skip_min_q_seq_len)

struct FmhaFwdPagedKVPrefillStrategyConfig {
    StrEnumConfigParam pipeline;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPagedKVPrefillStrategyConfig, pipeline)


struct FmhaFwdPagedKVPrefillLaunchConfig {
    IntEnumConfigParam max_thread_per_block;
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPagedKVPrefillLaunchConfig, max_thread_per_block, min_block_per_cu)


struct FmhaFwdPagedKVPrefillConfig {
    FmhaFwdPagedKVPrefillTileConfig tile_shape;
    FmhaFwdPagedKVPrefillTraitConfig trait;
    FmhaFwdPagedKVPrefillStrategyConfig strategy;
    FmhaFwdPagedKVPrefillLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FmhaFwdPagedKVPrefillConfig, tile_shape, trait, strategy, launch)

// ========== Norm Structs ==========
struct NormTileConfig {
    IntEnumConfigParam m_repeat, n_repeat, m_thread_per_block, n_thread_per_block, n_vector;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NormTileConfig, m_repeat, n_repeat, m_thread_per_block, n_thread_per_block, n_vector)

struct NormPaddingConfig {
    BoolEnumConfigParam n;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NormPaddingConfig, n)

struct NormTraitConfig {
    NormPaddingConfig padding;
    BoolEnumConfigParam is_two_pass;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NormTraitConfig, padding, is_two_pass)

struct NormLaunchConfig {
    IntEnumConfigParam max_thread_per_block;
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NormLaunchConfig, max_thread_per_block, min_block_per_cu)

struct NormConfig {
    NormTileConfig tile_shape;
    NormTraitConfig trait;
    NormLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NormConfig, tile_shape, trait, launch)

// ========== Moe Structs ==========
struct MoeGemmBlockConfig {
    IntEnumConfigParam token;
    IntEnumConfigParam intermediate;
    IntEnumConfigParam hidden;
    IntEnumConfigParam down;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeGemmBlockConfig, token, intermediate, hidden, down)

struct MoeGemmWarpConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeGemmWarpConfig, m0, n0, k0, m1, n1, k1)

struct MoeGemmWarpTileConfig {
    IntEnumConfigParam m0, n0, k0, m1, n1, k1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeGemmWarpTileConfig, m0, n0, k0, m1, n1, k1)

struct MoeGemmTileConfig {
    MoeGemmBlockConfig block_tile;
    MoeGemmWarpConfig block_warps;
    MoeGemmWarpTileConfig warp_tile;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeGemmTileConfig, block_tile, block_warps, warp_tile)

struct MoeGemmPaddingConfig {
    BoolEnumConfigParam hidden_size;
    BoolEnumConfigParam intermediate_size;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeGemmPaddingConfig, hidden_size, intermediate_size)

struct MoeGemmTraitConfig {
    BoolEnumConfigParam interleave;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeGemmTraitConfig, interleave)

struct MoeLaunchConfig {
    IntEnumConfigParam max_thread_per_block;
    IntEnumConfigParam min_block_per_cu;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeLaunchConfig, max_thread_per_block, min_block_per_cu)

struct MoeGemmConfig {
    MoeGemmTileConfig tile_shape;
    MoeGemmTraitConfig trait;
    MoeLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeGemmConfig, tile_shape, trait, launch);

struct MoeSmoothQuantTileConfig {
    IntEnumConfigParam m_repeat, n_repeat, m_thread_per_block, n_thread_per_block, n_vector;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeSmoothQuantTileConfig, m_repeat, n_repeat, m_thread_per_block, n_thread_per_block, n_vector)

struct MoeSmoothQuantPaddingConfig {
    BoolEnumConfigParam n;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeSmoothQuantPaddingConfig, n)

struct MoeSmoothQuantTraitConfig {
    MoeSmoothQuantPaddingConfig padding;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeSmoothQuantTraitConfig, padding)


struct MoeSmoothQuantStrategyConfig {
    BoolEnumConfigParam is_two_pass;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeSmoothQuantStrategyConfig, is_two_pass)

struct MoeSmoothQuantConfig {
    MoeSmoothQuantTileConfig tile_shape;
    MoeSmoothQuantTraitConfig trait;
    MoeSmoothQuantStrategyConfig strategy;
    MoeLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeSmoothQuantConfig, tile_shape, trait, strategy, launch)

struct MoeSortingConfig {
    IntEnumConfigParam internal_load_unroll;
    IntEnumConfigParam expert_tile;
    MoeLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoeSortingConfig, internal_load_unroll, expert_tile, launch);

struct TopKSoftmaxConfig {
    IntEnumConfigParam issues_per_col;
    IntEnumConfigParam launch_type;
    IntEnumConfigParam bytes_per_issue;
    IntEnumConfigParam block_size;
    MoeLaunchConfig launch;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TopKSoftmaxConfig, issues_per_col, launch_type, bytes_per_issue, block_size, launch);

// ========== Config Loader ==========
// Generic loader for config (single or array)
template <typename T>
inline T LoadConfigJson(const std::string& path) {
    std::ifstream f(path);
    json j;
    f >> j;
    return j.get<T>();
}

inline void CartesianProduct(
    const std::vector<std::vector<int64_t>>& value_lists,
    std::function<void(const std::vector<int64_t>&)> callback)
{
    std::vector<int64_t> current;
    std::function<void(size_t)> recurse = [&](size_t depth) {
        if (depth == value_lists.size()) {
            callback(current);
            return;
        }
        for (const auto& v : value_lists[depth]) {
            current.push_back(v);
            recurse(depth + 1);
            current.pop_back();
        }
    };
    recurse(0);
}

} // namespace flashck
