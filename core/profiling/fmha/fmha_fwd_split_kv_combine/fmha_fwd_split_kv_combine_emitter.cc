#include "core/profiling/fmha/fmha_fwd_split_kv_combine/fmha_fwd_split_kv_combine_emitter.h"

#include <algorithm>
#include <filesystem>
#include <random>

FC_DECLARE_int32(FC_TUNING_MODE);    // 0: heuristic, 1: autotuning, 2: hybrid
FC_DECLARE_bool(FC_ENABLE_BACKUP_JSON);   // Enable backup_config.json loading
FC_DECLARE_bool(FC_ENABLE_DEFAULT_JSON);  // Enable default_config.json loading  
FC_DECLARE_bool(FC_ENABLE_USER_JSON);     // Enable user_config.json loading
FC_DECLARE_string(FC_CONFIG_JSON_PATH);   // Base path for config files

namespace flashck {

bool FmhaFwdSplitKVCombineEmitter::IsValidTile(const FmhaFwdSplitKVCombineTileDesc& tile_desc, const FmhaProblem& fmha_problem)
{
    // Validate all tile parameters are positive for split-KV combine operations
    if (tile_desc.m0_block_ <= 0 || tile_desc.n0_block_ <= 0 || tile_desc.k0_block_ <= 0 || 
        tile_desc.k0_max_block_ <= 0 || tile_desc.n1_block_ <= 0 || tile_desc.k1_block_ <= 0 ||
        tile_desc.m0_warp_ <= 0 || tile_desc.n0_warp_ <= 0 || tile_desc.k0_warp_ < 0 ||
        tile_desc.m1_warp_ <= 0 || tile_desc.n1_warp_ <= 0 || tile_desc.k1_warp_ < 0 ||
        tile_desc.m0_warp_tile_ <= 0 || tile_desc.n0_warp_tile_ <= 0 || tile_desc.k0_warp_tile_ <= 0 ||
        tile_desc.m1_warp_tile_ <= 0 || tile_desc.n1_warp_tile_ <= 0 || tile_desc.k1_warp_tile_ <= 0) {
        VLOG(3) << "Invalid FMHA split-KV combine tile: negative or zero values not allowed";
        return false;
    }

    // Validate k0_block_ <= k0_max_block_ for split-KV combine constraints
    if (tile_desc.k0_block_ > tile_desc.k0_max_block_) {
        VLOG(3) << "Invalid FMHA split-KV combine tile: k0_block_ > k0_max_block_";
        return false;
    }

    // Validate warp*warp_tile <= block sizes for split-KV combine operations
    if (tile_desc.m0_warp_ * tile_desc.m0_warp_tile_ > tile_desc.m0_block_ ||
        tile_desc.n0_warp_ * tile_desc.n0_warp_tile_ > tile_desc.n0_block_ ||
        tile_desc.k0_warp_ * tile_desc.k0_warp_tile_ > tile_desc.k0_block_ ||
        tile_desc.m1_warp_ * tile_desc.m1_warp_tile_ > tile_desc.m0_block_ ||
        tile_desc.n1_warp_ * tile_desc.n1_warp_tile_ > tile_desc.n1_block_ ||
        tile_desc.k1_warp_ * tile_desc.k1_warp_tile_ > tile_desc.k1_block_) {
        VLOG(3) << "Invalid FMHA split-KV combine tile: warp*warp_tile exceeds block size";
        return false;
    }

    // Validate block sizes are divisible by warp*warp_tile for split-KV combine
    if ((tile_desc.m0_block_ % (tile_desc.m0_warp_ * tile_desc.m0_warp_tile_) != 0) ||
        (tile_desc.n0_block_ % (tile_desc.n0_warp_ * tile_desc.n0_warp_tile_) != 0) ||
        (tile_desc.k0_block_ % (std::max<int64_t>(1, tile_desc.k0_warp_ * tile_desc.k0_warp_tile_)) != 0) ||
        (tile_desc.m0_block_ % (tile_desc.m1_warp_ * tile_desc.m1_warp_tile_) != 0) ||
        (tile_desc.n1_block_ % (tile_desc.n1_warp_ * tile_desc.n1_warp_tile_) != 0) ||
        (tile_desc.k1_block_ % (std::max<int64_t>(1, tile_desc.k1_warp_ * tile_desc.k1_warp_tile_)) != 0)) {
        VLOG(3) << "Invalid FMHA split-KV combine tile: block size not divisible by warp*warp_tile";
        return false;
    }

    // Validate against problem dimensions for Batch mode with split-KV combine considerations
    if (fmha_problem.mode_ == FmhaMode::Batch) {
        if (tile_desc.m0_block_ > fmha_problem.q_seq_len_ || 
            tile_desc.n0_block_ > fmha_problem.kv_seq_len_ ||
            tile_desc.n1_block_ > fmha_problem.v_head_dim_ || 
            tile_desc.k0_max_block_ > fmha_problem.qk_head_dim_) {
            VLOG(3) << "Invalid FMHA split-KV combine tile: dimensions exceed problem dimensions";
            return false;
        }
    }

    return true;
}

bool FmhaFwdSplitKVCombineEmitter::IsValidInstance(const FmhaFwdSplitKVCombineCodeGen& instance)
{
    return IsValidTile(instance.tile_desc_, instance.problem_);
}

std::vector<FmhaFwdSplitKVCombineCodeGen> FmhaFwdSplitKVCombineEmitter::HeuristicFilter(
    const std::vector<FmhaFwdSplitKVCombineCodeGen>& instances,
    const FmhaProblem& fmha_problem)
{
    if (instances.empty()) {
        return {};
    }

    std::vector<FmhaFwdSplitKVCombineCodeGen> filtered;
    
    // Split-KV combine specific heuristics for efficient result aggregation
    // Heuristic 1: Optimize for result combination bandwidth
    constexpr int64_t preferred_combine_block_size = 512;
    
    // Heuristic 2: Minimize reduction synchronization overhead
    constexpr int64_t max_optimal_warps = 8;
    
    // Heuristic 3: Balance memory access for combine operations
    const int64_t q_seq_len = fmha_problem.q_seq_len_;
    const int64_t optimal_reduction_granularity = std::max<int64_t>(32, q_seq_len / 16);
    
    for (const auto& instance : instances) {
        const auto& tile = instance.tile_desc_;
        
        // Filter 1: Skip configurations with poor combine efficiency
        if (tile.m0_block_ < 32 || tile.n1_block_ < 16) {
            continue;
        }
        
        // Filter 2: Prefer configurations optimized for result aggregation
        int64_t combine_efficiency_score = 0;
        
        // Good aggregation granularity for combine operations
        if (tile.m0_block_ <= optimal_reduction_granularity * 2 && 
            tile.m0_block_ >= optimal_reduction_granularity / 2) {
            combine_efficiency_score += 2; 
        }
        
        // Memory coalescing friendly for combine operations
        if (tile.n1_block_ % 32 == 0) {
            combine_efficiency_score += 1; 
        }
        
        // Optimal reduction patterns
        if (tile.m0_block_ >= 64 && tile.n1_block_ >= 64) {
            combine_efficiency_score += 1;
        }
        
        // Filter 3: Ensure reasonable warp utilization for combine operations
        int64_t total_warps = tile.m0_warp_ * tile.n0_warp_ * std::max<int64_t>(1, tile.k0_warp_);
        if (total_warps > max_optimal_warps || total_warps < 1) {
            continue;
        }
        
        // Filter 4: Split-KV combine specific memory access optimization
        int64_t combine_working_set = tile.m0_block_ * tile.n1_block_;
        if (combine_working_set < 512) {
            continue; // Too small for efficient combine operations
        }
        
        // Filter 5: Avoid excessive fragmentation in combine operations
        if (tile.m0_warp_tile_ > tile.m0_block_ / 4 || 
            tile.n1_warp_tile_ > tile.n1_block_ / 4) {
            continue;
        }
        
        // Filter 6: Prefer block sizes that facilitate efficient reduction
        bool good_reduction_size = (tile.m0_block_ % 64 == 0) || 
                                  (tile.m0_block_ >= preferred_combine_block_size);
        if (!good_reduction_size && combine_efficiency_score < 2) {
            continue;
        }
        
        // Only keep instances that pass split-KV combine specific criteria
        if (combine_efficiency_score >= 1) {
            filtered.push_back(instance);
        }
    }
    
    // If filtering is too aggressive, return a subset with relaxed criteria
    if (filtered.empty()) {
        VLOG(2) << "Split-KV combine heuristic filter too aggressive, returning subset of original instances";
        const size_t subset_size = std::min<size_t>(instances.size(), 10);
        filtered.assign(instances.begin(), instances.begin() + subset_size);
    }
    
    VLOG(2) << "Split-KV combine heuristic filter: " << instances.size() << " -> " << filtered.size() << " instances";
    return filtered;
}

std::vector<FmhaFwdSplitKVCombineCodeGen> FmhaFwdSplitKVCombineEmitter::CreateInstanceForConfig(
    const FmhaFwdSplitKVCombineConfig& config, const FmhaProblem& fmha_problem) 
{
    std::vector<FmhaFwdSplitKVCombineCodeGen> result;

    // Convert all config parameters to int64_t vectors for CartesianProduct
    std::vector<std::vector<int64_t>> all_param_lists = {
        // Block tile configuration (6 parameters)
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_tile.m0.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_tile.n0.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_tile.k0.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_tile.k0_max.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_tile.n1.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_tile.k1.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        
        // Block warp configuration (6 parameters)
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_warps.m0.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_warps.n0.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_warps.k0.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_warps.m1.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_warps.n1.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_warps.k1.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        
        // Warp tile configuration (6 parameters)
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.warp_tile.m0.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.warp_tile.n0.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.warp_tile.k0.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.warp_tile.m1.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.warp_tile.n1.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.warp_tile.k1.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        
        // Padding configuration (4 parameters, bool->int64_t)
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.padding.s.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.padding.sk.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.padding.d.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.padding.dv.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        
        // Launch configuration (1 parameter)
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.launch.min_block_per_cu.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        
        // Pipeline configuration (1 parameter, string->enum->int64_t)
        [&]{ std::vector<int64_t> v; 
             for (const auto& x : config.pipeline.values) 
                 v.push_back(static_cast<int64_t>(StrToBlockFmhaPipelineEnum(x))); 
             return v; }(),
    };

    // Generate all parameter combinations using CartesianProduct
    CartesianProduct(all_param_lists, [&](const std::vector<int64_t>& param_values) {
        size_t idx = 0;
        
        // Extract block tile parameters
        int64_t m0_block = param_values[idx++];
        int64_t n0_block = param_values[idx++];
        int64_t k0_block = param_values[idx++];
        int64_t k0_max_block = param_values[idx++];
        int64_t n1_block = param_values[idx++];
        int64_t k1_block = param_values[idx++];
        
        // Extract block warp parameters
        int64_t m0_warp = param_values[idx++];
        int64_t n0_warp = param_values[idx++];
        int64_t k0_warp = param_values[idx++];
        int64_t m1_warp = param_values[idx++];
        int64_t n1_warp = param_values[idx++];
        int64_t k1_warp = param_values[idx++];
        
        // Extract warp tile parameters
        int64_t m0_warp_tile = param_values[idx++];
        int64_t n0_warp_tile = param_values[idx++];
        int64_t k0_warp_tile = param_values[idx++];
        int64_t m1_warp_tile = param_values[idx++];
        int64_t n1_warp_tile = param_values[idx++];
        int64_t k1_warp_tile = param_values[idx++];
        
        // Extract padding parameters
        bool is_pad_q_seq_len = static_cast<bool>(param_values[idx++]);
        bool is_pad_kv_seq_len = static_cast<bool>(param_values[idx++]);
        bool is_pad_qk_head_dim = static_cast<bool>(param_values[idx++]);
        bool is_pad_v_head_dim = static_cast<bool>(param_values[idx++]);
        
        // Extract launch parameters
        int64_t min_block_per_cu = param_values[idx++];
        
        // Extract pipeline parameters
        BlockFmhaPipelineEnum pipeline = static_cast<BlockFmhaPipelineEnum>(param_values[idx++]);

        // Construct FmhaFwdSplitKVCombineCodeGen instance
        FmhaFwdSplitKVCombineCodeGen instance;
        instance.problem_ = fmha_problem;
        
        // Set tile descriptor
        instance.tile_desc_.m0_block_ = m0_block;
        instance.tile_desc_.n0_block_ = n0_block;
        instance.tile_desc_.k0_block_ = k0_block;
        instance.tile_desc_.k0_max_block_ = k0_max_block;
        instance.tile_desc_.n1_block_ = n1_block;
        instance.tile_desc_.k1_block_ = k1_block;
        instance.tile_desc_.m0_warp_ = m0_warp;
        instance.tile_desc_.n0_warp_ = n0_warp;
        instance.tile_desc_.k0_warp_ = k0_warp;
        instance.tile_desc_.m1_warp_ = m1_warp;
        instance.tile_desc_.n1_warp_ = n1_warp;
        instance.tile_desc_.k1_warp_ = k1_warp;
        instance.tile_desc_.m0_warp_tile_ = m0_warp_tile;
        instance.tile_desc_.n0_warp_tile_ = n0_warp_tile;
        instance.tile_desc_.k0_warp_tile_ = k0_warp_tile;
        instance.tile_desc_.m1_warp_tile_ = m1_warp_tile;
        instance.tile_desc_.n1_warp_tile_ = n1_warp_tile;
        instance.tile_desc_.k1_warp_tile_ = k1_warp_tile;
        
        // Set padding configuration
        instance.is_pad_q_seq_len_ = is_pad_q_seq_len;
        instance.is_pad_kv_seq_len_ = is_pad_kv_seq_len;
        instance.is_pad_qk_head_dim_ = is_pad_qk_head_dim;
        instance.is_pad_v_head_dim_ = is_pad_v_head_dim;
        
        // Set launch configuration
        instance.min_block_per_cu_ = min_block_per_cu;
        
        // Set pipeline configuration
        instance.pipeline_ = pipeline;
        
        result.push_back(instance);
    });

    return result;
}

void FmhaFwdSplitKVCombineEmitter::GenerateInstances(FmhaProblem& fmha_problem)
{
    // Validate tuning mode
    FC_ENFORCE_EQ(FLAGS_FC_TUNING_MODE >= 0 && FLAGS_FC_TUNING_MODE <= 2, true,
                  Unavailable("Invalid tuning mode: {}, valid modes are 0 (heuristic), 1 (autotuning), 2 (hybrid)", 
                              FLAGS_FC_TUNING_MODE));

    // Check if instances already exist for this FMHA kind
    if (instance_map_.find(fmha_problem.kind_) != instance_map_.end() && 
        !instance_map_[fmha_problem.kind_].empty()) {
        VLOG(2) << "Split-KV combine instances already generated for FMHA kind: " << GetFmhaKindName(fmha_problem.kind_);
        return;
    }

    VLOG(1) << "Generating FMHA split-KV combine instances for mode: " << FLAGS_FC_TUNING_MODE;

    // Load configurations from JSON files
    auto base_json_path = std::filesystem::path(FLAGS_FC_CONFIG_JSON_PATH) / GetFmhaKindName(fmha_problem.kind_);
    std::vector<FmhaFwdSplitKVCombineCodeGen> all_instances;
    
    // Load backup configurations (pre-validated, single-value configs)
    if (FLAGS_FC_ENABLE_BACKUP_JSON) {
        std::filesystem::path json_path = base_json_path / "backup_config.json";
        try {
            auto backup_configs = LoadConfigJson<std::vector<FmhaFwdSplitKVCombineConfig>>(json_path);
            for (const auto& config : backup_configs) {
                auto instances = CreateInstanceForConfig(config, fmha_problem);
                all_instances.insert(all_instances.end(), instances.begin(), instances.end());
            }
            VLOG(2) << "Loaded " << backup_configs.size() << " split-KV combine backup configurations";
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load split-KV combine backup config: " << e.what();
        }
    }

    // Load default configurations (parameter ranges for tuning)
    if (FLAGS_FC_ENABLE_DEFAULT_JSON) {
        std::filesystem::path json_path = base_json_path / "default_config.json";
        try {
            auto default_config = LoadConfigJson<FmhaFwdSplitKVCombineConfig>(json_path);
            auto instances = CreateInstanceForConfig(default_config, fmha_problem);
            all_instances.insert(all_instances.end(), instances.begin(), instances.end());
            VLOG(2) << "Loaded split-KV combine default configuration with " << instances.size() << " instances";
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load split-KV combine default config: " << e.what();
        }
    }

    // Load user configurations (custom user-defined configs)
    if (FLAGS_FC_ENABLE_USER_JSON) {
        std::filesystem::path json_path = base_json_path / "user_config.json";
        try {
            auto user_config = LoadConfigJson<FmhaFwdSplitKVCombineConfig>(json_path);
            auto instances = CreateInstanceForConfig(user_config, fmha_problem);
            all_instances.insert(all_instances.end(), instances.begin(), instances.end());
            VLOG(2) << "Loaded split-KV combine user configuration with " << instances.size() << " instances";
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load split-KV combine user config: " << e.what();
        }
    }

    // Filter out invalid instances
    std::vector<FmhaFwdSplitKVCombineCodeGen> valid_instances;
    for (const auto& instance : all_instances) {
        if (IsValidInstance(instance)) {
            valid_instances.push_back(instance);
        }
    }

    VLOG(2) << "Split-KV combine validation: " << all_instances.size() << " -> " << valid_instances.size() << " valid instances";

    if (valid_instances.empty()) {
        FC_THROW(Unavailable("No valid FMHA split-KV combine instances found"));
    }

    // Apply mode-specific strategy
    std::vector<FmhaFwdSplitKVCombineCodeGen> final_instances;
    std::random_device rd;
    std::mt19937 rng(rd());
    
    switch (FLAGS_FC_TUNING_MODE) {
        case 0: {
            // Heuristic mode: filter + random selection
            final_instances = HeuristicFilter(valid_instances, fmha_problem);
            if (!final_instances.empty()) {
                // Randomly select one instance for fast execution
                std::uniform_int_distribution<> dist(0, final_instances.size() - 1);
                auto selected = final_instances[dist(rng)];
                final_instances = {selected};
                VLOG(1) << "Split-KV combine heuristic mode: selected 1 instance from " << valid_instances.size();
            }
            break;
        }
        case 1: {
            // Autotuning mode: use all valid instances
            final_instances = valid_instances;
            VLOG(1) << "Split-KV combine autotuning mode: using all " << final_instances.size() << " valid instances";
            break;
        }
        case 2: {
            // Hybrid mode: combine heuristic filtering + all instances
            auto heuristic_instances = HeuristicFilter(valid_instances, fmha_problem);
            final_instances = heuristic_instances;
            final_instances.insert(final_instances.end(), valid_instances.begin(), valid_instances.end());
            
            // Remove duplicates
            std::sort(final_instances.begin(), final_instances.end(), 
                     [](const auto& a, const auto& b) {
                         return a.GetInstanceName() < b.GetInstanceName();
                     });
            final_instances.erase(std::unique(final_instances.begin(), final_instances.end(),
                                            [](const auto& a, const auto& b) {
                                                return a.GetInstanceName() == b.GetInstanceName();
                                            }), final_instances.end());
            
            VLOG(1) << "Split-KV combine hybrid mode: using " << final_instances.size() << " unique instances";
            break;
        }
        default:
            FC_THROW(Unavailable("Invalid tuning mode: {}", FLAGS_FC_TUNING_MODE));
    }

    if (final_instances.empty()) {
        FC_THROW(Unavailable("No final FMHA split-KV combine instances after mode-specific filtering"));
    }

    // Store instances in the map
    auto& kind_instance_map = instance_map_[fmha_problem.kind_];
    int64_t generated_count = 0;

    for (const auto& instance : final_instances) {
        try {
            std::string instance_name = instance.GetInstanceName();
            
            // Avoid duplicates
            if (kind_instance_map.find(instance_name) == kind_instance_map.end()) {
                kind_instance_map[instance_name] = instance;
                generated_count++;
                VLOG(3) << "Generated FMHA split-KV combine instance: " << instance_name;
            } else {
                VLOG(3) << "Skipped duplicate FMHA split-KV combine instance: " << instance_name;
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to create FMHA split-KV combine codegen for instance: " << instance.GetInstanceName()
                         << ", error: " << e.what();
        }
    }

    num_instances_ += generated_count;
    VLOG(1) << "Generated " << generated_count << " FMHA split-KV combine instances for kind: " 
            << GetFmhaKindName(fmha_problem.kind_) << " (total: " << num_instances_ << ")";
}

void FmhaFwdSplitKVCombineEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
    VLOG(2) << "Cleared all FMHA split-KV combine instances";
}

}  // namespace flashck
