#include "core/profiling/fmha/fmha_batch_prefill/fmha_batch_prefill_emitter.h"

#include <algorithm>
#include <filesystem>
#include <random>

FC_DECLARE_int32(FC_TUNING_MODE);    // 0: heuristic, 1: autotuning, 2: hybrid
FC_DECLARE_bool(FC_ENABLE_BACKUP_JSON);   // Enable backup_config.json loading
FC_DECLARE_bool(FC_ENABLE_DEFAULT_JSON);  // Enable default_config.json loading  
FC_DECLARE_bool(FC_ENABLE_USER_JSON);     // Enable user_config.json loading
FC_DECLARE_string(FC_CONFIG_JSON_PATH);   // Base path for config files
FC_DECLARE_int32(FC_ENABLE_JSON_MODE);    // JSON configuration mode

namespace flashck {

bool FmhaBatchPrefillEmitter::IsValidTile(const FmhaBatchPrefillTileDesc& tile_desc, const FmhaProblem& fmha_problem)
{
    // Validate all tile parameters are positive
    if (tile_desc.m0_block_ <= 0 || tile_desc.n0_block_ <= 0 || tile_desc.k0_block_ <= 0 || 
        tile_desc.k0_max_block_ <= 0 || tile_desc.n1_block_ <= 0 || tile_desc.k1_block_ <= 0 ||
        tile_desc.m0_warp_ <= 0 || tile_desc.n0_warp_ <= 0 || tile_desc.k0_warp_ < 0 ||
        tile_desc.m1_warp_ <= 0 || tile_desc.n1_warp_ <= 0 || tile_desc.k1_warp_ < 0 ||
        tile_desc.m0_warp_tile_ <= 0 || tile_desc.n0_warp_tile_ <= 0 || tile_desc.k0_warp_tile_ <= 0 ||
        tile_desc.m1_warp_tile_ <= 0 || tile_desc.n1_warp_tile_ <= 0 || tile_desc.k1_warp_tile_ <= 0) {
        VLOG(3) << "Invalid FMHA tile: negative or zero values not allowed";
        return false;
    }

    // Validate k0_block_ <= k0_max_block_
    if (tile_desc.k0_block_ > tile_desc.k0_max_block_) {
        VLOG(3) << "Invalid FMHA tile: k0_block_ > k0_max_block_";
        return false;
    }

    // Validate warp*warp_tile <= block sizes
    if (tile_desc.m0_warp_ * tile_desc.m0_warp_tile_ > tile_desc.m0_block_ ||
        tile_desc.n0_warp_ * tile_desc.n0_warp_tile_ > tile_desc.n0_block_ ||
        tile_desc.k0_warp_ * tile_desc.k0_warp_tile_ > tile_desc.k0_block_ ||
        tile_desc.m1_warp_ * tile_desc.m1_warp_tile_ > tile_desc.m0_block_ ||
        tile_desc.n1_warp_ * tile_desc.n1_warp_tile_ > tile_desc.n1_block_ ||
        tile_desc.k1_warp_ * tile_desc.k1_warp_tile_ > tile_desc.k1_block_) {
        VLOG(3) << "Invalid FMHA tile: warp*warp_tile exceeds block size";
        return false;
    }

    // Validate block sizes are divisible by warp*warp_tile
    if ((tile_desc.m0_block_ % (tile_desc.m0_warp_ * tile_desc.m0_warp_tile_) != 0) ||
        (tile_desc.n0_block_ % (tile_desc.n0_warp_ * tile_desc.n0_warp_tile_) != 0) ||
        (tile_desc.k0_block_ % (std::max<int64_t>(1, tile_desc.k0_warp_ * tile_desc.k0_warp_tile_)) != 0) ||
        (tile_desc.m0_block_ % (tile_desc.m1_warp_ * tile_desc.m1_warp_tile_) != 0) ||
        (tile_desc.n1_block_ % (tile_desc.n1_warp_ * tile_desc.n1_warp_tile_) != 0) ||
        (tile_desc.k1_block_ % (std::max<int64_t>(1, tile_desc.k1_warp_ * tile_desc.k1_warp_tile_)) != 0)) {
        VLOG(3) << "Invalid FMHA tile: block size not divisible by warp*warp_tile";
        return false;
    }

    // Validate against problem dimensions for Batch mode
    if (fmha_problem.mode_ == FmhaMode::Batch) {
        if (tile_desc.m0_block_ > fmha_problem.q_seq_len_ || 
            tile_desc.n0_block_ > fmha_problem.kv_seq_len_ ||
            tile_desc.n1_block_ > fmha_problem.v_head_dim_ || 
            tile_desc.k0_max_block_ > fmha_problem.qk_head_dim_) {
            VLOG(3) << "Invalid FMHA tile: dimensions exceed problem dimensions";
            return false;
        }
    }

    return true;
}

bool FmhaBatchPrefillEmitter::IsValidInstance(const FmhaBatchPrefillCodeGen& instance)
{
    return IsValidTile(instance.tile_desc_, instance.problem_);
}

std::vector<FmhaBatchPrefillCodeGen> FmhaBatchPrefillEmitter::HeuristicFilter(
    const std::vector<FmhaBatchPrefillCodeGen>& instances,
    const FmhaProblem& fmha_problem) const
{
    if (instances.empty()) {
        return {};
    }

    std::vector<FmhaBatchPrefillCodeGen> filtered;
    
    // Heuristic 1: Prefer larger block sizes for better memory efficiency
    constexpr int64_t preferred_min_block_size = 128;
    
    // Heuristic 2: Prefer balanced warp configurations
    constexpr int64_t preferred_warp_count = 4;
    
    // Heuristic 3: Filter based on problem size ratios
    const int64_t seq_len_ratio = fmha_problem.q_seq_len_ / fmha_problem.kv_seq_len_;
    
    for (const auto& instance : instances) {
        const auto& tile = instance.tile_desc_;
        
        // Filter 1: Skip very small block sizes (poor efficiency)
        if (tile.m0_block_ < 64 || tile.n0_block_ < 32) {
            continue;
        }
        
        // Filter 2: Prefer configurations with reasonable warp distribution
        int64_t total_warps = tile.m0_warp_ * tile.n0_warp_ * tile.k0_warp_;
        if (total_warps > 8 || total_warps < 1) {
            continue;
        }
        
        // Filter 3: Problem-specific filtering for sequence length ratios
        if (seq_len_ratio > 2 && tile.m0_block_ < tile.n0_block_) {
            // For long query sequences, prefer larger m0 blocks
            continue;
        }
        
        // Filter 4: Avoid excessive head dimension splitting
        if (tile.k0_block_ < fmha_problem.qk_head_dim_ / 4) {
            continue;
        }
        
        filtered.push_back(instance);
    }
    
    // If filtering is too aggressive, return a subset of original instances
    if (filtered.empty()) {
        VLOG(2) << "Heuristic filter too aggressive, returning subset of original instances";
        const size_t subset_size = std::min<size_t>(instances.size(), 10);
        filtered.assign(instances.begin(), instances.begin() + subset_size);
    }
    
    VLOG(2) << "Heuristic filter: " << instances.size() << " -> " << filtered.size() << " instances";
    return filtered;
}

std::vector<FmhaBatchPrefillCodeGen> FmhaBatchPrefillEmitter::CreateInstanceForConfig(
    const FmhaBatchPrefillConfig& config, const FmhaProblem& fmha_problem) 
{
    std::vector<FmhaBatchPrefillCodeGen> result;

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

        // Construct FmhaBatchPrefillCodeGen instance
        FmhaBatchPrefillCodeGen instance;
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

void FmhaBatchPrefillEmitter::GenerateInstances(FmhaProblem& fmha_problem)
{
    // Validate tuning mode
    FC_ENFORCE_EQ(FLAGS_FC_TUNING_MODE >= 0 && FLAGS_FC_TUNING_MODE <= 2, true,
                  Unavailable("Invalid tuning mode: {}, valid modes are 0 (heuristic), 1 (autotuning), 2 (hybrid)", 
                              FLAGS_FC_TUNING_MODE));

    // Check if instances already exist for this FMHA kind
    if (instance_map_.find(fmha_problem.kind_) != instance_map_.end() && 
        !instance_map_[fmha_problem.kind_].empty()) {
        VLOG(2) << "Instances already generated for FMHA kind: " << GetFmhaKindName(fmha_problem.kind_);
        return;
    }

    VLOG(1) << "Generating FMHA batch prefill instances for mode: " << FLAGS_FC_TUNING_MODE;

    // Load configurations from JSON files
    auto base_json_path = std::filesystem::path(FLAGS_FC_CONFIG_JSON_PATH) / GetFmhaKindName(fmha_problem.kind_);
    std::vector<FmhaBatchPrefillCodeGen> all_instances;
    
    // Load backup configurations (pre-validated, single-value configs)
    if (FLAGS_FC_ENABLE_BACKUP_JSON) {
        std::filesystem::path json_path = base_json_path / "backup_config.json";
        try {
            auto backup_configs = LoadConfigJson<std::vector<FmhaBatchPrefillConfig>>(json_path);
            for (const auto& config : backup_configs) {
                auto instances = CreateInstanceForConfig(config, fmha_problem);
                all_instances.insert(all_instances.end(), instances.begin(), instances.end());
            }
            VLOG(2) << "Loaded " << backup_configs.size() << " backup configurations";
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load backup config: " << e.what();
        }
    }

    // Load default configurations (parameter ranges for tuning)
    if (FLAGS_FC_ENABLE_DEFAULT_JSON) {
        std::filesystem::path json_path = base_json_path / "default_config.json";
        try {
            auto default_config = LoadConfigJson<FmhaBatchPrefillConfig>(json_path);
            auto instances = CreateInstanceForConfig(default_config, fmha_problem);
            all_instances.insert(all_instances.end(), instances.begin(), instances.end());
            VLOG(2) << "Loaded default configuration with " << instances.size() << " instances";
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load default config: " << e.what();
        }
    }

    // Load user configurations (custom user-defined configs)
    if (FLAGS_FC_ENABLE_USER_JSON) {
        std::filesystem::path json_path = base_json_path / "user_config.json";
        try {
            auto user_config = LoadConfigJson<FmhaBatchPrefillConfig>(json_path);
            auto instances = CreateInstanceForConfig(user_config, fmha_problem);
            all_instances.insert(all_instances.end(), instances.begin(), instances.end());
            VLOG(2) << "Loaded user configuration with " << instances.size() << " instances";
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load user config: " << e.what();
        }
    }

    // Filter out invalid instances
    std::vector<FmhaBatchPrefillCodeGen> valid_instances;
    for (const auto& instance : all_instances) {
        if (IsValidInstance(instance)) {
            valid_instances.push_back(instance);
        }
    }

    VLOG(2) << "Validation: " << all_instances.size() << " -> " << valid_instances.size() << " valid instances";

    if (valid_instances.empty()) {
        FC_THROW(Unavailable("No valid FMHA batch prefill instances found"));
    }

    // Apply mode-specific strategy
    std::vector<FmhaBatchPrefillCodeGen> final_instances;
    switch (FLAGS_FC_TUNING_MODE) {
        case 0: {
            // Heuristic mode: filter + random selection
            final_instances = HeuristicFilter(valid_instances, fmha_problem);
            if (!final_instances.empty()) {
                // Randomly select one instance for fast execution
                std::uniform_int_distribution<> dist(0, final_instances.size() - 1);
                auto selected = final_instances[dist(rng_)];
                final_instances = {selected};
                VLOG(1) << "Heuristic mode: selected 1 instance from " << valid_instances.size();
            }
            break;
        }
        case 1: {
            // Autotuning mode: use all valid instances
            final_instances = valid_instances;
            VLOG(1) << "Autotuning mode: using all " << final_instances.size() << " valid instances";
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
            
            VLOG(1) << "Hybrid mode: using " << final_instances.size() << " unique instances";
            break;
        }
        default:
            FC_THROW(Unavailable("Invalid tuning mode: {}", FLAGS_FC_TUNING_MODE));
    }

    if (final_instances.empty()) {
        FC_THROW(Unavailable("No final FMHA batch prefill instances after mode-specific filtering"));
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
                VLOG(3) << "Generated FMHA instance: " << instance_name;
            } else {
                VLOG(3) << "Skipped duplicate FMHA instance: " << instance_name;
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to create FMHA codegen for instance: " << instance.GetInstanceName()
                         << ", error: " << e.what();
        }
    }

    num_instances_ += generated_count;
    VLOG(1) << "Generated " << generated_count << " FMHA batch prefill instances for kind: " 
            << GetFmhaKindName(fmha_problem.kind_) << " (total: " << num_instances_ << ")";
}

void FmhaBatchPrefillEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
    VLOG(2) << "Cleared all FMHA batch prefill instances";
}

}  // namespace flashck
