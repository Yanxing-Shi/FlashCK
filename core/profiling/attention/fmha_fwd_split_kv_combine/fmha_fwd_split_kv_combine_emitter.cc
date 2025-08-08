#include "core/profiling/attention/fmha_fwd_split_kv_combine/fmha_fwd_split_kv_combine_emitter.h"

#include <algorithm>
#include <filesystem>
#include <random>

FC_DECLARE_int32(FC_TUNING_MODE);    // 0: heuristic, 1: autotuning, 2: hybrid
FC_DECLARE_bool(FC_ENABLE_BACKUP_JSON);   // Enable backup_config.json loading
FC_DECLARE_bool(FC_ENABLE_DEFAULT_JSON);  // Enable default_config.json loading  
FC_DECLARE_bool(FC_ENABLE_USER_JSON);     // Enable user_config.json loading
FC_DECLARE_string(FC_CONFIG_JSON_PATH);   // Base path for config files

namespace flashck {

bool FmhaFwdSplitKVCombineEmitter::IsValidTile(const FmhaFwdSplitKVCombineTileDesc& tile_desc, const FmhaFwdSplitKVCombineProblem& fmha_fwd_split_kv_combine_problem)
{
    // Validate block tile dimensions
    if (tile_desc.n1_block_ <= 0) {
        return false;
    }
    
    // Check if n1_block aligns with v_head_dim
    if (fmha_fwd_split_kv_combine_problem.v_head_dim_ % tile_desc.n1_block_ != 0) {
        return false;
    }

    return true;
}

bool FmhaFwdSplitKVCombineEmitter::IsValidInstance(const FmhaFwdSplitKVCombineCodeGen& instance)
{
    return IsValidTile(instance.tile_desc_, instance.problem_);
}

std::vector<FmhaFwdSplitKVCombineCodeGen> FmhaFwdSplitKVCombineEmitter::HeuristicFilter(
    const std::vector<FmhaFwdSplitKVCombineCodeGen>& instances,
    const FmhaFwdSplitKVCombineProblem& fmha_fwd_split_kv_combine_problem)
{
    if (instances.empty()) {
        return {};
    }

    std::vector<FmhaFwdSplitKVCombineCodeGen> filtered;
    
    // Apply heuristic filters for split-kv combine operation
    for (const auto& instance : instances) {
        bool is_good = true;
        
        // Filter 1: Prefer block sizes that are multiples of warp size (32)
        if (instance.tile_desc_.n1_block_ % 32 != 0) {
            is_good = false;
        }
        
        // Filter 2: For small v_head_dim, prefer smaller block sizes
        if (fmha_fwd_split_kv_combine_problem.v_head_dim_ <= 64 && instance.tile_desc_.n1_block_ > 64) {
            is_good = false;
        }
        
        // Filter 3: For large v_head_dim, prefer larger block sizes for better efficiency
        if (fmha_fwd_split_kv_combine_problem.v_head_dim_ >= 128 && instance.tile_desc_.n1_block_ < 64) {
            is_good = false;
        }
        
        // Filter 4: Prefer power-of-2 block sizes for memory alignment
        int n1 = instance.tile_desc_.n1_block_;
        if ((n1 & (n1 - 1)) != 0) {
            is_good = false;
        }
        
        // Filter 5: Thread block size should be reasonable
        if (instance.max_thread_per_block_ < 64 || instance.max_thread_per_block_ > 1024) {
            is_good = false;
        }
        
        if (is_good) {
            filtered.push_back(instance);
        }
    }
    
    // If no instances pass the heuristic, return the original instances
    if (filtered.empty()) {
        VLOG(2) << "No instances passed heuristic filter, returning all instances";
        return instances;
    }
    
    VLOG(2) << "Heuristic filter: " << instances.size() << " -> " << filtered.size() << " instances";
    return filtered;
}

std::vector<FmhaFwdSplitKVCombineCodeGen> FmhaFwdSplitKVCombineEmitter::CreateInstanceForConfig(
    const FmhaFwdSplitKVCombineConfig& config, const FmhaFwdSplitKVCombineProblem& fmha_fwd_split_kv_combine_problem) 
{
    std::vector<FmhaFwdSplitKVCombineCodeGen> result;

    // Convert all config parameters to int64_t vectors for CartesianProduct
    std::vector<std::vector<int64_t>> all_param_lists = {
        // Block tile configuration (1 parameter) - using GetAllValues() for both modes compatibility
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.block_tile.n1.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        
        // Padding configuration (2 parameters, bool->int64_t)
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.trait.padding.s.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.trait.padding.dv.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        
        // Launch configuration (2 parameter)
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.launch.max_thread_per_block.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.launch.min_block_per_cu.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
    };

    // Generate all parameter combinations using CartesianProduct
    CartesianProduct(all_param_lists, [&](const std::vector<int64_t>& param_values) {
        size_t idx = 0;
        
        // Extract block tile parameters (only n1 for combine)
        int64_t n1_block = param_values[idx++];
        
        // Extract padding parameters (only s and dv for combine)
        bool is_pad_q_seq_len = static_cast<bool>(param_values[idx++]);
        bool is_pad_v_head_dim = static_cast<bool>(param_values[idx++]);
        
        // Extract launch parameters
        int64_t max_thread_per_block = param_values[idx++];
        int64_t min_block_per_cu = param_values[idx++];

        // Construct FmhaFwdSplitKVCombineCodeGen instance
        FmhaFwdSplitKVCombineCodeGen instance;
        instance.problem_ = fmha_fwd_split_kv_combine_problem;
        
        // Set tile descriptor (only n1 for combine operation)
        instance.tile_desc_.n1_block_ = n1_block;
        
        // Set padding configuration (only s and dv for combine)
        instance.is_pad_q_seq_len_ = is_pad_q_seq_len;
        instance.is_pad_v_head_dim_ = is_pad_v_head_dim;
        
        // Set launch configuration
        instance.max_thread_per_block_ = max_thread_per_block;
        instance.min_block_per_cu_ = min_block_per_cu;
        
        result.push_back(instance);
    });

    return result;
}

void FmhaFwdSplitKVCombineEmitter::GenerateInstances(FmhaFwdSplitKVCombineProblem& fmha_fwd_split_kv_combine_problem)
{
    // Validate tuning mode
    FC_ENFORCE_EQ(FLAGS_FC_TUNING_MODE >= 0 && FLAGS_FC_TUNING_MODE <= 2, true,
                  Unavailable("Invalid tuning mode: {}, valid modes are 0 (heuristic), 1 (autotuning), 2 (hybrid)", 
                              FLAGS_FC_TUNING_MODE));

    VLOG(1) << "Generating FMHA split-KV combine instances for mode: " << FLAGS_FC_TUNING_MODE;

    // Load configurations from JSON files
    auto base_json_path = std::filesystem::path(FLAGS_FC_CONFIG_JSON_PATH) / "attention" / "fmha_fwd_split_kv_combine";
    std::vector<FmhaFwdSplitKVCombineCodeGen> all_instances;
    
    // Load backup configurations (pre-validated, single-value configs)
    if (FLAGS_FC_ENABLE_BACKUP_JSON) {
        std::filesystem::path json_path = base_json_path / "backup_config.json";
        try {
            auto backup_configs = LoadConfigJson<std::vector<FmhaFwdSplitKVCombineConfig>>(json_path);
            for (const auto& config : backup_configs) {
                auto instances = CreateInstanceForConfig(config, fmha_fwd_split_kv_combine_problem);
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
            auto instances = CreateInstanceForConfig(default_config, fmha_fwd_split_kv_combine_problem);
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
            auto instances = CreateInstanceForConfig(user_config, fmha_fwd_split_kv_combine_problem);
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
            final_instances = HeuristicFilter(valid_instances, fmha_fwd_split_kv_combine_problem);
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
            auto heuristic_instances = HeuristicFilter(valid_instances, fmha_fwd_split_kv_combine_problem);
            final_instances = heuristic_instances;
            final_instances.insert(final_instances.end(), valid_instances.begin(), valid_instances.end());
            
            // Remove duplicates
            std::sort(final_instances.begin(), final_instances.end(), 
                     []( auto& a,  auto& b) {
                         return a.GetInstanceName() < b.GetInstanceName();
                     });
            final_instances.erase(std::unique(final_instances.begin(), final_instances.end(),
                                            []( auto& a,  auto& b) {
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
    int64_t generated_count = 0;

    for (auto& instance : final_instances) {
        try {
            std::string instance_name = instance.GetInstanceName();
            
            // Avoid duplicates
            if (instance_map_.find(instance_name) == instance_map_.end()) {
                instance_map_[instance_name] = instance;
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
    VLOG(1) << "Generated " << generated_count << " FMHA split-KV combine instances "
            << " (total: " << num_instances_ << ")";
}

void FmhaFwdSplitKVCombineEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
    VLOG(2) << "Cleared all FMHA split-KV combine instances";
}

}  // namespace flashck
