#include "core/profiling/moe/moe_sorting/moe_sorting_emitter.h"

#include <algorithm>
#include <filesystem>
#include <random>

FC_DECLARE_int32(FC_TUNING_MODE);    // 0: heuristic, 1: autotuning, 2: hybrid
FC_DECLARE_bool(FC_ENABLE_BACKUP_JSON);   // Enable backup_config.json loading
FC_DECLARE_bool(FC_ENABLE_DEFAULT_JSON);  // Enable default_config.json loading  
FC_DECLARE_bool(FC_ENABLE_USER_JSON);     // Enable user_config.json loading
FC_DECLARE_string(FC_CONFIG_JSON_PATH);   // Base path for config files

namespace flashck {

bool MoeSortingEmitter::IsValidInstance(const MoeSortingCodeGen& instance)
{
    const auto& problem = instance.problem_;
    
    // Validate parameters are positive
    if (instance.internal_load_unroll_ <= 0 || instance.expert_tile_ <= 0 || 
        instance.min_block_per_cu_ <= 0) {
        VLOG(3) << "Invalid MoE sorting instance: negative or zero parameters not allowed";
        return false;
    }

    // Validate expert tile size doesn't exceed number of experts
    if (instance.expert_tile_ > problem.num_experts_) {
        VLOG(3) << "Invalid MoE sorting instance: expert tile " << instance.expert_tile_ 
                << " exceeds number of experts " << problem.num_experts_;
        return false;
    }

    // Validate unroll factor is reasonable for memory bandwidth
    if (instance.internal_load_unroll_ > 16) {
        VLOG(3) << "Invalid MoE sorting instance: unroll factor " << instance.internal_load_unroll_ 
                << " too large (may cause register pressure)";
        return false;
    }

    // Validate total memory footprint is reasonable
    const size_t estimated_memory = problem.input_tokens_ * sizeof(int32_t) * 2; // indices + values
    const size_t max_memory = 1ULL << 30; // 1GB limit
    if (estimated_memory > max_memory) {
        VLOG(3) << "Invalid MoE sorting instance: estimated memory " << estimated_memory 
                << " exceeds limit " << max_memory;
        return false;
    }

    return true;
}

std::vector<MoeSortingCodeGen> MoeSortingEmitter::HeuristicFilter(
    const std::vector<MoeSortingCodeGen>& instances,
    const MoeProblem& moe_problem)
{
    if (instances.empty()) {
        return {};
    }

    std::vector<MoeSortingCodeGen> filtered;
    
    // Score and rank instances based on sorting performance heuristics
    std::vector<std::pair<double, size_t>> scored_instances;
    
    for (size_t i = 0; i < instances.size(); ++i) {
        const auto& instance = instances[i];
        double score = 0.0;
        
        // 1. Memory bandwidth efficiency (prefer moderate unroll factors)
        if (instance.internal_load_unroll_ >= 2 && instance.internal_load_unroll_ <= 8) {
            score += 0.3;  // Good memory coalescing
        }
        
        // 2. Expert tile size efficiency
        const double expert_utilization = static_cast<double>(instance.expert_tile_) / moe_problem.num_experts_;
        if (expert_utilization >= 0.25 && expert_utilization <= 1.0) {
            score += 0.25;  // Good expert coverage
        }
        
        // 3. Load balancing considerations
        const int64_t tokens_per_expert = moe_problem.input_tokens_ * moe_problem.top_k_ / moe_problem.num_experts_;
        const int64_t work_per_tile = tokens_per_expert * instance.expert_tile_;
        if (work_per_tile >= 256 && work_per_tile <= 4096) {
            score += 0.2;  // Good workload granularity
        }
        
        // 4. Block occupancy optimization
        if (instance.min_block_per_cu_ >= 1 && instance.min_block_per_cu_ <= 4) {
            score += 0.15;  // Good occupancy for sorting
        }
        
        // 5. Problem size adaptation
        if (moe_problem.input_tokens_ >= 1024) {
            // For large problems, prefer higher unroll
            if (instance.internal_load_unroll_ >= 4) score += 0.1;
        } else {
            // For small problems, prefer lower unroll
            if (instance.internal_load_unroll_ <= 4) score += 0.1;
        }
        
        scored_instances.emplace_back(score, i);
    }
    
    // Sort by score (highest first)
    std::sort(scored_instances.begin(), scored_instances.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Select top candidates (limit to reasonable number for heuristic mode)
    size_t max_candidates = std::min(static_cast<size_t>(8), instances.size());
    filtered.reserve(max_candidates);
    
    for (size_t i = 0; i < max_candidates; ++i) {
        filtered.push_back(instances[scored_instances[i].second]);
    }
    
    VLOG(2) << "MoE sorting heuristic filter: reduced " << instances.size() 
            << " instances to " << filtered.size() << " candidates";
    
    return filtered;
}

std::vector<MoeSortingCodeGen> MoeSortingEmitter::CreateInstanceForConfig(
    const MoeSortingConfig& config, const MoeProblem& moe_problem) 
{
    std::vector<MoeSortingCodeGen> result;

    // Convert all config parameters to int64_t vectors for CartesianProduct
    std::vector<std::vector<int64_t>> all_param_lists = {
        // Sorting configuration
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.internal_load_unroll.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.expert_tile.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        // Launch configuration
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.launch.min_block_per_cu.values) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
    };

    CartesianProduct(all_param_lists, [&](const std::vector<int64_t>& vals) {
        size_t idx = 0;
        
        // Sorting configuration
        int64_t internal_load_unroll = vals[idx++];
        int64_t expert_tile = vals[idx++];
        
        // Launch configuration
        int64_t min_block_per_cu = vals[idx++];

        // Construct MoeSortingCodeGen
        MoeSortingCodeGen instance;
        instance.problem_ = moe_problem;
        instance.internal_load_unroll_ = internal_load_unroll;
        instance.expert_tile_ = expert_tile;
        instance.min_block_per_cu_ = min_block_per_cu;
        
        result.push_back(instance);
    });
    
    return result;
}

void MoeSortingEmitter::GenerateInstances(MoeProblem& moe_problem)
{
    // Validate tuning mode
    FC_ENFORCE_EQ(FLAGS_FC_TUNING_MODE >= 0 && FLAGS_FC_TUNING_MODE <= 2, true,
                  Unavailable("Invalid FC_TUNING_MODE: {}. Valid values: 0(heuristic), 1(autotuning), 2(hybrid)", 
                             FLAGS_FC_TUNING_MODE));

    std::vector<MoeSortingCodeGen> all_instances;

    // Configuration loading based on enabled flags
    auto base_json_path = std::filesystem::path(FLAGS_FC_CONFIG_JSON_PATH) / "moe_sorting";

    // Load backup configurations (pre-validated single configs)
    if (FLAGS_FC_ENABLE_BACKUP_JSON) {
        try {
            std::filesystem::path backup_path = base_json_path / "backup_config.json";
            if (std::filesystem::exists(backup_path)) {
                auto backup_configs = LoadConfigJson<std::vector<MoeSortingConfig>>(backup_path);
                for (const auto& config : backup_configs) {
                    auto backup_instances = CreateInstanceForConfig(config, moe_problem);
                    all_instances.insert(all_instances.end(), backup_instances.begin(), backup_instances.end());
                }
                VLOG(2) << "Loaded " << backup_configs.size() << " MoE sorting backup configurations";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load MoE sorting backup config: " << e.what();
        }
    }

    // Load default configurations (parameter ranges)
    if (FLAGS_FC_ENABLE_DEFAULT_JSON) {
        try {
            std::filesystem::path default_path = base_json_path / "default_config.json";
            if (std::filesystem::exists(default_path)) {
                auto default_config = LoadConfigJson<MoeSortingConfig>(default_path);
                auto default_instances = CreateInstanceForConfig(default_config, moe_problem);
                all_instances.insert(all_instances.end(), default_instances.begin(), default_instances.end());
                VLOG(2) << "Loaded MoE sorting default configuration with " << default_instances.size() << " instances";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load MoE sorting default config: " << e.what();
        }
    }

    // Load user configurations (custom parameter ranges)
    if (FLAGS_FC_ENABLE_USER_JSON) {
        try {
            std::filesystem::path user_path = base_json_path / "user_config.json";
            if (std::filesystem::exists(user_path)) {
                auto user_config = LoadConfigJson<MoeSortingConfig>(user_path);
                auto user_instances = CreateInstanceForConfig(user_config, moe_problem);
                all_instances.insert(all_instances.end(), user_instances.begin(), user_instances.end());
                VLOG(2) << "Loaded MoE sorting user configuration with " << user_instances.size() << " instances";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load MoE sorting user config: " << e.what();
        }
    }

    // Apply mode-specific processing
    std::vector<MoeSortingCodeGen> final_instances;
    
    switch (FLAGS_FC_TUNING_MODE) {
        case 0: {  // Heuristic mode: filter + random selection for fast execution
            auto filtered_instances = HeuristicFilter(all_instances, moe_problem);
            if (!filtered_instances.empty()) {
                // Randomly select one optimal configuration for fast execution
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dis(0, filtered_instances.size() - 1);
                final_instances.push_back(filtered_instances[dis(gen)]);
                VLOG(1) << "MoE sorting heuristic mode: selected 1 instance from " 
                        << filtered_instances.size() << " filtered candidates";
            }
            break;
        }
        case 1: {  // Autotuning mode: use all valid instances for comprehensive search
            final_instances = all_instances;
            VLOG(1) << "MoE sorting autotuning mode: using all " << all_instances.size() << " instances";
            break;
        }
        case 2: {  // Hybrid mode: heuristic filtering + broader search
            auto filtered_instances = HeuristicFilter(all_instances, moe_problem);
            final_instances = filtered_instances.empty() ? all_instances : filtered_instances;
            VLOG(1) << "MoE sorting hybrid mode: using " << final_instances.size() 
                    << " instances (filtered from " << all_instances.size() << ")";
            break;
        }
    }

    // Validate and store instances
    num_instances_ = 0;
    for (const auto& instance : final_instances) {
        if (IsValidInstance(instance)) {
            instance_map_[instance.GetInstanceName()] = instance;
            ++num_instances_;
        }
    }

    VLOG(1) << "Generated " << num_instances_ << " valid MoE sorting instances ";
            << " (mode " << FLAGS_FC_TUNING_MODE << ")";
}

void MoeSortingEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
}

}  // namespace flashck
