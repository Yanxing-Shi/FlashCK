#include "core/profiling/norm/layer_norm/layer_norm_emitter.h"

#include <algorithm>
#include <filesystem>
#include <random>

FC_DECLARE_int32(FC_TUNING_MODE);    // 0: heuristic, 1: autotuning, 2: hybrid
FC_DECLARE_bool(FC_ENABLE_BACKUP_JSON);   // Enable backup_config.json loading
FC_DECLARE_bool(FC_ENABLE_DEFAULT_JSON);  // Enable default_config.json loading  
FC_DECLARE_bool(FC_ENABLE_USER_JSON);     // Enable user_config.json loading
FC_DECLARE_string(FC_CONFIG_JSON_PATH);   // Base path for config files

namespace flashck {

bool LayerNormEmitter::IsValidTile(const LayerNormTileDesc& tile_desc, const LayerNormProblem& layer_norm_problem)
{
    // Validate all tile parameters are positive
    if (tile_desc.m_repeat_ <= 0 || tile_desc.n_repeat_ <= 0 || 
        tile_desc.m_thread_per_block_ <= 0 || tile_desc.n_thread_per_block_ <= 0 ||
        tile_desc.n_vector_ <= 0) {
        VLOG(3) << "Invalid Layer Normalization tile: negative or zero values not allowed";
        return false;
    }

    // Validate thread block size doesn't exceed hardware limits
    const int total_threads = tile_desc.m_thread_per_block_ * tile_desc.n_thread_per_block_;
    if (total_threads > 1024) {
        VLOG(3) << "Invalid Layer Normalization tile: thread block size " << total_threads << " exceeds limit (1024)";
        return false;
    }

    // Validate vector size alignment with data types
    const int data_type_size = SizeOf(layer_norm_problem.x_dtype_);
    if (tile_desc.n_vector_ % (4 / data_type_size) != 0) {
        VLOG(3) << "Invalid Layer Normalization tile: vector size " << tile_desc.n_vector_
                << " not aligned with data type size " << data_type_size;
        return false;
    }

    // Validate total work per thread is reasonable
    const int work_per_thread = tile_desc.m_repeat_ * tile_desc.n_repeat_;
    if (work_per_thread > 64 || work_per_thread < 1) {
        VLOG(3) << "Invalid Layer Normalization tile: work per thread " << work_per_thread 
                << " outside reasonable range [1, 64]";
        return false;
    }

    // Validate against problem dimensions
    const int64_t total_m_coverage = tile_desc.m_thread_per_block_ * tile_desc.m_repeat_;
    const int64_t total_n_coverage = tile_desc.n_thread_per_block_ * tile_desc.n_repeat_ * tile_desc.n_vector_;

    if (total_m_coverage > layer_norm_problem.m_ || total_n_coverage > layer_norm_problem.n_) {
        VLOG(3) << "Invalid Layer Normalization tile: coverage (" << total_m_coverage << "," << total_n_coverage 
                << ") exceeds problem dims (" << layer_norm_problem.m_ << "," << layer_norm_problem.n_ << ")";
        return false;
    }

    return true;
}

bool LayerNormEmitter::IsValidInstance(const LayerNormCodeGen& instance)
{
    return IsValidTile(instance.tile_desc_, instance.problem_);
}

std::vector<LayerNormCodeGen> LayerNormEmitter::HeuristicFilter(
    const std::vector<LayerNormCodeGen>& instances,
    const LayerNormProblem& layer_norm_problem)
{
    if (instances.empty()) {
        return {};
    }

    std::vector<LayerNormCodeGen> filtered;
    
    // Score and rank instances based on Layer Normalization performance heuristics
    std::vector<std::pair<double, size_t>> scored_instances;
    
    for (size_t i = 0; i < instances.size(); ++i) {
        const auto& tile_desc = instances[i].tile_desc_;
        double score = 0.0;
        
        // 1. Memory coalescing efficiency (prefer aligned vector access)
        if (tile_desc.n_vector_ >= 4) {
            score += 0.3;  // Good vectorization
        }
        if (tile_desc.n_vector_ % 4 == 0) {
            score += 0.2;  // Aligned access
        }
        
        // 2. Thread utilization efficiency
        const int total_threads = tile_desc.m_thread_per_block_ * tile_desc.n_thread_per_block_;
        if (total_threads >= 256 && total_threads <= 512) {
            score += 0.25;  // Sweet spot for most GPUs
        }
        
        // 3. Work balance per thread
        const int work_per_thread = tile_desc.m_repeat_ * tile_desc.n_repeat_;
        if (work_per_thread >= 4 && work_per_thread <= 16) {
            score += 0.15;  // Good work granularity
        }
        
        // 4. Feature dimension coverage (efficient reduction)
        const int64_t n_coverage_per_block = tile_desc.n_thread_per_block_ * tile_desc.n_repeat_ * tile_desc.n_vector_;
        if (layer_norm_problem.n_ % n_coverage_per_block == 0) {
            score += 0.1;  // Perfect feature dimension fit
        }
        
        scored_instances.emplace_back(score, i);
    }
    
    // Sort by score (highest first)
    std::sort(scored_instances.begin(), scored_instances.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Select top candidates (limit to reasonable number for heuristic mode)
    size_t max_candidates = std::min(static_cast<size_t>(12), instances.size());
    filtered.reserve(max_candidates);
    
    for (size_t i = 0; i < max_candidates; ++i) {
        filtered.push_back(instances[scored_instances[i].second]);
    }
    
    VLOG(2) << "Layer Normalization heuristic filter: reduced " << instances.size() 
            << " instances to " << filtered.size() << " candidates";
    
    return filtered;
}

std::vector<LayerNormCodeGen> LayerNormEmitter::CreateInstanceForConfig(
    const NormConfig& config, const LayerNormProblem& layer_norm_problem) 
{
    std::vector<LayerNormCodeGen> result;

    // Convert all config parameters to int64_t vectors for CartesianProduct
    std::vector<std::vector<int64_t>> all_param_lists = {
        // Thread configuration
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.m_repeat.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.n_repeat.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.m_thread_per_block.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.n_thread_per_block.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        [&]{ std::vector<int64_t> v; 
             for (auto x : config.tile_shape.n_vector.GetAllValues()) v.push_back(static_cast<int64_t>(x)); 
             return v; }(),
        // TraitConfig (convert bool to int64_t)
        [&]{ std::vector<int64_t> v; for (auto x : config.trait.padding.n.GetAllValues()) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.trait.is_two_pass.GetAllValues()) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // LaunchConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.launch.max_thread_per_block.GetAllValues()) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.launch.min_block_per_cu.GetAllValues()) v.emplace_back(static_cast<int64_t>(x)); return v; }(),

    };

    CartesianProduct(all_param_lists, [&](const std::vector<int64_t>& vals) {
        size_t idx = 0;
        
        // Tile configuration
        int64_t m_repeat = vals[idx++];
        int64_t n_repeat = vals[idx++];
        int64_t m_thread_per_block = vals[idx++];
        int64_t n_thread_per_block = vals[idx++];
        int64_t n_vector = vals[idx++];
        
        // Trait
        bool is_pad_n = static_cast<bool>(vals[idx++]);
        bool is_two_pass = static_cast<bool>(vals[idx++]);

        // launch
        int64_t max_thread_per_block = vals[idx++];
        int64_t min_block_per_cu = vals[idx++];

        // Construct LayerNormCodeGen for Layer Normalization
        LayerNormCodeGen instance;
        instance.problem_ = layer_norm_problem;
        instance.tile_desc_ = LayerNormTileDesc{m_repeat, n_repeat, m_thread_per_block, 
                                               n_thread_per_block, n_vector};
        instance.is_pad_n_ = is_pad_n;
        instance.is_two_pass_ = is_two_pass; 
        instance.max_thread_per_block_ = max_thread_per_block;
        instance.min_block_per_cu_ = min_block_per_cu;
        result.push_back(instance);
    });
    
    return result;
}

void LayerNormEmitter::GenerateInstances(LayerNormProblem& layer_norm_problem)
{
    // Validate tuning mode
    FC_ENFORCE_EQ(FLAGS_FC_TUNING_MODE >= 0 && FLAGS_FC_TUNING_MODE <= 2, true,
                  Unavailable("Invalid FC_TUNING_MODE: {}. Valid values: 0(heuristic), 1(autotuning), 2(hybrid)", 
                             FLAGS_FC_TUNING_MODE));

    std::vector<LayerNormCodeGen> all_instances;

    // Configuration loading based on enabled flags
    auto base_json_path = std::filesystem::path(FLAGS_FC_CONFIG_JSON_PATH) / "norm" / "layer_norm";

    // Load backup configurations (pre-validated single configs)
    if (FLAGS_FC_ENABLE_BACKUP_JSON) {
        try {
            std::filesystem::path backup_path = base_json_path / "backup_config.json";
            if (std::filesystem::exists(backup_path)) {
                auto backup_configs = LoadConfigJson<std::vector<NormConfig>>(backup_path);
                for (const auto& config : backup_configs) {
                    auto backup_instances = CreateInstanceForConfig(config, layer_norm_problem);
                    all_instances.insert(all_instances.end(), backup_instances.begin(), backup_instances.end());
                }
                VLOG(2) << "Loaded " << backup_configs.size() << " Layer Normalization backup configurations";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load Layer Normalization backup config: " << e.what();
        }
    }

    // Load default configurations (parameter ranges)
    if (FLAGS_FC_ENABLE_DEFAULT_JSON) {
        try {
            std::filesystem::path default_path = base_json_path / "default_config.json";
            if (std::filesystem::exists(default_path)) {
                auto default_config = LoadConfigJson<NormConfig>(default_path);
                auto default_instances = CreateInstanceForConfig(default_config, layer_norm_problem);
                all_instances.insert(all_instances.end(), default_instances.begin(), default_instances.end());
                VLOG(2) << "Loaded Layer Normalization default configuration with " << default_instances.size() << " instances";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load Layer Normalization default config: " << e.what();
        }
    }

    // Load user configurations (custom parameter ranges)
    if (FLAGS_FC_ENABLE_USER_JSON) {
        try {
            std::filesystem::path user_path = base_json_path / "user_config.json";
            if (std::filesystem::exists(user_path)) {
                auto user_config = LoadConfigJson<NormConfig>(user_path);
                auto user_instances = CreateInstanceForConfig(user_config, layer_norm_problem);
                all_instances.insert(all_instances.end(), user_instances.begin(), user_instances.end());
                VLOG(2) << "Loaded Layer Normalization user configuration with " << user_instances.size() << " instances";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load Layer Normalization user config: " << e.what();
        }
    }

    // Apply mode-specific processing
    std::vector<LayerNormCodeGen> final_instances;
    
    switch (FLAGS_FC_TUNING_MODE) {
        case 0: {  // Heuristic mode: filter + random selection for fast execution
            auto filtered_instances = HeuristicFilter(all_instances, layer_norm_problem);
            if (!filtered_instances.empty()) {
                // Randomly select one optimal configuration for fast execution
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dis(0, filtered_instances.size() - 1);
                final_instances.push_back(filtered_instances[dis(gen)]);
                VLOG(1) << "Layer Normalization heuristic mode: selected 1 instance from " 
                        << filtered_instances.size() << " filtered candidates";
            }
            break;
        }
        case 1: {  // Autotuning mode: use all valid instances for comprehensive search
            final_instances = all_instances;
            VLOG(1) << "Layer Normalization autotuning mode: using all " << all_instances.size() << " instances";
            break;
        }
        case 2: {  // Hybrid mode: heuristic filtering + broader search
            auto filtered_instances = HeuristicFilter(all_instances, layer_norm_problem);
            final_instances = filtered_instances.empty() ? all_instances : filtered_instances;
            VLOG(1) << "Layer Normalization hybrid mode: using " << final_instances.size() 
                    << " instances (filtered from " << all_instances.size() << ")";
            break;
        }
    }

    // Validate and store instances
    num_instances_ = 0;
    for (auto& instance : final_instances) {
        if (IsValidInstance(instance)) {
            instance_map_[instance.GetInstanceName()] = instance;
            ++num_instances_;
        }
    }

    VLOG(1) << "Generated " << num_instances_ << " valid Layer Normalization instances " 
            << " (mode " << FLAGS_FC_TUNING_MODE << ")";
}


void LayerNormEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
}

}  // namespace flashck
