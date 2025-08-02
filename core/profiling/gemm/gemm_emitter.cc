#include "core/profiling/gemm/gemm_emitter.h"

#include <algorithm>
#include <filesystem>
#include <random>
#include "core/utils/common.h"

FC_DECLARE_int32(FC_TUNING_MODE);      // 0: heuristic, 1: autotuning, 2: hybrid
FC_DECLARE_bool(FC_ENABLE_BACKUP_JSON);    // Enable backup_config.json loading
FC_DECLARE_bool(FC_ENABLE_DEFAULT_JSON);   // Enable default_config.json loading  
FC_DECLARE_bool(FC_ENABLE_USER_JSON);      // Enable user_config.json loading
FC_DECLARE_string(FC_CONFIG_JSON_PATH);    // Base path for config files

namespace flashck {

bool GemmEmitter::IsValidTile(const GemmTileDesc& tile_desc, const GemmProblem& gemm_problem) const
{
    // Validate all tile parameters are positive
    if (tile_desc.m_block_ <= 0 || tile_desc.n_block_ <= 0 || tile_desc.k_block_ <= 0 ||
        tile_desc.m_warp_ <= 0 || tile_desc.n_warp_ <= 0 || tile_desc.k_warp_ <= 0 ||
        tile_desc.m_warp_tile_ <= 0 || tile_desc.n_warp_tile_ <= 0 || tile_desc.k_warp_tile_ <= 0) {
        VLOG(3) << "Invalid GEMM tile: negative or zero values not allowed";
        return false;
    }

    // Validate thread block size doesn't exceed hardware limits
    const int total_threads = tile_desc.m_block_ * tile_desc.n_block_;
    if (total_threads > 1024) {
        VLOG(3) << "Invalid GEMM tile: thread block size " << total_threads << " exceeds limit (1024)";
        return false;
    }

    // Validate tile sizes don't exceed problem dimensions
    if (tile_desc.m_block_ > gemm_problem.m_ || tile_desc.n_block_ > gemm_problem.n_ || 
        tile_desc.k_block_ > gemm_problem.k_) {
        VLOG(3) << "Invalid GEMM tile: block dims (" << tile_desc.m_block_ << "," 
                << tile_desc.n_block_ << "," << tile_desc.k_block_
                << ") exceed problem dims (" << gemm_problem.m_ << "," << gemm_problem.n_ 
                << "," << gemm_problem.k_ << ")";
        return false;
    }

    // Validate warp combination is allowed
    std::tuple<int, int, int> warp_tuple = std::make_tuple(tile_desc.m_warp_, tile_desc.n_warp_, tile_desc.k_warp_);
    if (std::find(g_tile_gemm_allowed_warp_combinations.begin(), g_tile_gemm_allowed_warp_combinations.end(), 
                  warp_tuple) == g_tile_gemm_allowed_warp_combinations.end()) {
        VLOG(3) << "Invalid GEMM warp combination: (" << tile_desc.m_warp_ << "," 
                << tile_desc.n_warp_ << "," << tile_desc.k_warp_ << ")";
        return false;
    }

    // Validate dimension alignment: block dims must be divisible by warp*warp_tile dims
    if (tile_desc.m_block_ % (tile_desc.m_warp_ * tile_desc.m_warp_tile_) != 0) {
        VLOG(3) << "GEMM dimension alignment failed: m_block(" << tile_desc.m_block_ 
                << ") % [" << tile_desc.m_warp_ << "x" << tile_desc.m_warp_tile_ << "] != 0";
        return false;
    }
    if (tile_desc.n_block_ % (tile_desc.n_warp_ * tile_desc.n_warp_tile_) != 0) {
        VLOG(3) << "GEMM dimension alignment failed: n_block(" << tile_desc.n_block_ 
                << ") % [" << tile_desc.n_warp_ << "x" << tile_desc.n_warp_tile_ << "] != 0";
        return false;
    }
    if (tile_desc.k_block_ % (tile_desc.k_warp_ * tile_desc.k_warp_tile_) != 0) {
        VLOG(3) << "Dimension alignment failed: tile_k(" << tile_desc.k_block_ << ") % [" << tile_desc.k_warp_ << "x" << tile_desc.k_warp_tile_ << "] = "
                << (tile_desc.k_block_ % (tile_desc.k_warp_ * tile_desc.k_warp_tile_));
        return false;
    }

    // LDS capacity verification
    size_t matrix_a_size = tile_desc.m_block_ * tile_desc.k_block_ *  SizeOf(gemm_problem.a_dtype_);
    size_t matrix_b_size = tile_desc.n_block_ * tile_desc.k_block_ * SizeOf(gemm_problem.b_dtype_);
    size_t total_tile_in_lds = matrix_a_size + matrix_b_size;

    size_t max_tile_size =  (1 << 16); // 64KB

    if (total_tile_in_lds > max_tile_size) {
        VLOG(3) << "LDS capacity exceeded: Total required " << total_tile_in_lds << "B (" << (total_tile_in_lds / 1024.0) << "KB) > "
                << "maximum allowed " << max_tile_size << "B (" << (max_tile_size / 1024) << "KB). Breakdown:\n"
                << "- Matrix A (" << DataTypeToString(gemm_problem.a_dtype_) << "): " << tile_desc.m_block_ << "x" << tile_desc.k_block_ << " = " << matrix_a_size << "B\n"
                << "- Matrix B (" << DataTypeToString(gemm_problem.b_dtype_) << "): " << tile_desc.n_block_ << "x" << tile_desc.k_block_ << " = " << matrix_b_size << "B";
        return false;
    }

    // Warp tile combination validation
    // Compose warp_tile_key as "matrix_a_matrix_b_matrix_c"
    std::string warp_tile_key = Sprintf("{}_{}_{}", DataTypeToString(gemm_problem.a_dtype_),
                                         DataTypeToString(gemm_problem.b_dtype_),
                                         DataTypeToString(gemm_problem.c_dtype_));
    std::array<int64_t, 3> current_combination = {tile_desc.m_warp_tile_, tile_desc.n_warp_tile_, tile_desc.k_warp_tile_};

    std::string gpu_name = GetDeviceName();

    auto gpu_warp_tile_key_it = g_tile_gemm_warp_tile_supported_combinations.find(gpu_name);
    if (gpu_warp_tile_key_it == g_tile_gemm_warp_tile_supported_combinations.end()) {
        VLOG(3) << "Trait: [GEMM], No valid warp tile combinations found for " << gpu_name << "/" << warp_tile_key << ", skip this check.";
        return false;
    }
    const auto& gpu_warp_tile_key = gpu_warp_tile_key_it->second;
    auto allowed_combinations_it = gpu_warp_tile_key.find(warp_tile_key);
    if (allowed_combinations_it == gpu_warp_tile_key.end()) {
        VLOG(3) << "Trait: [GEMM], No valid warp tile combinations found for " << gpu_name << "/" << warp_tile_key << ", skip this check.";
        return false;
    }
    const auto& allowed_combinations = allowed_combinations_it->second;
    if (std::find(allowed_combinations.begin(), allowed_combinations.end(), current_combination) == allowed_combinations.end()) {
        VLOG(3) << "Trait: [GEMM], Invalid warp combination: [" << current_combination[0] << ", " << current_combination[1] << ", " << current_combination[2]
                << "] not in allowed list. Valid combinations for data type '" << warp_tile_key << "': ";
        for (const auto& comb : allowed_combinations) {
            VLOG(3) << "  [" << comb[0] << ", " << comb[1] << ", " << comb[2] << "]";
        }
        return false;
    }

    return true;
}

bool GemmEmitter::IsValidCombination(const PipelineVersionEnum& pipeline, const EpilogueEnum& epilogue, const PipelineSchedulerEnum& scheduler)
{
    // Check if the current combination is valid (compare enums, not strings)
    return std::find(g_tile_gemm_unsupported_combinations.begin(), g_tile_gemm_unsupported_combinations.end(),
                    std::make_tuple(pipeline, epilogue, scheduler)) == g_tile_gemm_unsupported_combinations.end();
}

bool GemmEmitter::IsValidInstance(const GemmCodeGen& instance)
{
    return IsValidTile(instance.tile_desc_, instance.problem_) && 
           IsValidCombination(instance.pipeline_version_, instance.pipeline_epilogue_, instance.pipeline_scheduler_);
}

std::vector<GemmCodeGen> GemmEmitter::HeuristicFilter(const std::vector<GemmCodeGen>& instances, 
                                                     const GemmProblem& gemm_problem)
{
    if (instances.empty()) {
        return {};
    }

    std::vector<GemmCodeGen> filtered_instances;
    
    // Score and rank instances based on multiple performance heuristics
    std::vector<std::pair<double, size_t>> scored_instances;
    
    for (size_t i = 0; i < instances.size(); ++i) {
        const auto& tile_desc = instances[i].tile_desc_;
        double score = 0.0;
        
        // 1. Memory throughput efficiency (favor larger tile sizes up to sweet spot)
        int64_t total_work = tile_desc.m_block_ * tile_desc.n_block_ * tile_desc.k_block_;
        if (total_work >= 4096 && total_work <= 32768) {  // Sweet spot for most problems
            score += 0.3;
        }
        
        // 2. Register efficiency (balance between utilization and pressure)
        int64_t reg_estimate = tile_desc.m_warp_tile_ * tile_desc.n_warp_tile_ * tile_desc.k_warp_tile_;
        if (reg_estimate >= 16 && reg_estimate <= 256) {  // Good register usage range
            score += 0.25;
        }
        
        // 3. Problem fit analysis
        // Prefer tile sizes that divide evenly into problem dimensions
        if (gemm_problem.m_ % tile_desc.m_block_ == 0) score += 0.15;
        if (gemm_problem.n_ % tile_desc.n_block_ == 0) score += 0.15;
        if (gemm_problem.k_ % tile_desc.k_block_ == 0) score += 0.1;
        
        // 4. Memory access efficiency (favor configurations with good coalescing)
        auto is_efficient_size = [](int64_t size) {
            return (size % 32 == 0) || (size & (size - 1)) == 0;  // Multiple of 32 or power of 2
        };
        
        if (is_efficient_size(tile_desc.m_block_)) score += 0.05;
        if (is_efficient_size(tile_desc.n_block_)) score += 0.05;
        if (is_efficient_size(tile_desc.k_block_)) score += 0.05;
        
        // 5. Pipeline efficiency bonus
        if (instances[i].pipeline_version_ == GetPipelineVersionEnumFromString("compv3") ||
            instances[i].pipeline_version_ == GetPipelineVersionEnumFromString("compv4")) {
            score += 0.1;  // Favor newer pipeline versions
        }
        
        scored_instances.emplace_back(score, i);
    }
    
    // Sort by score (highest first)
    std::sort(scored_instances.begin(), scored_instances.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Select top candidates (limit to reasonable number for heuristic mode)
    size_t max_candidates = std::min(static_cast<size_t>(20), instances.size());
    filtered_instances.reserve(max_candidates);
    
    for (size_t i = 0; i < max_candidates; ++i) {
        filtered_instances.push_back(instances[scored_instances[i].second]);
    }
    
    VLOG(2) << "GEMM heuristic filter: reduced " << instances.size() 
            << " instances to " << filtered_instances.size() << " candidates";
    
    return filtered_instances;
}


// Generate all possible GemmCodeGen instances from a GemmConfig
std::vector<GemmCodeGen> GemmEmitter::CreateInstanceForConfig(const GemmConfig& config, const GemmProblem& gemm_problem) {
    std::vector<GemmCodeGen> result;

    std::vector<std::vector<int64_t>> all_lists = {
        // BlockConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.m.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.n.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.k.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // WarpConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_warps.m.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_warps.n.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_warps.k.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // WarpTileConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.warp_tile.m.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.warp_tile.n.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.warp_tile.k.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // PaddingConfig (convert bool to int64_t)
        [&]{ std::vector<int64_t> v; for (auto x : config.padding.m.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.padding.n.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.padding.k.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // LaunchConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.launch.min_block_per_cu.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // PartitionConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.partition.num_wave_groups.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.partition.tile_partitioner_group_num.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.partition.tile_partitioner_m01.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // PipelineConfig (enum as int64_t)
        [&]{ std::vector<int64_t> v; for (const auto& x : config.pipeline.version.values) v.emplace_back(static_cast<int64_t>(GetPipelineVersionEnumFromString(x))); return v; }(),
        [&]{ std::vector<int64_t> v; for (const auto& x : config.pipeline.scheduler.values) v.emplace_back(static_cast<int64_t>(GetPipelineSchedulerEnumFromString(x))); return v; }(),
        [&]{ std::vector<int64_t> v; for (const auto& x : config.pipeline.epilogue.values) v.emplace_back(static_cast<int64_t>(GetEpilogueEnumFromString(x))); return v; }(),
    };

    CartesianProduct(all_lists, [&](const std::vector<int64_t>& vals) {
        size_t idx = 0;
        // BlockConfig
        int64_t m_block = vals[idx++];
        int64_t n_block = vals[idx++];
        int64_t k_block = vals[idx++];
        // WarpConfig
        int64_t m_warp = vals[idx++];
        int64_t n_warp = vals[idx++];
        int64_t k_warp = vals[idx++];
        // WarpTileConfig
        int64_t m_warp_tile = vals[idx++];
        int64_t n_warp_tile = vals[idx++];
        int64_t k_warp_tile = vals[idx++];
        // PaddingConfig
        bool pad_m = static_cast<bool>(vals[idx++]);
        bool pad_n = static_cast<bool>(vals[idx++]);
        bool pad_k = static_cast<bool>(vals[idx++]);
        // LaunchConfig
        int64_t min_block_per_cu = vals[idx++];
        // PartitionConfig
        int64_t num_wave_groups = vals[idx++];
        int64_t tile_partitioner_group_num = vals[idx++];
        int64_t tile_partitioner_m01 = vals[idx++];
        // PipelineConfig
        auto version = static_cast<PipelineVersionEnum>(vals[idx++]);
        auto scheduler = static_cast<PipelineSchedulerEnum>(vals[idx++]);
        // EpilogueConfig
        auto epilogue = static_cast<EpilogueEnum>(vals[idx++]);

        // Construct  GemmCodeGen
        GemmCodeGen gemm;
        gemm.problem_ = gemm_problem;
        gemm.tile_desc_ = GemmTileDesc{m_block, n_block, k_block, m_warp, n_warp, k_warp, m_warp_tile, n_warp_tile, k_warp_tile};
        gemm.pipeline_version_ = version;
        gemm.pipeline_scheduler_ = scheduler;
        gemm.pipeline_epilogue_ = epilogue;
        gemm.is_pad_m_ = pad_m;
        gemm.is_pad_n_ = pad_n;
        gemm.is_pad_k_ = pad_k;
        gemm.min_block_per_cu_ = min_block_per_cu;
        gemm.num_wave_groups_ = num_wave_groups;
        gemm.tile_partitioner_group_num_ = tile_partitioner_group_num;
        gemm.tile_partitioner_m01_ = tile_partitioner_m01;
        result.push_back(gemm);
    });
    return result;
}

void GemmEmitter::GenerateInstances(GemmProblem& gemm_problem)
{
    // Validate tuning mode
    FC_ENFORCE_EQ(FLAGS_FC_TUNING_MODE >= 0 && FLAGS_FC_TUNING_MODE <= 2, true,
                  Unavailable("Invalid FC_TUNING_MODE: {}. Valid values: 0(heuristic), 1(autotuning), 2(hybrid)", 
                             FLAGS_FC_TUNING_MODE));

    // Check if instances already exist for this GEMM kind
    if (instance_map_.find(gemm_problem.kind_) != instance_map_.end() && 
        !instance_map_[gemm_problem.kind_].empty()) {
        VLOG(2) << "GEMM instances already generated for kind: " << GetGemmKindName(gemm_problem.kind_);
        return;
    }

    std::vector<GemmCodeGen> all_instances;

    // Configuration loading based on enabled flags
    auto base_json_path = std::filesystem::path(FLAGS_FC_CONFIG_JSON_PATH) / GetGemmKindName(gemm_problem.kind_);

    // Load backup configurations (pre-validated single configs)
    if (FLAGS_FC_ENABLE_BACKUP_JSON) {
        try {
            std::filesystem::path backup_path = base_json_path / "backup_config.json";
            if (std::filesystem::exists(backup_path)) {
                auto backup_configs = LoadConfigJson<std::vector<GemmConfig>>(backup_path);
                for (const auto& config : backup_configs) {
                    auto backup_instances = CreateInstanceForConfig(config, gemm_problem);
                    all_instances.insert(all_instances.end(), backup_instances.begin(), backup_instances.end());
                }
                VLOG(2) << "Loaded " << backup_configs.size() << " GEMM backup configurations";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load GEMM backup config: " << e.what();
        }
    }

    // Load default configurations (parameter ranges)
    if (FLAGS_FC_ENABLE_DEFAULT_JSON) {
        try {
            std::filesystem::path default_path = base_json_path / "default_config.json";
            if (std::filesystem::exists(default_path)) {
                auto default_config = LoadConfigJson<GemmConfig>(default_path);
                auto default_instances = CreateInstanceForConfig(default_config, gemm_problem);
                all_instances.insert(all_instances.end(), default_instances.begin(), default_instances.end());
                VLOG(2) << "Loaded GEMM default configuration with " << default_instances.size() << " instances";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load GEMM default config: " << e.what();
        }
    }

    // Load user configurations (custom parameter ranges)
    if (FLAGS_FC_ENABLE_USER_JSON) {
        try {
            std::filesystem::path user_path = base_json_path / "user_config.json";
            if (std::filesystem::exists(user_path)) {
                auto user_config = LoadConfigJson<GemmConfig>(user_path);
                auto user_instances = CreateInstanceForConfig(user_config, gemm_problem);
                all_instances.insert(all_instances.end(), user_instances.begin(), user_instances.end());
                VLOG(2) << "Loaded GEMM user configuration with " << user_instances.size() << " instances";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load GEMM user config: " << e.what();
        }
    }

    // Apply mode-specific processing
    std::vector<GemmCodeGen> final_instances;
    
    switch (FLAGS_FC_TUNING_MODE) {
        case 0: {  // Heuristic mode: filter + random selection for fast execution
            auto filtered_instances = HeuristicFilter(all_instances, gemm_problem);
            if (!filtered_instances.empty()) {
                // Randomly select one optimal configuration for fast execution
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dis(0, filtered_instances.size() - 1);
                final_instances.push_back(filtered_instances[dis(gen)]);
                VLOG(1) << "GEMM heuristic mode: selected 1 instance from " 
                        << filtered_instances.size() << " filtered candidates";
            }
            break;
        }
        case 1: {  // Autotuning mode: use all valid instances for comprehensive search
            final_instances = all_instances;
            VLOG(1) << "GEMM autotuning mode: using all " << all_instances.size() << " instances";
            break;
        }
        case 2: {  // Hybrid mode: heuristic filtering + broader search
            auto filtered_instances = HeuristicFilter(all_instances, gemm_problem);
            final_instances = filtered_instances.empty() ? all_instances : filtered_instances;
            VLOG(1) << "GEMM hybrid mode: using " << final_instances.size() 
                    << " instances (filtered from " << all_instances.size() << ")";
            break;
        }
    }

    // Validate and store instances
    num_instances_ = 0;
    for (const auto& instance : final_instances) {
        if (IsValidInstance(instance)) {
            instance_map_[gemm_problem.kind_][instance.GetInstanceName()] = instance;
            ++num_instances_;
        }
    }

    VLOG(1) << "Generated " << num_instances_ << " valid GEMM instances for " 
            << GetGemmKindName(gemm_problem.kind_) << " (mode " << FLAGS_FC_TUNING_MODE << ")";
}

int64_t GemmEmitter::GetNumInstances() const
{
    return num_instances_;
}

void GemmEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
}

}  // namespace flashck