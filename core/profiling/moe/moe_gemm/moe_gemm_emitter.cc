#include "core/profiling/moe/moe_gemm/moe_gemm_emitter.h"

#include <algorithm>
#include <filesystem>
#include <random>
#include "core/utils/common.h"

FC_DECLARE_int32(FC_TUNING_MODE);       // 0: heuristic, 1: autotuning, 2: hybrid
FC_DECLARE_bool(FC_ENABLE_BACKUP_JSON);     // Enable backup_config.json loading
FC_DECLARE_bool(FC_ENABLE_DEFAULT_JSON);    // Enable default_config.json loading  
FC_DECLARE_bool(FC_ENABLE_USER_JSON);       // Enable user_config.json loading
FC_DECLARE_string(FC_CONFIG_JSON_PATH);     // Base path for config files

namespace flashck {

bool MoeGemmEmitter::IsValidTile(const MoeGemmTileDesc& tile_desc, const MoeProblem& moe_problem) const
{
    // Validate all dual-stage tile parameters are positive
    if (tile_desc.m0_block_ <= 0 || tile_desc.n0_block_ <= 0 || tile_desc.k0_block_ <= 0 ||
        tile_desc.m1_block_ <= 0 || tile_desc.n1_block_ <= 0 || tile_desc.k1_block_ <= 0 ||
        tile_desc.m0_warp_ <= 0 || tile_desc.n0_warp_ <= 0 || tile_desc.k0_warp_ <= 0 ||
        tile_desc.m1_warp_ <= 0 || tile_desc.n1_warp_ <= 0 || tile_desc.k1_warp_ <= 0 ||
        tile_desc.m0_warp_tile_ <= 0 || tile_desc.n0_warp_tile_ <= 0 || tile_desc.k0_warp_tile_ <= 0 ||
        tile_desc.m1_warp_tile_ <= 0 || tile_desc.n1_warp_tile_ <= 0 || tile_desc.k1_warp_tile_ <= 0) {
        VLOG(3) << "Invalid MoE GEMM tile: negative or zero values not allowed in dual-stage configuration";
        return false;
    }

    // Validate thread block size doesn't exceed hardware limits for both stages
    const int total_threads_stage0 = tile_desc.m0_block_ * tile_desc.n0_block_;
    const int total_threads_stage1 = tile_desc.m1_block_ * tile_desc.n1_block_;
    if (total_threads_stage0 > 1024 || total_threads_stage1 > 1024) {
        VLOG(3) << "Invalid MoE GEMM tile: thread block size exceeds limit (1024). "
                << "Stage 0: " << total_threads_stage0 << ", Stage 1: " << total_threads_stage1;
        return false;
    }

    // Validate tile sizes don't exceed problem dimensions for both stages
    // Stage 0: Token-to-Intermediate
    if (tile_desc.m0_block_ > moe_problem.m_ || tile_desc.n0_block_ > moe_problem.intermediate_size_ || 
        tile_desc.k0_block_ > moe_problem.k_) {
        VLOG(3) << "Invalid MoE GEMM Stage 0 tile: block dims (" << tile_desc.m0_block_ << "," 
                << tile_desc.n0_block_ << "," << tile_desc.k0_block_
                << ") exceed problem dims (" << moe_problem.m_ << "," << moe_problem.intermediate_size_
                << "," << moe_problem.k_ << ")";
        return false;
    }

    // Stage 1: Intermediate-to-Output  
    if (tile_desc.m1_block_ > moe_problem.m_ || tile_desc.n1_block_ > moe_problem.n_ || 
        tile_desc.k1_block_ > moe_problem.intermediate_size_) {
        VLOG(3) << "Invalid MoE GEMM Stage 1 tile: block dims (" << tile_desc.m1_block_ << "," 
                << tile_desc.n1_block_ << "," << tile_desc.k1_block_
                << ") exceed problem dims (" << moe_problem.m_ << "," << moe_problem.n_ 
                << "," << moe_problem.intermediate_size_ << ")";
        return false;
    }

    // Validate warp combinations are allowed for both stages
    std::tuple<int, int, int> warp_tuple_stage0 = std::make_tuple(tile_desc.m0_warp_, tile_desc.n0_warp_, tile_desc.k0_warp_);
    std::tuple<int, int, int> warp_tuple_stage1 = std::make_tuple(tile_desc.m1_warp_, tile_desc.n1_warp_, tile_desc.k1_warp_);
    
    if (std::find(g_moe_gemm_allowed_warp_combinations.begin(), g_moe_gemm_allowed_warp_combinations.end(), 
                  warp_tuple_stage0) == g_moe_gemm_allowed_warp_combinations.end()) {
        VLOG(3) << "Invalid MoE GEMM Stage 0 warp combination: (" << tile_desc.m0_warp_ << "," 
                << tile_desc.n0_warp_ << "," << tile_desc.k0_warp_ << ")";
        return false;
    }
    
    if (std::find(g_moe_gemm_allowed_warp_combinations.begin(), g_moe_gemm_allowed_warp_combinations.end(), 
                  warp_tuple_stage1) == g_moe_gemm_allowed_warp_combinations.end()) {
        VLOG(3) << "Invalid MoE GEMM Stage 1 warp combination: (" << tile_desc.m1_warp_ << "," 
                << tile_desc.n1_warp_ << "," << tile_desc.k1_warp_ << ")";
        return false;
    }

    // Validate dimension alignment for both stages: block dims must be divisible by warp*warp_tile dims
    // Stage 0 validation
    if (tile_desc.m0_block_ % (tile_desc.m0_warp_ * tile_desc.m0_warp_tile_) != 0) {
        VLOG(3) << "MoE GEMM Stage 0 dimension alignment failed: m0_block(" << tile_desc.m0_block_ 
                << ") % [" << tile_desc.m0_warp_ << "x" << tile_desc.m0_warp_tile_ << "] != 0";
        return false;
    }
    if (tile_desc.n0_block_ % (tile_desc.n0_warp_ * tile_desc.n0_warp_tile_) != 0) {
        VLOG(3) << "MoE GEMM Stage 0 dimension alignment failed: n0_block(" << tile_desc.n0_block_ 
                << ") % [" << tile_desc.n0_warp_ << "x" << tile_desc.n0_warp_tile_ << "] != 0";
        return false;
    }
    if (tile_desc.k0_block_ % (tile_desc.k0_warp_ * tile_desc.k0_warp_tile_) != 0) {
        VLOG(3) << "MoE GEMM Stage 0 dimension alignment failed: k0_block(" << tile_desc.k0_block_ 
                << ") % [" << tile_desc.k0_warp_ << "x" << tile_desc.k0_warp_tile_ << "] != 0";
        return false;
    }

    // Stage 1 validation
    if (tile_desc.m1_block_ % (tile_desc.m1_warp_ * tile_desc.m1_warp_tile_) != 0) {
        VLOG(3) << "MoE GEMM Stage 1 dimension alignment failed: m1_block(" << tile_desc.m1_block_ 
                << ") % [" << tile_desc.m1_warp_ << "x" << tile_desc.m1_warp_tile_ << "] != 0";
        return false;
    }
    if (tile_desc.n1_block_ % (tile_desc.n1_warp_ * tile_desc.n1_warp_tile_) != 0) {
        VLOG(3) << "MoE GEMM Stage 1 dimension alignment failed: n1_block(" << tile_desc.n1_block_ 
                << ") % [" << tile_desc.n1_warp_ << "x" << tile_desc.n1_warp_tile_ << "] != 0";
        return false;
    }
    if (tile_desc.k1_block_ % (tile_desc.k1_warp_ * tile_desc.k1_warp_tile_) != 0) {
        VLOG(3) << "MoE GEMM Stage 1 dimension alignment failed: k1_block(" << tile_desc.k1_block_ 
                << ") % [" << tile_desc.k1_warp_ << "x" << tile_desc.k1_warp_tile_ << "] != 0";
        return false;
    }

    // LDS capacity verification for both stages
    size_t stage0_input_size = tile_desc.m0_block_ * tile_desc.k0_block_ * SizeOf(moe_problem.input_dtype_);
    size_t stage0_gate_weight_size = tile_desc.n0_block_ * tile_desc.k0_block_ * SizeOf(moe_problem.weight_dtype_);
    size_t stage0_intermediate_size = tile_desc.m0_block_ * tile_desc.n0_block_ * SizeOf(moe_problem.intermediate_dtype_);
    
    size_t stage1_intermediate_size = tile_desc.m1_block_ * tile_desc.k1_block_ * SizeOf(moe_problem.intermediate_dtype_);
    size_t stage1_down_weight_size = tile_desc.n1_block_ * tile_desc.k1_block_ * SizeOf(moe_problem.weight_dtype_);
    size_t stage1_output_size = tile_desc.m1_block_ * tile_desc.n1_block_ * SizeOf(moe_problem.output_dtype_);

    size_t total_stage0_in_lds = stage0_input_size + stage0_gate_weight_size + stage0_intermediate_size;
    size_t total_stage1_in_lds = stage1_intermediate_size + stage1_down_weight_size + stage1_output_size;
    size_t max_stage_lds = std::max(total_stage0_in_lds, total_stage1_in_lds);

    size_t max_tile_size = (1 << 16); // 64KB

    if (max_stage_lds > max_tile_size) {
        VLOG(3) << "MoE GEMM LDS capacity exceeded: Max stage " << max_stage_lds << "B (" << (max_stage_lds / 1024.0) << "KB) > "
                << "maximum allowed " << max_tile_size << "B (" << (max_tile_size / 1024) << "KB). Breakdown:\n"
                << "- Stage 0 total: " << total_stage0_in_lds << "B (Input: " << stage0_input_size 
                << "B, Gate: " << stage0_gate_weight_size << "B, Intermediate: " << stage0_intermediate_size << "B)\n"
                << "- Stage 1 total: " << total_stage1_in_lds << "B (Intermediate: " << stage1_intermediate_size 
                << "B, Down: " << stage1_down_weight_size << "B, Output: " << stage1_output_size << "B)";
        return false;
    }

    // MoE-specific validations
    if (!IsValidExpertRouting(tile_desc, moe_problem) || 
        !IsValidLoadBalancing(tile_desc, moe_problem) ||
        !IsValidInterStageBandwidth(tile_desc, moe_problem)) {
        return false;
    }

    // Warp tile combination validation for dual-stage MoE
    std::string warp_tile_key = Sprintf("{}_{}_{}", DataTypeToString(moe_problem.input_dtype_),
                                         DataTypeToString(moe_problem.weight_dtype_),
                                         DataTypeToString(moe_problem.output_dtype_));
    
    std::array<int64_t, 6> current_combination = {tile_desc.m0_warp_tile_, tile_desc.n0_warp_tile_, tile_desc.k0_warp_tile_,
                                                   tile_desc.m1_warp_tile_, tile_desc.n1_warp_tile_, tile_desc.k1_warp_tile_};

    std::string gpu_name = GetDeviceName();

    auto gpu_warp_tile_key_it = g_moe_gemm_warp_tile_supported_combinations.find(gpu_name);
    if (gpu_warp_tile_key_it == g_moe_gemm_warp_tile_supported_combinations.end()) {
        VLOG(3) << "Trait: [MoE GEMM], No valid warp tile combinations found for " << gpu_name << "/" << warp_tile_key << ", skip this check.";
        return false;
    }
    
    const auto& gpu_warp_tile_key = gpu_warp_tile_key_it->second;
    auto allowed_combinations_it = gpu_warp_tile_key.find(warp_tile_key);
    if (allowed_combinations_it == gpu_warp_tile_key.end()) {
        VLOG(3) << "Trait: [MoE GEMM], No valid warp tile combinations found for " << gpu_name << "/" << warp_tile_key << ", skip this check.";
        return false;
    }
    
    const auto& allowed_combinations = allowed_combinations_it->second;
    if (std::find(allowed_combinations.begin(), allowed_combinations.end(), current_combination) == allowed_combinations.end()) {
        VLOG(3) << "Trait: [MoE GEMM], Invalid dual-stage warp combination: [" 
                << current_combination[0] << ", " << current_combination[1] << ", " << current_combination[2] << ", "
                << current_combination[3] << ", " << current_combination[4] << ", " << current_combination[5]
                << "] not in allowed list. Valid combinations for data type '" << warp_tile_key << "': ";
        for (const auto& comb : allowed_combinations) {
            VLOG(3) << "  [" << comb[0] << ", " << comb[1] << ", " << comb[2] << ", " 
                    << comb[3] << ", " << comb[4] << ", " << comb[5] << "]";
        }
        return false;
    }

    return true;
}

bool MoeGemmEmitter::IsValidExpertRouting(const MoeGemmTileDesc& tile_desc, const MoeProblem& moe_problem) const
{
    // Expert routing efficiency validation
    // Ensure tile sizes support efficient expert selection and routing
    
    // Check if tile sizes allow for good expert utilization
    int64_t tokens_per_tile = tile_desc.m0_block_;
    int64_t experts_per_token = moe_problem.top_k_;
    
    // Validate that we have enough work per tile to justify expert routing overhead
    if (tokens_per_tile * experts_per_token < 64) {
        VLOG(3) << "MoE GEMM: Insufficient work per tile for efficient expert routing. "
                << "Tokens: " << tokens_per_tile << ", Experts per token: " << experts_per_token
                << ", Total work: " << (tokens_per_tile * experts_per_token) << " < 64";
        return false;
    }

    // Validate that tile sizes are compatible with expert selection patterns
    if (tile_desc.n0_block_ % moe_problem.num_experts_ != 0 && moe_problem.num_experts_ % tile_desc.n0_block_ != 0) {
        VLOG(3) << "MoE GEMM: Expert routing incompatible. Tile n0_block (" << tile_desc.n0_block_ 
                << ") and num_experts (" << moe_problem.num_experts_ << ") don't have divisibility relationship";
        return false;
    }

    return true;
}

bool MoeGemmEmitter::IsValidLoadBalancing(const MoeGemmTileDesc& tile_desc, const MoeProblem& moe_problem) const
{
    // Load balancing validation across experts
    
    // Calculate expected load per expert
    double tokens_per_expert = static_cast<double>(moe_problem.m_ * moe_problem.top_k_) / moe_problem.num_experts_;
    
    // Validate that tile sizes allow for reasonable load balancing
    double tile_tokens_per_expert = static_cast<double>(tile_desc.m0_block_ * moe_problem.top_k_) / moe_problem.num_experts_;
    
    // Check for extreme load imbalance (more than 4x variance)
    if (tile_tokens_per_expert > 4.0 * tokens_per_expert || tile_tokens_per_expert < 0.25 * tokens_per_expert) {
        VLOG(3) << "MoE GEMM: Load balancing issue. Expected tokens per expert: " << tokens_per_expert
                << ", Tile tokens per expert: " << tile_tokens_per_expert;
        return false;
    }

    // Validate capacity factor constraints
    if (moe_problem.capacity_factor_ > 0.0) {
        double max_tokens_per_expert = tokens_per_expert * moe_problem.capacity_factor_;
        if (tile_tokens_per_expert > max_tokens_per_expert) {
            VLOG(3) << "MoE GEMM: Capacity factor violation. Tile tokens per expert: " << tile_tokens_per_expert
                    << " > Max allowed: " << max_tokens_per_expert;
            return false;
        }
    }

    return true;
}

bool MoeGemmEmitter::IsValidInterStageBandwidth(const MoeGemmTileDesc& tile_desc, const MoeProblem& moe_problem) const
{
    // Inter-stage bandwidth validation
    
    // Calculate data movement between stages
    size_t stage0_output_size = tile_desc.m0_block_ * tile_desc.n0_block_ * SizeOf(moe_problem.intermediate_dtype_);
    size_t stage1_input_size = tile_desc.m1_block_ * tile_desc.k1_block_ * SizeOf(moe_problem.intermediate_dtype_);
    
    // Validate that intermediate data sizes are compatible
    if (stage0_output_size != stage1_input_size) {
        // Check if dimensions are compatible through reshaping
        if (tile_desc.m0_block_ * tile_desc.n0_block_ != tile_desc.m1_block_ * tile_desc.k1_block_) {
            VLOG(3) << "MoE GEMM: Inter-stage dimension mismatch. Stage 0 output: " 
                    << tile_desc.m0_block_ << "x" << tile_desc.n0_block_ 
                    << ", Stage 1 input: " << tile_desc.m1_block_ << "x" << tile_desc.k1_block_;
            return false;
        }
    }

    // Validate bandwidth constraints (simplified model)
    size_t max_inter_stage_bandwidth = (1ULL << 20); // 1MB per transfer
    if (stage0_output_size > max_inter_stage_bandwidth) {
        VLOG(3) << "MoE GEMM: Inter-stage bandwidth exceeded. Transfer size: " << stage0_output_size
                << "B > Max: " << max_inter_stage_bandwidth << "B";
        return false;
    }

    return true;
}

bool MoeGemmEmitter::IsValidMoeCombination(const std::string& activation, const std::string& routing_method, 
                                          const std::string& expert_parallel_mode)
{
    // Check if the current MoE combination is valid
    return std::find(g_moe_gemm_unsupported_combinations.begin(), g_moe_gemm_unsupported_combinations.end(),
                    std::make_tuple(activation, routing_method, expert_parallel_mode)) == g_moe_gemm_unsupported_combinations.end();
}

bool MoeGemmEmitter::IsValidInstance(const MoeGemmCodeGen& instance)
{
    return IsValidTile(instance.tile_desc_, instance.problem_) && 
           IsValidMoeCombination(GetActivationEnumShortName(instance.act_), 
                               "topk_routing",  // Default routing method
                               "data_parallel"); // Default parallelism
}

std::vector<MoeGemmCodeGen> MoeGemmEmitter::HeuristicFilter(const std::vector<MoeGemmCodeGen>& instances, 
                                                           const MoeProblem& moe_problem)
{
    if (instances.empty()) {
        return {};
    }

    std::vector<MoeGemmCodeGen> filtered_instances;
    
    // Score and rank instances based on MoE-specific performance heuristics
    std::vector<std::pair<double, size_t>> scored_instances;
    
    for (size_t i = 0; i < instances.size(); ++i) {
        const auto& tile_desc = instances[i].tile_desc_;
        double score = 0.0;
        
        // 1. Expert routing efficiency (favor configurations that balance expert load well)
        double tokens_per_expert = static_cast<double>(moe_problem.m_ * moe_problem.top_k_) / moe_problem.num_experts_;
        double tile_tokens_per_expert = static_cast<double>(tile_desc.m0_block_ * moe_problem.top_k_) / moe_problem.num_experts_;
        double load_balance_ratio = std::min(tokens_per_expert / tile_tokens_per_expert, tile_tokens_per_expert / tokens_per_expert);
        score += 0.3 * load_balance_ratio;
        
        // 2. Dual-stage memory throughput efficiency
        int64_t stage0_work = tile_desc.m0_block_ * tile_desc.n0_block_ * tile_desc.k0_block_;
        int64_t stage1_work = tile_desc.m1_block_ * tile_desc.n1_block_ * tile_desc.k1_block_;
        int64_t total_work = stage0_work + stage1_work;
        
        if (total_work >= 8192 && total_work <= 65536) {  // Sweet spot for MoE workloads
            score += 0.25;
        }
        
        // 3. Register efficiency for both stages
        int64_t reg_estimate_stage0 = tile_desc.m0_warp_tile_ * tile_desc.n0_warp_tile_ * tile_desc.k0_warp_tile_;
        int64_t reg_estimate_stage1 = tile_desc.m1_warp_tile_ * tile_desc.n1_warp_tile_ * tile_desc.k1_warp_tile_;
        
        if (reg_estimate_stage0 >= 8 && reg_estimate_stage0 <= 128) score += 0.1;
        if (reg_estimate_stage1 >= 8 && reg_estimate_stage1 <= 128) score += 0.1;
        
        // 4. Problem fit analysis for MoE dimensions
        if (moe_problem.m_ % tile_desc.m0_block_ == 0) score += 0.05;
        if (moe_problem.intermediate_size_ % tile_desc.n0_block_ == 0) score += 0.05;
        if (moe_problem.n_ % tile_desc.n1_block_ == 0) score += 0.05;
        if (moe_problem.k_ % tile_desc.k0_block_ == 0) score += 0.05;
        
        // 5. Activation function fusion efficiency
        if (instances[i].act_ == GetActivationEnumFromString("swiglu") ||
            instances[i].act_ == GetActivationEnumFromString("gelu")) {
            score += 0.05;  // Favor activations that fuse well
        }
        
        scored_instances.emplace_back(score, i);
    }
    
    // Sort by score (highest first)
    std::sort(scored_instances.begin(), scored_instances.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Select top candidates (limit to reasonable number for heuristic mode)
    size_t max_candidates = std::min(static_cast<size_t>(15), instances.size());
    filtered_instances.reserve(max_candidates);
    
    for (size_t i = 0; i < max_candidates; ++i) {
        filtered_instances.push_back(instances[scored_instances[i].second]);
    }
    
    VLOG(2) << "MoE GEMM heuristic filter: reduced " << instances.size() 
            << " instances to " << filtered_instances.size() << " candidates";
    
    return filtered_instances;
}

// Generate all possible MoeGemmCodeGen instances from a MoeGemmConfig
std::vector<MoeGemmCodeGen> MoeGemmEmitter::CreateInstanceForConfig(const MoeGemmConfig& config, const MoeProblem& moe_problem) {
    std::vector<MoeGemmCodeGen> result;

    std::vector<std::vector<int64_t>> all_lists = {
        // Stage 0 BlockConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.m0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.n0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.k0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // Stage 1 BlockConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.m1.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.n1.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_tile.k1.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // Stage 0 WarpConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_warps.m0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_warps.n0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_warps.k0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // Stage 1 WarpConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_warps.m1.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_warps.n1.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.block_warps.k1.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // Stage 0 WarpTileConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.warp_tile.m0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.warp_tile.n0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.warp_tile.k0.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // Stage 1 WarpTileConfig
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.warp_tile.m1.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.warp_tile.n1.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.tile_shape.warp_tile.k1.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // PaddingConfig (convert bool to int64_t)
        [&]{ std::vector<int64_t> v; for (auto x : config.padding.hidden_size.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        [&]{ std::vector<int64_t> v; for (auto x : config.padding.intermediate_size.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // pipeline
        [&]{ std::vector<int64_t> v; for (auto x : config.pipeline.interleave.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
        // launch
        [&]{ std::vector<int64_t> v; for (auto x : config.launch.min_block_per_cu.values) v.emplace_back(static_cast<int64_t>(x)); return v; }(),
    };

    CartesianProduct(all_lists, [&](const std::vector<int64_t>& vals) {
        size_t idx = 0;
        // Stage 0 BlockConfig
        int64_t m0_block = vals[idx++];
        int64_t n0_block = vals[idx++];
        int64_t k0_block = vals[idx++];
        // Stage 1 BlockConfig
        int64_t m1_block = vals[idx++];
        int64_t n1_block = vals[idx++];
        int64_t k1_block = vals[idx++];
        // Stage 0 WarpConfig
        int64_t m0_warp = vals[idx++];
        int64_t n0_warp = vals[idx++];
        int64_t k0_warp = vals[idx++];
        // Stage 1 WarpConfig
        int64_t m1_warp = vals[idx++];
        int64_t n1_warp = vals[idx++];
        int64_t k1_warp = vals[idx++];
        // Stage 0 WarpTileConfig
        int64_t m0_warp_tile = vals[idx++];
        int64_t n0_warp_tile = vals[idx++];
        int64_t k0_warp_tile = vals[idx++];
        // Stage 1 WarpTileConfig
        int64_t m1_warp_tile = vals[idx++];
        int64_t n1_warp_tile = vals[idx++];
        int64_t k1_warp_tile = vals[idx++];

        // padding
        bool is_pad_hidden_size = static_cast<bool>(vals[idx++]);
        bool is_pad_intermediate_size = static_cast<bool>(vals[idx++]);
        bool is_interleave = static_cast<bool>(vals[idx++]);

        // Launch config
        int64_t min_block_per_cu = vals[idx++];

        // Construct MoeGemmCodeGen
        MoeGemmCodeGen moe_gemm;
        moe_gemm.problem_ = moe_problem;
        moe_gemm.tile_desc_ = MoeGemmTileDesc{m0_block, n0_block, k0_block, m1_block, n1_block, k1_block,
                                             m0_warp, n0_warp, k0_warp, m1_warp, n1_warp, k1_warp,
                                             m0_warp_tile, n0_warp_tile, k0_warp_tile, m1_warp_tile, n1_warp_tile, k1_warp_tile};

        moe_gemm.is_pad_hidden_size_ = is_pad_hidden_size;
        moe_gemm.is_pad_intermediate_size_ = is_pad_intermediate_size;
        moe_gemm.is_interleave_ = is_interleave;
        moe_gemm.min_block_per_cu_ = min_block_per_cu;
        result.push_back(moe_gemm);
    });
    return result;
}

void MoeGemmEmitter::GenerateInstances(MoeProblem& moe_problem)
{
    // Validate tuning mode
    FC_ENFORCE_EQ(FLAGS_FC_TUNING_MODE >= 0 && FLAGS_FC_TUNING_MODE <= 2, true,
                  Unavailable("Invalid FC_TUNING_MODE: {}. Valid values: 0(heuristic), 1(autotuning), 2(hybrid)", 
                             FLAGS_FC_TUNING_MODE));

    // Check if instances already exist for this MoE GEMM kind
    if (instance_map_.find(moe_problem.kind_) != instance_map_.end() && 
        !instance_map_[moe_problem.kind_].empty()) {
        VLOG(2) << "MoE GEMM instances already generated for kind: " << GetMoeGemmKindName(moe_problem.kind_);
        return;
    }

    std::vector<MoeGemmCodeGen> all_instances;

    // Configuration loading based on enabled flags
    auto base_json_path = std::filesystem::path(FLAGS_FC_CONFIG_JSON_PATH) / GetMoeGemmKindName(moe_problem.kind_);

    // Load backup configurations (pre-validated single configs)
    if (FLAGS_FC_ENABLE_BACKUP_JSON) {
        try {
            std::filesystem::path backup_path = base_json_path / "backup_config.json";
            if (std::filesystem::exists(backup_path)) {
                auto backup_configs = LoadConfigJson<std::vector<MoeGemmConfig>>(backup_path);
                for (const auto& config : backup_configs) {
                    auto backup_instances = CreateInstanceForConfig(config, moe_problem);
                    all_instances.insert(all_instances.end(), backup_instances.begin(), backup_instances.end());
                }
                VLOG(2) << "Loaded " << backup_configs.size() << " MoE GEMM backup configurations";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load MoE GEMM backup config: " << e.what();
        }
    }

    // Load default configurations (parameter ranges)
    if (FLAGS_FC_ENABLE_DEFAULT_JSON) {
        try {
            std::filesystem::path default_path = base_json_path / "default_config.json";
            if (std::filesystem::exists(default_path)) {
                auto default_config = LoadConfigJson<MoeGemmConfig>(default_path);
                auto default_instances = CreateInstanceForConfig(default_config, moe_problem);
                all_instances.insert(all_instances.end(), default_instances.begin(), default_instances.end());
                VLOG(2) << "Loaded MoE GEMM default configuration with " << default_instances.size() << " instances";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load MoE GEMM default config: " << e.what();
        }
    }

    // Load user configurations (custom parameter ranges)
    if (FLAGS_FC_ENABLE_USER_JSON) {
        try {
            std::filesystem::path user_path = base_json_path / "user_config.json";
            if (std::filesystem::exists(user_path)) {
                auto user_config = LoadConfigJson<MoeGemmConfig>(user_path);
                auto user_instances = CreateInstanceForConfig(user_config, moe_problem);
                all_instances.insert(all_instances.end(), user_instances.begin(), user_instances.end());
                VLOG(2) << "Loaded MoE GEMM user configuration with " << user_instances.size() << " instances";
            }
        } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to load MoE GEMM user config: " << e.what();
        }
    }

    // Apply mode-specific processing
    std::vector<MoeGemmCodeGen> final_instances;
    
    switch (FLAGS_FC_TUNING_MODE) {
        case 0: {  // Heuristic mode: filter + random selection for fast execution
            auto filtered_instances = HeuristicFilter(all_instances, moe_problem);
            if (!filtered_instances.empty()) {
                // Randomly select one optimal configuration for fast execution
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dis(0, filtered_instances.size() - 1);
                final_instances.push_back(filtered_instances[dis(gen)]);
                VLOG(1) << "MoE GEMM heuristic mode: selected 1 instance from " 
                        << filtered_instances.size() << " filtered candidates";
            }
            break;
        }
        case 1: {  // Autotuning mode: use all valid instances for comprehensive search
            final_instances = all_instances;
            VLOG(1) << "MoE GEMM autotuning mode: using all " << all_instances.size() << " instances";
            break;
        }
        case 2: {  // Hybrid mode: heuristic filtering + broader search
            auto filtered_instances = HeuristicFilter(all_instances, moe_problem);
            final_instances = filtered_instances.empty() ? all_instances : filtered_instances;
            VLOG(1) << "MoE GEMM hybrid mode: using " << final_instances.size() 
                    << " instances (filtered from " << all_instances.size() << ")";
            break;
        }
    }

    // Validate and store instances
    num_instances_ = 0;
    for (const auto& instance : final_instances) {
        if (IsValidInstance(instance)) {
            instance_map_[moe_problem.kind_][instance.GetInstanceName()] = instance;
            ++num_instances_;
        }
    }

    VLOG(1) << "Generated " << num_instances_ << " valid MoE GEMM instances for " 
            << GetMoeGemmKindName(moe_problem.kind_) << " (mode " << FLAGS_FC_TUNING_MODE << ")";
}

int64_t MoeGemmEmitter::GetNumInstances() const
{
    return num_instances_;
}

void MoeGemmEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
}

}  // namespace flashck
