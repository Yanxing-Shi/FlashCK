#include "core/profiling/tile/fmha/fmha_emitter.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

#include "core/utils/enforce.h"

FC_DECLARE_int32(FC_TUNING_MODE);  // Mode for FMHA operation: 0 - heuristic, 1 - autotuning, 2 - hybrid
FC_DECLARE_bool(FC_ENABLE_CONFIG_JSON);
FC_DECLARE_string(FC_CONFIG_JSON_PATH);
FC_DECLARE_int32(FC_ENABLE_JSON_MODE);

namespace flashck {

bool FmhaEmitter::IsValidTile(const FmhaFwdTileDesc& tile_desc, const FmhaProblem& fmha_problem)
{
    // All tile parameters must be positive
    if (tile_desc.m0_block_ <= 0 || tile_desc.n0_block_ <= 0 || tile_desc.k0_block_ <= 0 || tile_desc.k0_max_block_ <= 0 ||
        tile_desc.n1_block_ <= 0 || tile_desc.k1_block_ <= 0 ||
        tile_desc.m0_warp_ <= 0 || tile_desc.n0_warp_ <= 0 || tile_desc.k0_warp_ < 0 ||
        tile_desc.m1_warp_ <= 0 || tile_desc.n1_warp_ <= 0 || tile_desc.k1_warp_ < 0 ||
        tile_desc.m0_warp_tile_ <= 0 || tile_desc.n0_warp_tile_ <= 0 || tile_desc.k0_warp_tile_ <= 0 ||
        tile_desc.m1_warp_tile_ <= 0 || tile_desc.n1_warp_tile_ <= 0 || tile_desc.k1_warp_tile_ <= 0) {
        VLOG(3) << "Invalid FMHA tile descriptor: negative or zero values not allowed";
        return false;
    }

    // Cross-parameter constraints
    // 1. k0_block_ should not exceed k0_max_block_
    if (tile_desc.k0_block_ > tile_desc.k0_max_block_) {
        VLOG(3) << "Invalid FMHA tile descriptor: k0_block_ > k0_max_block_";
        return false;
    }
    // 2. Warp and warp tile sizes should not exceed block sizes
    if (tile_desc.m0_warp_ * tile_desc.m0_warp_tile_ > tile_desc.m0_block_ ||
        tile_desc.n0_warp_ * tile_desc.n0_warp_tile_ > tile_desc.n0_block_ ||
        tile_desc.k0_warp_ * tile_desc.k0_warp_tile_ > tile_desc.k0_block_ ||
        tile_desc.m1_warp_ * tile_desc.m1_warp_tile_ > tile_desc.m0_block_ ||
        tile_desc.n1_warp_ * tile_desc.n1_warp_tile_ > tile_desc.n1_block_ ||
        tile_desc.k1_warp_ * tile_desc.k1_warp_tile_ > tile_desc.k1_block_) {
        VLOG(3) << "Invalid FMHA tile descriptor: warp*warp_tile exceeds block size";
        return false;
    }
    // 3. All block sizes should be divisible by warp*warp_tile sizes
    if ((tile_desc.m0_block_ % (tile_desc.m0_warp_ * tile_desc.m0_warp_tile_) != 0) ||
        (tile_desc.n0_block_ % (tile_desc.n0_warp_ * tile_desc.n0_warp_tile_) != 0) ||
        (tile_desc.k0_block_ % (std::max<int64_t>(1, tile_desc.k0_warp_ * tile_desc.k0_warp_tile_)) != 0) ||
        (tile_desc.m0_block_ % (tile_desc.m1_warp_ * tile_desc.m1_warp_tile_) != 0) ||
        (tile_desc.n1_block_ % (tile_desc.n1_warp_ * tile_desc.n1_warp_tile_) != 0) ||
        (tile_desc.k1_block_ % (std::max<int64_t>(1, tile_desc.k1_warp_ * tile_desc.k1_warp_tile_)) != 0)) {
        VLOG(3) << "Invalid FMHA tile descriptor: block size not divisible by warp*warp_tile";
        return false;
    }

    // Validate against problem dimensions for Batch mode
    if (fmha_problem.mode_ == FmhaMode::Batch) {
        if (tile_desc.m0_block_ > fmha_problem.q_seq_len_ || tile_desc.n0_block_ > fmha_problem.kv_seq_len_ ||
            tile_desc.n1_block_ > fmha_problem.v_head_dim_ || tile_desc.k0_max_block_ > fmha_problem.qk_head_dim_) {
            VLOG(3) << "Invalid FMHA tile descriptor: tile dimensions exceed problem dimensions";
            return false;
        }
    }

    return true;
}

bool FmhaEmitter::IsValidInstance(const FmhaFwdCodeGen& instance)
{
    return IsValidTile(instance.tile_desc_, instance.problem_);
} 


// std::vector<FmhaFwdTileDesc> FmhaEmitter::HeuristicFilter(const std::vector<FmhaFwdTileDesc>& fmha_tile_desc,
//                                                        const FmhaProblem&               fmha_problem) const
// {
// }

// std::vector<FmhaFwdAppendKVTileDesc> FmhaEmitter::HeuristicFilter(const std::vector<FmhaFwdAppendKVTileDesc>& fmha_tile_desc,
//                                                                const FmhaProblem& fmha_problem) const
// {   
// }

// std::vector<FmhaFwdSplitKVCombineTileDesc>
// FmhaEmitter::HeuristicFilter(const std::vector<FmhaFwdSplitKVCombineTileDesc>& fmha_tile_desc,
//                              const FmhaProblem&                             fmha_problem) const
// {
// }


void FmhaEmitter::GenerateInstances(FmhaProblem& fmha_problem)
{
    FC_ENFORCE_EQ(FLAGS_FC_TUNING_MODE == 0 || FLAGS_FC_TUNING_MODE == 1 || FLAGS_FC_TUNING_MODE == 2,
                  true,
                  Unavailable("Unsupported mode: {}, valid modes are 0 (heuristic), 1 (autotuning), 2 (hybrid)", FLAGS_FC_TUNING_MODE));


    // Check if instances already exist for this FMHA kind
    if (instance_map_.find(fmha_problem.kind_) != instance_map_.end() && !instance_map_[fmha_problem.kind_].empty()) {
        VLOG(2) << "Instances already generated for FMHA kind: " << GetFmhaKindName(fmha_problem.kind_);
        return;
    }

    // Load legacy GEMM configuration if available
    std::vector<FmhaFwdCodeGen> fmha_instances;
    if (FLAGS_FC_ENABLE_CONFIG_JSON) {
        auto base_json_path = std::filesystem::path(FLAGS_FC_CONFIG_JSON_PATH) / GetFmhaKindName(fmha_problem.kind_);
        if(FLAGS_FC_ENABLE_JSON_MODE == 0) {
            std::filesystem::path json_path = base_json_path / "default_config.json";
            FmhaFwdConfig config = LoadConfigJson<FmhaFwdConfig>(json_path);
            fmha_instances = GenerateFmhaInstances(config, fmha_problem);
        } else if (FLAGS_FC_ENABLE_JSON_MODE == 1) {
            std::filesystem::path json_path = base_json_path / "user_config.json";
            FmhaFwdConfig config = LoadConfigJson<FmhaFwdConfig>(json_path);
            fmha_instances = GenerateFmhaInstances(config, fmha_problem);
        } else if (FLAGS_FC_ENABLE_JSON_MODE == 2) {
            std::filesystem::path default_json_path = base_json_path / "default_config.json";
            FmhaFwdConfig default_config = LoadConfigJson<FmhaFwdConfig>(default_json_path);
            auto gemm_default_instances = GenerateFmhaInstances(default_config, fmha_problem);

            std::filesystem::path user_json_path = base_json_path / "user_config.json";
            FmhaFwdConfig user_config = LoadConfigJson<FmhaFwdConfig>(user_json_path);
            auto gemm_user_instances = GenerateFmhaInstances(user_config, fmha_problem);

            fmha_instances.insert(fmha_instances.end(), gemm_default_instances.begin(), gemm_default_instances.end());
            fmha_instances.insert(fmha_instances.end(), gemm_user_instances.begin(), gemm_user_instances.end());
        }
    else{
            LOG(WARNING)<< "FC_ENABLE_JSON_MODE is set to an unsupported value: " << FLAGS_FC_ENABLE_JSON_MODE;
        }
    } else {
        LOG(WARNING)<< "FC_ENABLE_CONFIG_JSON is not enabled";
    }

    for (const auto& config : g_backup_fmha_config) {
        FmhaFwdCodeGen fmha;

        fmha.problem_ = fmha_problem;

        fmha.tile_desc_ = FmhaFwdTileDesc{
            config.tile_config_.block_.m0_.values_[0][0],
            config.tile_config_.block_.n0_.values_[0][0],
            config.tile_config_.block_.k0_.values_[0][0],
            config.tile_config_.block_.k0_max_.values_[0][0],
            config.tile_config_.block_.n1_.values_[0][0],
            config.tile_config_.block_.k1_.values_[0][0],
            config.tile_config_.warp_.m0_.values_[0][0],
            config.tile_config_.warp_.n0_.values_[0][0],
            config.tile_config_.warp_.k0_.values_[0][0],
            config.tile_config_.warp_.m1_.values_[0][0],
            config.tile_config_.warp_.n1_.values_[0][0],
            config.tile_config_.warp_.k1_.values_[0][0],
            config.tile_config_.warp_tile_.m0_.values_[0][0],
            config.tile_config_.warp_tile_.n0_.values_[0][0],
            config.tile_config_.warp_tile_.k0_.values_[0][0],
            config.tile_config_.warp_tile_.m1_.values_[0][0],
            config.tile_config_.warp_tile_.n1_.values_[0][0],
            config.tile_config_.warp_tile_.k1_.values_[0][0]
        };

        fmha.is_pad_q_seq_len_ = config.padding_.s_.values_[0];
        fmha.is_pad_kv_seq_len_ = config.padding_.sk_.values_[0];
        fmha.is_pad_qk_head_dim_ = config.padding_.d_.values_[0];
        fmha.is_pad_v_head_dim_ = config.padding_.dv_.values_[0];

        fmha.min_block_per_cu_ = config.launch_.min_block_per_cu_.values_[0][0];

        fmha.pipeline_ = GetBlockFmhaPipelineEnumFromString(config.pipeline_.values_[0]);

        fmha_instances.push_back(fmha);
    }

    // check instances
    std::vector<FmhaFwdCodeGen> valid_fmha_instances;
    for (const auto& fmha_instance : fmha_instances) {
        if (IsValidInstance(fmha_instance)) {
            valid_fmha_instances.push_back(fmha_instance);
        }
    }

    switch (FLAGS_FC_TUNING_MODE) {
        case 0: {
            // Heuristic mode
        }
        case 1: {
            VLOG(1) << "Generating instances using autotuning mode for FMHA kind: "
                    << GetFmhaKindName(fmha_problem.kind_);
            break;
        }
        case 2: {
            VLOG(1) << "Generating instances using hybrid mode for FMHA kind: " << GetFmhaKindName(fmha_problem.kind_);
            break;
        }
        default:
            FC_THROW(Unavailable("Invalid mode:{} ", FLAGS_FC_TUNING_MODE));
    }


    if (valid_fmha_instances.empty()) {
        FC_THROW(Unavailable("No valid FMHA instances found for FMHA problem"));
    }

    // Generate instances
    auto& kind_instance_map = instance_map_[fmha_problem.kind_];
    int64_t                             generated_count   = 0;

    for (const auto& instance : valid_fmha_instances) {
        try {
            std::string instance_name = instance.GetInstanceName();

            // Avoid duplicates
            if (kind_instance_map.find(instance_name) == kind_instance_map.end()) {
                kind_instance_map[instance_name] = std::move(instance);
                generated_count++;
                VLOG(2) << "Generated FMHA instance: " << instance_name;
            }
            else {
                VLOG(3) << "Skipped duplicate FMHA instance: " << instance_name;
            }
        }
        catch (const std::exception& e) {
            LOG(WARNING) << "Failed to create FMHA codegen for instance: " << instance.GetInstanceName()
                         << ", error: " << e.what();
        }
    }

    num_instances_ += generated_count;
    VLOG(1) << "Generated " << generated_count << " FMHA instances for kind: " << GetFmhaKindName(fmha_problem.kind_)
            << " (total: " << num_instances_ << ")";
}

void FmhaEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
}

}  // namespace flashck
