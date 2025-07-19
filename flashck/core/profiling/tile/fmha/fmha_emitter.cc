#include "flashck/core/profiling/tile/fmha/fmha_emitter.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

#include "flashck/core/utils/enforce.h"

FC_DECLARE_int32(FC_TUNING_MODE);  // Mode for FMHA operation: 0 - heuristic, 1 - autotuning, 2 - hybrid

namespace flashck {

bool FmhaEmitter::IsValidTile(const FmhaTileDesc& tile_desc, const FmhaProblem& fmha_problem) const
{
    // Validate tile descriptor parameters
    if (tile_desc.bm0_ <= 0 || tile_desc.bn0_ <= 0 || tile_desc.bn1_ <= 0 || tile_desc.bk0_max_ <= 0) {
        VLOG(3) << "Invalid tile descriptor: negative or zero values not allowed";
        return false;
    }

    // Validate against problem dimensions for Batch mode
    if (fmha_problem.mode_ == FmhaMode::Batch) {
        if (tile_desc.bm0_ > fmha_problem.q_seq_len_ || tile_desc.bn0_ > fmha_problem.kv_seq_len_
            || tile_desc.bn1_ > fmha_problem.v_head_dim_ || tile_desc.bk0_max_ > fmha_problem.qk_head_dim_) {
            VLOG(3) << "Invalid tile descriptor: tile dimensions exceed problem dimensions";
            return false;
        }
    }

    return true;
}

bool FmhaEmitter::IsValidTile(const FmhaAppendKVTileDesc& tile_desc, const FmhaProblem& fmha_problem) const
{
    // Validate tile descriptor parameters
    if (tile_desc.bs_ <= 0 || tile_desc.bsk_ <= 0 || tile_desc.bd_ <= 0 || tile_desc.bdv_ <= 0) {
        VLOG(3) << "Invalid AppendKV tile descriptor: negative or zero values not allowed";
        return false;
    }

    // Validate against problem dimensions for Batch mode
    if (fmha_problem.mode_ == FmhaMode::Batch) {
        if (tile_desc.bs_ > fmha_problem.q_seq_len_ || tile_desc.bsk_ > fmha_problem.kv_seq_len_
            || tile_desc.bd_ > fmha_problem.qk_head_dim_ || tile_desc.bdv_ > fmha_problem.v_head_dim_) {
            VLOG(3) << "Invalid AppendKV tile descriptor: tile dimensions exceed problem dimensions";
            return false;
        }
    }

    return true;
}

bool FmhaEmitter::IsValidTile(const FmhaSplitKVCombineTileDesc& tile_desc, const FmhaProblem& fmha_problem) const
{
    // Validate tile descriptor parameters
    if (tile_desc.bm0_ <= 0 || tile_desc.bn1_ <= 0) {
        VLOG(3) << "Invalid SplitKV Combine tile descriptor: negative or zero values not allowed";
        return false;
    }

    // Validate against problem dimensions for Batch mode
    if (fmha_problem.mode_ == FmhaMode::Batch) {
        if (tile_desc.bm0_ > fmha_problem.q_seq_len_ || tile_desc.bn1_ > fmha_problem.v_head_dim_) {
            VLOG(3) << "Invalid SplitKV Combine tile descriptor: tile dimensions exceed problem dimensions";
            return false;
        }
    }

    return true;
}

std::vector<FmhaTileDesc> FmhaEmitter::HeuristicFilter(const std::vector<FmhaTileDesc>& fmha_tile_desc,
                                                       const FmhaProblem&               fmha_problem) const
{
    std::vector<FmhaTileDesc> filtered_tile_desc;

    for (const auto& tile_desc : fmha_tile_desc) {
        // Enhanced heuristic based on problem characteristics
        bool should_include = false;

        // For small problems, prefer smaller tile sizes
        if (fmha_problem.q_seq_len_ <= 128 && fmha_problem.kv_seq_len_ <= 128 && fmha_problem.qk_head_dim_ <= 64) {
            if (tile_desc.bm0_ <= 64 && tile_desc.bn0_ <= 64 && tile_desc.bk0_max_ <= 32) {
                should_include = true;
            }
        }
        // For medium problems, prefer balanced tiles
        else if (fmha_problem.q_seq_len_ <= 512 && fmha_problem.kv_seq_len_ <= 512
                 && fmha_problem.qk_head_dim_ <= 128) {
            if (tile_desc.bm0_ == 128 && tile_desc.bn0_ == 128 && tile_desc.bk0_max_ == 32) {
                should_include = true;
            }
        }
        // For large problems, prefer larger tiles
        else {
            if (tile_desc.bm0_ >= 128 && tile_desc.bn0_ >= 128 && tile_desc.bk0_max_ >= 32) {
                should_include = true;
            }
        }

        if (should_include && IsValidTile(tile_desc, fmha_problem)) {
            filtered_tile_desc.push_back(tile_desc);
            VLOG(2) << "Selected FMHA tile descriptor: " << tile_desc.GetInstanceName();
        }
        else {
            VLOG(3) << "Filtered out FMHA tile descriptor: " << tile_desc.GetInstanceName();
        }
    }

    // Ensure we have at least one tile descriptor
    if (filtered_tile_desc.empty() && !fmha_tile_desc.empty()) {
        LOG(WARNING) << "No FMHA tile descriptors passed heuristic filter, using first valid tile";
        for (const auto& tile_desc : fmha_tile_desc) {
            if (IsValidTile(tile_desc, fmha_problem)) {
                filtered_tile_desc.push_back(tile_desc);
                break;
            }
        }
    }

    return filtered_tile_desc;
}

std::vector<FmhaAppendKVTileDesc> FmhaEmitter::HeuristicFilter(const std::vector<FmhaAppendKVTileDesc>& fmha_tile_desc,
                                                               const FmhaProblem& fmha_problem) const
{
    std::vector<FmhaAppendKVTileDesc> filtered_tile_desc;

    for (const auto& tile_desc : fmha_tile_desc) {
        if (IsValidTile(tile_desc, fmha_problem)) {
            filtered_tile_desc.push_back(tile_desc);
            VLOG(2) << "Selected FMHA AppendKV tile descriptor: " << tile_desc.GetInstanceName();
        }
    }

    // Ensure we have at least one tile descriptor
    if (filtered_tile_desc.empty() && !fmha_tile_desc.empty()) {
        LOG(WARNING) << "No FMHA AppendKV tile descriptors passed filter, using first valid tile";
        for (const auto& tile_desc : fmha_tile_desc) {
            if (IsValidTile(tile_desc, fmha_problem)) {
                filtered_tile_desc.push_back(tile_desc);
                break;
            }
        }
    }

    return filtered_tile_desc;
}

std::vector<FmhaSplitKVCombineTileDesc>
FmhaEmitter::HeuristicFilter(const std::vector<FmhaSplitKVCombineTileDesc>& fmha_tile_desc,
                             const FmhaProblem&                             fmha_problem) const
{
    std::vector<FmhaSplitKVCombineTileDesc> filtered_tile_desc;

    for (const auto& tile_desc : fmha_tile_desc) {
        if (IsValidTile(tile_desc, fmha_problem)) {
            filtered_tile_desc.push_back(tile_desc);
            VLOG(2) << "Selected FMHA SplitKV Combine tile descriptor: " << tile_desc.GetInstanceName();
        }
    }

    // Ensure we have at least one tile descriptor
    if (filtered_tile_desc.empty() && !fmha_tile_desc.empty()) {
        LOG(WARNING) << "No FMHA SplitKV Combine tile descriptors passed filter, using first valid tile";
        for (const auto& tile_desc : fmha_tile_desc) {
            if (IsValidTile(tile_desc, fmha_problem)) {
                filtered_tile_desc.push_back(tile_desc);
                break;
            }
        }
    }

    return filtered_tile_desc;
}

void FmhaEmitter::ValidateMode(int mode) const
{
    FC_ENFORCE_EQ(mode == 0 || mode == 1 || mode == 2,
                  true,
                  Unavailable("Unsupported mode: {}, valid modes are 0 (heuristic), 1 (autotuning), 2 (hybrid)", mode));
}

void FmhaEmitter::GenerateInstances(FmhaProblem& fmha_problem)
{
    // Check if instances already exist for this FMHA kind
    if (instance_map_.find(fmha_problem.kind_) != instance_map_.end() && !instance_map_[fmha_problem.kind_].empty()) {
        VLOG(2) << "Instances already generated for FMHA kind: " << GetFmhaKindName(fmha_problem.kind_);
        return;
    }

    ValidateMode(FLAGS_FC_TUNING_MODE);

    // Generate instances based on FMHA operation kind
    switch (fmha_problem.kind_) {
        case FmhaKind::Fwd:
        case FmhaKind::FwdSplitKV: {
            std::vector<FmhaTileDesc> tile_descriptors;

            switch (FLAGS_FC_TUNING_MODE) {
                case 0:  // Heuristic mode
                    VLOG(1) << "Generating instances using heuristic mode for FMHA kind: "
                            << GetFmhaKindName(fmha_problem.kind_);
                    tile_descriptors = HeuristicFilter(g_fmha_tile_descriptions, fmha_problem);
                    break;

                case 1:  // Autotuning mode
                    VLOG(1) << "Generating instances using autotuning mode for FMHA kind: "
                            << GetFmhaKindName(fmha_problem.kind_);
                    for (const auto& tile_desc : g_fmha_tile_descriptions) {
                        if (IsValidTile(tile_desc, fmha_problem)) {
                            tile_descriptors.push_back(tile_desc);
                        }
                    }
                    break;

                case 2:  // Hybrid mode
                    VLOG(1) << "Generating instances using hybrid mode for FMHA kind: "
                            << GetFmhaKindName(fmha_problem.kind_);
                    tile_descriptors = HeuristicFilter(g_fmha_tile_descriptions, fmha_problem);
                    if (tile_descriptors.size() < 3) {  // Expand if too few options
                        for (const auto& tile_desc : g_fmha_tile_descriptions) {
                            if (IsValidTile(tile_desc, fmha_problem)) {
                                auto it =
                                    std::find_if(tile_descriptors.begin(),
                                                 tile_descriptors.end(),
                                                 [&tile_desc](const FmhaTileDesc& existing) {
                                                     return existing.GetInstanceName() == tile_desc.GetInstanceName();
                                                 });
                                if (it == tile_descriptors.end()) {
                                    tile_descriptors.push_back(tile_desc);
                                }
                            }
                        }
                    }
                    break;

                default:
                    FC_THROW(Unavailable("Invalid tuning mode: {}", FLAGS_FC_TUNING_MODE));
            }

            if (fmha_problem.kind_ == FmhaKind::Fwd) {
                CreateFwdInstances(fmha_problem, tile_descriptors);
            }
            else {
                CreateSplitKVInstances(fmha_problem, tile_descriptors);
            }
        } break;

        case FmhaKind::FwdAppendKV: {
            std::vector<FmhaAppendKVTileDesc> tile_descriptors;

            switch (FLAGS_FC_TUNING_MODE) {
                case 0:  // Heuristic mode
                    tile_descriptors = HeuristicFilter(g_fmha_appendkv_tile_descriptions, fmha_problem);
                    break;
                case 1:  // Autotuning mode
                    for (const auto& tile_desc : g_fmha_appendkv_tile_descriptions) {
                        if (IsValidTile(tile_desc, fmha_problem)) {
                            tile_descriptors.push_back(tile_desc);
                        }
                    }
                    break;
                case 2:  // Hybrid mode
                    tile_descriptors = HeuristicFilter(g_fmha_appendkv_tile_descriptions, fmha_problem);
                    break;
                default:
                    FC_THROW(Unavailable("Invalid tuning mode: {}", FLAGS_FC_TUNING_MODE));
            }

            CreateAppendKVInstances(fmha_problem, tile_descriptors);
        } break;

        case FmhaKind::FwdSplitKVCombine: {
            std::vector<FmhaSplitKVCombineTileDesc> tile_descriptors;

            switch (FLAGS_FC_TUNING_MODE) {
                case 0:  // Heuristic mode
                    tile_descriptors = HeuristicFilter(g_fmha_splitkv_combine_tile_descriptions, fmha_problem);
                    break;
                case 1:  // Autotuning mode
                    for (const auto& tile_desc : g_fmha_splitkv_combine_tile_descriptions) {
                        if (IsValidTile(tile_desc, fmha_problem)) {
                            tile_descriptors.push_back(tile_desc);
                        }
                    }
                    break;
                case 2:  // Hybrid mode
                    tile_descriptors = HeuristicFilter(g_fmha_splitkv_combine_tile_descriptions, fmha_problem);
                    break;
                default:
                    FC_THROW(Unavailable("Invalid tuning mode: {}", FLAGS_FC_TUNING_MODE));
            }

            CreateSplitKVCombineInstances(fmha_problem, tile_descriptors);
        } break;

        default:
            FC_THROW(InvalidArgument("Unsupported FMHA operation kind: {}", static_cast<int>(fmha_problem.kind_)));
    }
}

void FmhaEmitter::CreateFwdInstances(const FmhaProblem& fmha_problem, const std::vector<FmhaTileDesc>& tile_descriptors)
{
    std::map<std::string, std::string>& kind_instance_map = instance_map_[fmha_problem.kind_];
    int64_t                             generated_count   = 0;

    for (const auto& tile_desc : tile_descriptors) {
        try {
            FmhaFwdCodeGen operation     = GenFmhaFwdInstance(fmha_problem, tile_desc);
            std::string    instance_name = operation.GetInstanceName();

            // Avoid duplicates
            if (kind_instance_map.find(instance_name) == kind_instance_map.end()) {
                kind_instance_map[instance_name] = operation.Emit();
                generated_count++;
                VLOG(2) << "Generated FMHA Forward instance: " << instance_name;
            }
            else {
                VLOG(3) << "Skipped duplicate FMHA Forward instance: " << instance_name;
            }
        }
        catch (const std::exception& e) {
            LOG(WARNING) << "Failed to create FMHA Forward operation for tile: " << tile_desc.GetInstanceName()
                         << ", error: " << e.what();
        }
    }

    num_instances_ += generated_count;
    LOG(INFO) << "Generated " << generated_count
              << " FMHA Forward instances for kind: " << GetFmhaKindName(fmha_problem.kind_)
              << " (total: " << num_instances_ << ")";
}

void FmhaEmitter::CreateSplitKVInstances(const FmhaProblem&               fmha_problem,
                                         const std::vector<FmhaTileDesc>& tile_descriptors)
{
    std::map<std::string, std::string>& kind_instance_map = instance_map_[fmha_problem.kind_];
    int64_t                             generated_count   = 0;

    for (const auto& tile_desc : tile_descriptors) {
        try {
            FmhaFwdSplitKVCodeGen operation     = GenFmhaFwdSplitKVInstance(fmha_problem, tile_desc);
            std::string           instance_name = operation.GetInstanceName();

            // Avoid duplicates
            if (kind_instance_map.find(instance_name) == kind_instance_map.end()) {
                kind_instance_map[instance_name] = operation.Emit();
                generated_count++;
                VLOG(2) << "Generated FMHA SplitKV instance: " << instance_name;
            }
            else {
                VLOG(3) << "Skipped duplicate FMHA SplitKV instance: " << instance_name;
            }
        }
        catch (const std::exception& e) {
            LOG(WARNING) << "Failed to create FMHA SplitKV operation for tile: " << tile_desc.GetInstanceName()
                         << ", error: " << e.what();
        }
    }

    num_instances_ += generated_count;
    LOG(INFO) << "Generated " << generated_count
              << " FMHA SplitKV instances for kind: " << GetFmhaKindName(fmha_problem.kind_)
              << " (total: " << num_instances_ << ")";
}

void FmhaEmitter::CreateSplitKVCombineInstances(const FmhaProblem&                             fmha_problem,
                                                const std::vector<FmhaSplitKVCombineTileDesc>& tile_descriptors)
{
    std::map<std::string, std::string>& kind_instance_map = instance_map_[fmha_problem.kind_];
    int64_t                             generated_count   = 0;

    for (const auto& tile_desc : tile_descriptors) {
        try {
            FmhaFwdSplitKVCombineCodeGen operation     = GenFmhaFwdSplitKVCombineInstance(fmha_problem, tile_desc);
            std::string                  instance_name = operation.GetInstanceName();

            // Avoid duplicates
            if (kind_instance_map.find(instance_name) == kind_instance_map.end()) {
                kind_instance_map[instance_name] = operation.Emit();
                generated_count++;
                VLOG(2) << "Generated FMHA SplitKV Combine instance: " << instance_name;
            }
            else {
                VLOG(3) << "Skipped duplicate FMHA SplitKV Combine instance: " << instance_name;
            }
        }
        catch (const std::exception& e) {
            LOG(WARNING) << "Failed to create FMHA SplitKV Combine operation for tile: " << tile_desc.GetInstanceName()
                         << ", error: " << e.what();
        }
    }

    num_instances_ += generated_count;
    LOG(INFO) << "Generated " << generated_count
              << " FMHA SplitKV Combine instances for kind: " << GetFmhaKindName(fmha_problem.kind_)
              << " (total: " << num_instances_ << ")";
}

void FmhaEmitter::CreateAppendKVInstances(const FmhaProblem&                       fmha_problem,
                                          const std::vector<FmhaAppendKVTileDesc>& tile_descriptors)
{
    std::map<std::string, std::string>& kind_instance_map = instance_map_[fmha_problem.kind_];
    int64_t                             generated_count   = 0;

    for (const auto& tile_desc : tile_descriptors) {
        try {
            FmhaFwdAppendKVCodeGen operation     = GenFmhaFwdAppendKVInstance(fmha_problem, tile_desc);
            std::string            instance_name = operation.GetInstanceName();

            // Avoid duplicates
            if (kind_instance_map.find(instance_name) == kind_instance_map.end()) {
                kind_instance_map[instance_name] = operation.Emit();
                generated_count++;
                VLOG(2) << "Generated FMHA AppendKV instance: " << instance_name;
            }
            else {
                VLOG(3) << "Skipped duplicate FMHA AppendKV instance: " << instance_name;
            }
        }
        catch (const std::exception& e) {
            LOG(WARNING) << "Failed to create FMHA AppendKV operation for tile: " << tile_desc.GetInstanceName()
                         << ", error: " << e.what();
        }
    }

    num_instances_ += generated_count;
    LOG(INFO) << "Generated " << generated_count
              << " FMHA AppendKV instances for kind: " << GetFmhaKindName(fmha_problem.kind_)
              << " (total: " << num_instances_ << ")";
}

FmhaFwdCodeGen FmhaEmitter::GenFmhaFwdInstance(const FmhaProblem& fmha_problem, const FmhaTileDesc& tile_desc) const
{
    FmhaFwdCodeGen operation;

    // Set basic configuration
    operation.tile_desc_    = tile_desc;
    operation.kind_         = FmhaKind::Fwd;
    operation.mode_         = fmha_problem.mode_;
    operation.dtype_        = fmha_problem.dtype_;
    operation.mask_type_    = fmha_problem.mask_type_;
    operation.window_size_  = fmha_problem.window_size_;
    operation.bias_enum_    = fmha_problem.bias_enum_;
    operation.block_per_cu_ = -1;
    operation.pipeline_     = BlockFmhaPipelineEnum::QRKSVS_ASYNC;

    // Determine padding configuration
    auto padding_config = DetermineFwdPaddingConfig(fmha_problem, tile_desc, operation.mode_, operation.pipeline_);
    operation.is_pad_q_seq_len_    = padding_config.is_pad_q_seq_len;
    operation.is_pad_kv_seq_len_   = padding_config.is_pad_kv_seq_len;
    operation.is_pad_qk_head_dim_  = padding_config.is_pad_qk_head_dim;
    operation.is_pad_v_head_dim_   = padding_config.is_pad_v_head_dim;
    operation.is_pad_qkv_head_dim_ = padding_config.is_pad_qkv_head_dim;

    return operation;
}

FmhaFwdSplitKVCodeGen FmhaEmitter::GenFmhaFwdSplitKVInstance(const FmhaProblem&  fmha_problem,
                                                             const FmhaTileDesc& tile_desc) const
{
    FmhaFwdSplitKVCodeGen operation;

    // Set basic configuration
    operation.tile_desc_    = tile_desc;
    operation.kind_         = FmhaKind::FwdSplitKV;
    operation.mode_         = fmha_problem.mode_;
    operation.dtype_        = fmha_problem.dtype_;
    operation.mask_type_    = fmha_problem.mask_type_;
    operation.window_size_  = fmha_problem.window_size_;
    operation.bias_enum_    = fmha_problem.bias_enum_;
    operation.is_store_lse_ = fmha_problem.num_splits_ > 1;

    // Split-KV specific configurations
    operation.has_uneven_splits_ = fmha_problem.mode_ == FmhaMode::Batch ?
                                       fmha_problem.kv_seq_len_ % (fmha_problem.num_splits_ * tile_desc.bn0_) != 0 :
                                       true;

    operation.is_merge_num_head_groups_seq_len_q_ =
        fmha_problem.q_max_seq_len_ == 1 && fmha_problem.kv_num_heads_ < fmha_problem.q_num_heads_;

    // Determine padding configuration for Split-KV
    auto padding_config            = DetermineSplitKVPaddingConfig(fmha_problem, tile_desc, operation.mode_);
    operation.is_pad_q_seq_len_    = padding_config.is_pad_q_seq_len;
    operation.is_pad_kv_seq_len_   = fmha_problem.mode_ == FmhaMode::Batch ? operation.has_uneven_splits_ : true;
    operation.is_pad_qk_head_dim_  = padding_config.is_pad_qk_head_dim;
    operation.is_pad_v_head_dim_   = padding_config.is_pad_v_head_dim;
    operation.is_pad_qkv_head_dim_ = padding_config.is_pad_qkv_head_dim;

    return operation;
}

FmhaFwdSplitKVCombineCodeGen
FmhaEmitter::GenFmhaFwdSplitKVCombineInstance(const FmhaProblem&                fmha_problem,
                                              const FmhaSplitKVCombineTileDesc& tile_desc) const
{
    FmhaFwdSplitKVCombineCodeGen operation;

    // Set basic configuration
    operation.tile_desc_      = tile_desc;
    operation.kind_           = FmhaKind::FwdSplitKVCombine;
    operation.mode_           = fmha_problem.mode_;
    operation.dtype_          = fmha_problem.dtype_;
    operation.hdim_           = fmha_problem.v_head_dim_;
    operation.log_max_splits_ = CalculateLogMaxSplits(fmha_problem.num_splits_);

    // Determine padding configuration for Split-KV Combine
    auto padding_config          = DetermineSplitKVCombinePaddingConfig(fmha_problem, tile_desc, operation.mode_);
    operation.is_pad_q_seq_len_  = padding_config.is_pad_q_seq_len;
    operation.is_pad_v_head_dim_ = padding_config.is_pad_v_head_dim;

    return operation;
}

FmhaFwdAppendKVCodeGen FmhaEmitter::GenFmhaFwdAppendKVInstance(const FmhaProblem&          fmha_problem,
                                                               const FmhaAppendKVTileDesc& tile_desc) const
{
    FmhaFwdAppendKVCodeGen operation;

    // Set basic configuration
    operation.tile_desc_   = tile_desc;
    operation.kind_        = FmhaKind::FwdAppendKV;
    operation.mode_        = fmha_problem.mode_;
    operation.dtype_       = fmha_problem.dtype_;
    operation.rope_type_   = fmha_problem.rope_type_;
    operation.is_paged_kv_ = fmha_problem.paged_block_size_ > 0;

    // Determine padding configuration
    auto padding_config            = DetermineAppendKVPaddingConfig(fmha_problem, tile_desc, operation.mode_);
    operation.is_pad_q_seq_len_    = padding_config.is_pad_q_seq_len;
    operation.is_pad_kv_seq_len_   = padding_config.is_pad_kv_seq_len;
    operation.is_pad_qk_head_dim_  = padding_config.is_pad_qk_head_dim;
    operation.is_pad_v_head_dim_   = padding_config.is_pad_v_head_dim;
    operation.is_pad_qkv_head_dim_ = padding_config.is_pad_qkv_head_dim;

    return operation;
}

FmhaEmitter::PaddingConfig FmhaEmitter::DetermineFwdPaddingConfig(const FmhaProblem&    problem,
                                                                  const FmhaTileDesc&   tile_desc,
                                                                  FmhaMode              operation_mode,
                                                                  BlockFmhaPipelineEnum pipeline) const
{
    PaddingConfig config;

    // For ASYNC pipeline, always enable padding
    bool is_async_pipeline = (pipeline == BlockFmhaPipelineEnum::QRKSVS_ASYNC);

    if (is_async_pipeline) {
        config.is_pad_q_seq_len   = true;
        config.is_pad_kv_seq_len  = true;
        config.is_pad_qk_head_dim = true;
        config.is_pad_v_head_dim  = true;
    }
    else {
        // For Batch mode, check alignment with tile sizes
        if (operation_mode == FmhaMode::Batch) {
            config.is_pad_q_seq_len   = (problem.q_seq_len_ % tile_desc.bm0_ != 0);
            config.is_pad_kv_seq_len  = (problem.kv_seq_len_ == 0) || (problem.kv_seq_len_ % tile_desc.bn0_ != 0);
            config.is_pad_qk_head_dim = (problem.qk_head_dim_ % tile_desc.bk0_max_ != 0);
            config.is_pad_v_head_dim  = (problem.v_head_dim_ % tile_desc.bn1_ != 0);
        }
        else {
            // For Group mode, enable padding
            config.is_pad_q_seq_len   = true;
            config.is_pad_kv_seq_len  = true;
            config.is_pad_qk_head_dim = true;
            config.is_pad_v_head_dim  = true;
        }
    }

    // Combined head dimension padding flag
    config.is_pad_qkv_head_dim = (config.is_pad_qk_head_dim || config.is_pad_v_head_dim);

    return config;
}

FmhaEmitter::PaddingConfig FmhaEmitter::DetermineSplitKVPaddingConfig(const FmhaProblem&  problem,
                                                                      const FmhaTileDesc& tile_desc,
                                                                      FmhaMode            operation_mode) const
{
    PaddingConfig config;

    // For Batch mode, check alignment with tile sizes
    if (operation_mode == FmhaMode::Batch) {
        config.is_pad_q_seq_len   = (problem.q_seq_len_ % tile_desc.bm0_ != 0);
        config.is_pad_qk_head_dim = (problem.qk_head_dim_ % tile_desc.bk0_max_ != 0);
        config.is_pad_v_head_dim  = (problem.v_head_dim_ % tile_desc.bn1_ != 0);
    }
    else {
        // For Group mode, enable padding
        config.is_pad_q_seq_len   = true;
        config.is_pad_qk_head_dim = true;
        config.is_pad_v_head_dim  = true;
    }

    // Combined head dimension padding flag
    config.is_pad_qkv_head_dim = (config.is_pad_qk_head_dim || config.is_pad_v_head_dim);

    return config;
}

FmhaEmitter::PaddingConfig FmhaEmitter::DetermineSplitKVCombinePaddingConfig(
    const FmhaProblem& problem, const FmhaSplitKVCombineTileDesc& tile_desc, FmhaMode operation_mode) const
{
    PaddingConfig config;

    // For Batch mode, check alignment with tile sizes
    if (operation_mode == FmhaMode::Batch) {
        config.is_pad_q_seq_len  = (problem.q_seq_len_ % tile_desc.bm0_ != 0);
        config.is_pad_v_head_dim = (problem.v_head_dim_ % tile_desc.bn1_ != 0);
    }
    else {
        // For Group mode, enable padding
        config.is_pad_q_seq_len  = true;
        config.is_pad_v_head_dim = true;
    }

    return config;
}

FmhaEmitter::PaddingConfig FmhaEmitter::DetermineAppendKVPaddingConfig(const FmhaProblem&          problem,
                                                                       const FmhaAppendKVTileDesc& tile_desc,
                                                                       FmhaMode                    operation_mode) const
{
    PaddingConfig config;

    // For Batch mode, check alignment with tile sizes
    if (operation_mode == FmhaMode::Batch) {
        config.is_pad_q_seq_len   = (problem.q_seq_len_ % tile_desc.bs_ != 0);
        config.is_pad_kv_seq_len  = (problem.kv_seq_len_ == 0) || (problem.kv_seq_len_ % tile_desc.bsk_ != 0);
        config.is_pad_qk_head_dim = (problem.qk_head_dim_ % tile_desc.bd_ != 0);
        config.is_pad_v_head_dim  = (problem.v_head_dim_ % tile_desc.bdv_ != 0);
    }
    else {
        // For Group mode, enable padding
        config.is_pad_q_seq_len   = true;
        config.is_pad_kv_seq_len  = true;
        config.is_pad_qk_head_dim = true;
        config.is_pad_v_head_dim  = true;
    }

    // Combined head dimension padding flag
    config.is_pad_qkv_head_dim = (config.is_pad_qk_head_dim || config.is_pad_v_head_dim);

    return config;
}

int FmhaEmitter::CalculateLogMaxSplits(int num_splits) const
{
    if (num_splits <= 8) {
        return 3;
    }
    else if (num_splits <= 16) {
        return 4;
    }
    else if (num_splits <= 32) {
        return 5;
    }
    else if (num_splits <= 64) {
        return 6;
    }
    else if (num_splits <= 128) {
        return 7;
    }
    else {
        FC_THROW(InvalidArgument("num_splits {} is too large (max 128)", num_splits));
    }
}

void FmhaEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
}

}  // namespace flashck
