#include "core/profiling/legacy/gemm/gemm_emitter.h"

#include <algorithm>
#include <stdexcept>

#include "core/utils/common.h"

FC_DECLARE_int32(FC_TUNING_MODE);  // Mode for GEMM operation: 0 - heuristic, 1 - autotuning, 2 - hybrid
FC_DECLARE_bool(FC_ENABLE_CONFIG_JSON);
FC_DECLARE_string(FC_CONFIG_JSON_PATH);
FC_DECLARE_int32(FC_ENABLE_JSON_MODE);

namespace flashck {

namespace legacy{

bool GemmEmitter::IsValidTile(const GemmTileDesc& tile_desc, const GemmProblem& gemm_problem) const
{
    // Basic parameter checks
    if (tile_desc.block_size_ <= 0 || tile_desc.m_per_block_ <= 0 || tile_desc.n_per_block_ <= 0 || tile_desc.k_per_block_ <= 0) {
        VLOG(3) << "Invalid tile descriptor: negative or zero values not allowed";
        return false;
    }

    // Block size constraint: must not exceed hardware thread block limit
    if (tile_desc.block_size_ > 1024) {  // Common GPU thread block limit
        VLOG(3) << "Invalid tile descriptor: block size " << tile_desc.block_size_ << " exceeds limit (1024)";
        return false;
    }

    // XDL dimensions must be positive
    if (tile_desc.m_per_xdl_ <= 0 || tile_desc.n_per_xdl_ <= 0 || tile_desc.m_xdl_per_wave_ <= 0 || tile_desc.n_xdl_per_wave_ <= 0) {
        VLOG(3) << "Invalid tile descriptor: XDL dimensions must be positive";
        return false;
    }

    // K1 dimensions must be positive
    if (tile_desc.a_k1_ <= 0 || tile_desc.b_k1_ <= 0) {
        VLOG(3) << "Invalid tile descriptor: K1 dimensions must be positive";
        return false;
    }

    // Block dimensions must not exceed problem dimensions
    if (tile_desc.m_per_block_ > gemm_problem.m_ || tile_desc.n_per_block_ > gemm_problem.n_ || tile_desc.k_per_block_ > gemm_problem.k_) {
        VLOG(3) << "Invalid tile descriptor: block dimensions (" << tile_desc.m_per_block_ << "x" << tile_desc.n_per_block_ << "x" << tile_desc.k_per_block_ << ") exceed problem dimensions (" << gemm_problem.m_ << "x" << gemm_problem.n_ << "x" << gemm_problem.k_ << ")";
        return false;
    }

    // --- Value relationship constraints ---
    // MPerBlock must be a multiple of (MPerXdl * MXdlPerWave)
    int m_xdl = tile_desc.m_per_xdl_;
    int m_xdl_wave = tile_desc.m_xdl_per_wave_;
    int m_xdl_total = m_xdl * m_xdl_wave;
    if (tile_desc.m_per_block_ % m_xdl_total != 0) {
        VLOG(3) << "Invalid tile descriptor: MPerBlock " << tile_desc.m_per_block_ << " is not a multiple of (MPerXdl " << m_xdl << " * MXdlPerWave " << m_xdl_wave << ") = " << m_xdl_total;
        return false;
    }

    // NPerBlock must be a multiple of (NPerXdl * NXdlPerWave)
    int n_xdl = tile_desc.n_per_xdl_;
    int n_xdl_wave = tile_desc.n_xdl_per_wave_;
    int n_xdl_total = n_xdl * n_xdl_wave;
    if (tile_desc.n_per_block_ % n_xdl_total != 0) {
        VLOG(3) << "Invalid tile descriptor: NPerBlock " << tile_desc.n_per_block_ << " is not a multiple of (NPerXdl " << n_xdl << " * NXdlPerWave " << n_xdl_wave << ") = " << n_xdl_total;
        return false;
    }

    // KPerBlock must be divisible by AK1 and BK1
    if (tile_desc.k_per_block_ % tile_desc.a_k1_ != 0) {
        VLOG(3) << "Invalid tile descriptor: KPerBlock " << tile_desc.k_per_block_ << " is not divisible by AK1 " << tile_desc.a_k1_;
        return false;
    }
    if (tile_desc.k_per_block_ % tile_desc.b_k1_ != 0) {
        VLOG(3) << "Invalid tile descriptor: KPerBlock " << tile_desc.k_per_block_ << " is not divisible by BK1 " << tile_desc.b_k1_;
        return false;
    }

    // KPerBlock must align with MMA K tile (if applicable)
    // For legacy CK, this is typically 8, 16, or 32 depending on architecture. Here, use 8 as a common default.
    constexpr int k_mma_k_tile = 8;
    if (tile_desc.k_per_block_ % k_mma_k_tile != 0) {
        VLOG(3) << "Invalid tile descriptor: KPerBlock " << tile_desc.k_per_block_ << " is not aligned to MMA K tile (" << k_mma_k_tile << ")";
        return false;
    }

    // XDL configuration must not exceed block dimensions
    const int total_xdl_per_block_m = tile_desc.m_xdl_per_wave_ * tile_desc.m_per_xdl_;
    const int total_xdl_per_block_n = tile_desc.n_xdl_per_wave_ * tile_desc.n_per_xdl_;
    if (total_xdl_per_block_m > tile_desc.m_per_block_ || total_xdl_per_block_n > tile_desc.n_per_block_) {
        VLOG(3) << "Invalid tile descriptor: XDL configuration exceeds block dimensions";
        return false;
    }

    return true;
}

bool GemmEmitter::IsValidBlockTransfer(const GemmTileDesc& tile_desc,const BlockTransferDesc& block_transfer_desc) const{

    // Check thread cluster length dimensions
    if (block_transfer_desc.thread_cluster_length_.size() != 3) {
        VLOG(3) << "Invalid block transfer: thread_cluster_length size != 3";
        return false;
    }
    for (int v : block_transfer_desc.thread_cluster_length_) {
        if (v <= 0) {
            VLOG(3) << "Invalid block transfer: thread_cluster_length contains non-positive value";
            return false;
        }
    }

    // Check src_access_order
    if (block_transfer_desc.src_access_order_.size() != 3) {
        VLOG(3) << "Invalid block transfer: src_access_order size != 3";
        return false;
    }

    // Check src_vector_dim
    if (block_transfer_desc.src_vector_dim_ < 0) {
        VLOG(3) << "Invalid block transfer: src_vector_dim negative";
        return false;
    }

    // Check src_scalar_per_vector and dst_scalar_per_vector
    if (block_transfer_desc.src_scalar_per_vector_ <= 0 || block_transfer_desc.dst_scalar_per_vector_ <= 0) {
        VLOG(3) << "Invalid block transfer: scalar_per_vector values must be positive";
        return false;
    }

    // --- Value relationship checks with tile_desc ---
    // 1. The product of thread_cluster_length should match m_per_block, n_per_block, k_per_block in some order
    int cluster_m = block_transfer_desc.thread_cluster_length_[0];
    int cluster_n = block_transfer_desc.thread_cluster_length_[1];
    int cluster_k = block_transfer_desc.thread_cluster_length_[2];
    int cluster_total = cluster_m * cluster_n * cluster_k;
    if (cluster_total != tile_desc.block_size_) {
        VLOG(3) << "Invalid block transfer: thread_cluster_length product " << cluster_total << " != block_size " << tile_desc.block_size_;
        return false;
    }

    // 2. src_scalar_per_vector should divide m_per_block, n_per_block, or k_per_block depending on src_vector_dim
    int dim = block_transfer_desc.src_vector_dim_;
    int scalar_per_vec = block_transfer_desc.src_scalar_per_vector_;
    bool vector_dim_ok = false;
    if (dim == 0 && tile_desc.m_per_block_ % scalar_per_vec == 0) vector_dim_ok = true;
    if (dim == 1 && tile_desc.n_per_block_ % scalar_per_vec == 0) vector_dim_ok = true;
    if (dim == 2 && tile_desc.k_per_block_ % scalar_per_vec == 0) vector_dim_ok = true;
    if (!vector_dim_ok) {
        VLOG(3) << "Invalid block transfer: src_scalar_per_vector " << scalar_per_vec << " does not divide tile block dim for src_vector_dim " << dim;
        return false;
    }

    // 3. dst_scalar_per_vector should divide a_k1 or b_k1 (for A/B block transfer)
    int dst_scalar = block_transfer_desc.dst_scalar_per_vector_;
    if (dst_scalar <= 0) {
        VLOG(3) << "Invalid block transfer: dst_scalar_per_vector <= 0";
        return false;
    }
    // For A block transfer, check a_k1; for B, check b_k1. Here, check both for generality.
    if (tile_desc.a_k1_ % dst_scalar != 0 && tile_desc.b_k1_ % dst_scalar != 0) {
        VLOG(3) << "Invalid block transfer: dst_scalar_per_vector " << dst_scalar << " does not divide a_k1 or b_k1 (" << tile_desc.a_k1_ << ", " << tile_desc.b_k1_ << ")";
        return false;
    }

    return true;

}

bool GemmEmitter::IsValidCBlockTransfer(const GemmTileDesc& tile_desc, const CBlockTransferDesc& c_block_transfer_desc) const{

    // 1. Check thread cluster length dimensions
    if (c_block_transfer_desc.m_n_block_wave_per_xdl_.size() != 2) {
        VLOG(3) << "Invalid C block transfer: thread_cluster_length size != 2";
        return false;
    }
    for (int v : c_block_transfer_desc.m_n_block_wave_per_xdl_) {
        if (v <= 0) {
            VLOG(3) << "Invalid C block transfer: thread_cluster_length contains non-positive value";
            return false;
        }
    }

    // 2. Check arrange_order and access_order
    // CBlockTransferDesc does not have arrange_order_ or access_order_ fields; skip or adjust this check as needed.
    // If you need to check m_n_block_wave_per_xdl_ only:
    if (c_block_transfer_desc.m_n_block_wave_per_xdl_.size() != 2) {
        VLOG(3) << "Invalid C block transfer: arrange_order or access_order size != 2";
        return false;
    }

    // 3. Check scalar_per_vector
    if (c_block_transfer_desc.scalar_per_vector_ <= 0) {
        VLOG(3) << "Invalid C block transfer: scalar_per_vector must be positive";
        return false;
    }

    // 4. Value relationship checks with tile_desc
    // The product of thread_cluster_length should match block_size
    int cluster_0 = c_block_transfer_desc.m_n_block_wave_per_xdl_[0];
    int cluster_1 = c_block_transfer_desc.m_n_block_wave_per_xdl_[1];
    int cluster_total = cluster_0 * cluster_1;
    if (cluster_total != tile_desc.block_size_) {
        VLOG(3) << "Invalid C block transfer: thread_cluster_length product " << cluster_total << " != block_size " << tile_desc.block_size_;
        return false;
    }

    return true;
}

bool GemmEmitter::IsValidInstance(const GemmCodeGen& gemm_instance, const GemmProblem& gemm_problem) const
{
    // Check if the instance is valid for the given GEMM problem
    return IsValidTile(gemm_instance.tile_desc_, gemm_problem) &&
           IsValidBlockTransfer(gemm_instance.tile_desc_, gemm_instance.a_block_desc_) &&
           IsValidBlockTransfer(gemm_instance.tile_desc_, gemm_instance.b_block_desc_) &&
           IsValidCBlockTransfer(gemm_instance.tile_desc_, gemm_instance.c_block_desc_);
}

// std::vector<GemmCodeGen> GemmEmitter::HeuristicFilter(const std::vector<GemmCodeGen>& gemm_instances,
//                                                        const GemmProblem&               gemm_problem) const
// {

//     return filtered_instances;
// }

GemmSpecialization GemmEmitter::DetermineGemmSpecialization(const GemmProblem&  gemm_problem,
                                               const GemmTileDesc& tile_desc) 
{
    auto IntegerDivideCeil = [](int64_t dividend, int64_t divisor) -> int64_t {
        return (dividend + divisor - 1) / divisor;
    };

    if (gemm_problem.m_ % tile_desc.m_per_block_ != 0 && gemm_problem.n_ % tile_desc.n_per_block_ != 0
        && gemm_problem.k_ % tile_desc.k_per_block_ != 0) {
        return GemmSpecialization::MNKPadding;
    }
    else if (gemm_problem.m_ % tile_desc.m_per_block_ != 0 && gemm_problem.n_ % tile_desc.n_per_block_ != 0) {
        return GemmSpecialization::MNPadding;
    }
    else if (gemm_problem.m_ % tile_desc.m_per_block_ != 0 && gemm_problem.k_ % tile_desc.k_per_block_ != 0) {
        return GemmSpecialization::MKPadding;
    }
    else if (gemm_problem.n_ % tile_desc.n_per_block_ != 0 && gemm_problem.k_ % tile_desc.k_per_block_ != 0) {
        return GemmSpecialization::NKPadding;
    }
    else if (gemm_problem.m_ % tile_desc.m_per_block_ != 0) {
        return GemmSpecialization::MPadding;
    }
    else if (gemm_problem.n_ % tile_desc.n_per_block_ != 0) {
        return GemmSpecialization::NPadding;
    }
    else if (gemm_problem.k_ % tile_desc.k_per_block_ != 0) {
        return GemmSpecialization::KPadding;
    }
    else {
        return GemmSpecialization::Default;
    }
}


// Generate all possible GemmCodeGen instances from a LegacyGemmConfig
std::vector<GemmCodeGen> GemmEmitter::GenerateLegacyGemmInstances(const flashck::LegacyGemmConfig& config, const GemmProblem& gemm_problem) {
    std::vector<GemmCodeGen> result;
    // Directly use config fields for clarity
    auto flatten = [](const std::vector<std::vector<int>>& v) -> std::vector<int64_t> {
        std::vector<int64_t> out;
        for (const auto& inner : v) out.insert(out.end(), inner.begin(), inner.end());
        return out;
    };
    std::vector<std::vector<int64_t>> all_lists = {
        flatten(config.tile.scale_block_size.values),
        flatten(config.tile.block_size.values),
        flatten(config.tile.m_per_block.values),
        flatten(config.tile.n_per_block.values),
        flatten(config.tile.k_per_block.values),
        flatten(config.tile.a_k1.values),
        flatten(config.tile.b_k1.values),
        flatten(config.tile.m_per_xdl.values),
        flatten(config.tile.n_per_xdl.values),
        flatten(config.tile.m_xdl_per_wave.values),
        flatten(config.tile.n_xdl_per_wave.values),
        flatten(config.a_block.thread_cluster_length.values),
        flatten(config.a_block.arrange_order.values),
        flatten(config.a_block.src_access_order.values),
        flatten(config.a_block.src_vector_dim.values),
        flatten(config.a_block.src_scalar_per_vector.values),
        flatten(config.a_block.dst_scalar_per_vector_k1.values),
        flatten(config.a_block.lds_add_extra_m.values),
        flatten(config.b_block.thread_cluster_length.values),
        flatten(config.b_block.arrange_order.values),
        flatten(config.b_block.src_access_order.values),
        flatten(config.b_block.src_vector_dim.values),
        flatten(config.b_block.src_scalar_per_vector.values),
        flatten(config.b_block.dst_scalar_per_vector_k1.values),
        flatten(config.b_block.lds_add_extra_m.values),
        flatten(config.c_block.m_xdl_per_wave.values),
        flatten(config.c_block.n_xdl_per_wave.values),
        flatten(config.c_block.thread_cluster_length.values),
        flatten(config.c_block.scalar_per_vector.values)
    };

    // Prepare all string lists
    std::vector<BlockGemmPipelineScheduler> pipeline_schedulers;
    for (const auto& s : config.pipeline.scheduler.values) {
        auto sched = GetPipelineSchedulerFromString(s);
        if (sched != BlockGemmPipelineScheduler::COUNT)
            pipeline_schedulers.push_back(sched);
    }

    std::vector<BlockGemmPipelineVersion> pipeline_versions;
    for (const auto& s : config.pipeline.version.values) {
        auto ver = GetPipelineVersionFromString(s);
        if (ver != BlockGemmPipelineVersion::COUNT)
            pipeline_versions.push_back(ver);
    }

    std::vector<std::vector<flashck::ProductElem>> product_lists;
    product_lists.reserve(all_lists.size());
    for (const auto& vec : all_lists) {
        std::vector<flashck::ProductElem> tmp;
        tmp.reserve(vec.size());
        for (auto v : vec) tmp.emplace_back(flashck::ProductElem(v));
        product_lists.push_back(std::move(tmp));
    }
    flashck::CartesianProduct(product_lists, [&](const std::vector<flashck::ProductElem>& vals) {
        std::vector<int64_t> flat_vals;
        for (const auto& v : vals) {
            if (std::holds_alternative<int64_t>(v)) {
                flat_vals.push_back(std::get<int64_t>(v));
            } else {
                const auto& vec = std::get<std::vector<int64_t>>(v);
                flat_vals.insert(flat_vals.end(), vec.begin(), vec.end());
            }
        }
        size_t idx = 0;
        // Tile
        GemmTileDesc tile_desc;
        tile_desc.scale_block_size_ = flat_vals[idx++];
        tile_desc.block_size_ = flat_vals[idx++];
        tile_desc.m_per_block_ = flat_vals[idx++];
        tile_desc.n_per_block_ = flat_vals[idx++];
        tile_desc.k_per_block_ = flat_vals[idx++];
        tile_desc.a_k1_ = flat_vals[idx++];
        tile_desc.b_k1_ = flat_vals[idx++];
        tile_desc.m_per_xdl_ = flat_vals[idx++];
        tile_desc.n_per_xdl_ = flat_vals[idx++];
        tile_desc.m_xdl_per_wave_ = flat_vals[idx++];
        tile_desc.n_xdl_per_wave_ = flat_vals[idx++];

        auto gemm_spec = DetermineGemmSpecialization(gemm_problem, tile_desc);

        // A block
        auto a0 = flat_vals[idx++];
        auto a1 = flat_vals[idx++];
        auto a2 = flat_vals[idx++];
        auto a3 = flat_vals[idx++];
        auto a4 = flat_vals[idx++];
        auto a5 = flat_vals[idx++];
        auto a6 = flat_vals[idx++];
        BlockTransferDesc a_desc({a0}, {a1}, {a2}, a3, a4, a5, a6);
        // B block
        auto b0 = flat_vals[idx++];
        auto b1 = flat_vals[idx++];
        auto b2 = flat_vals[idx++];
        auto b3 = flat_vals[idx++];
        auto b4 = flat_vals[idx++];
        auto b5 = flat_vals[idx++];
        auto b6 = flat_vals[idx++];
        BlockTransferDesc b_desc({b0}, {b1}, {b2}, b3, b4, b5, b6);
        // C block
        auto c0 = flat_vals[idx++];
        auto c1 = flat_vals[idx++];
        auto c2 = flat_vals[idx++];
        auto c3 = flat_vals[idx++];
        CBlockTransferDesc c_desc(c0, c1, {c2}, c3);
        // For all string combinations
        for (const auto& sch : pipeline_schedulers) {
            for (const auto& p : pipeline_versions) {
                GemmCodeGen gemm;
                gemm.problem_ = gemm_problem;
                gemm.tile_desc_ = tile_desc;
                gemm.a_block_desc_ = a_desc;
                gemm.b_block_desc_ = b_desc;
                gemm.c_block_desc_ = c_desc;
                gemm.gemm_spec_ = gemm_spec; 
                gemm.pipeline_scheduler_ = sch; 
                gemm.pipeline_version_ = p;
                result.push_back(gemm);
            }
        }
    });
    return result;
}


void GemmEmitter::GenerateInstances(GemmProblem& gemm_problem)
{
    FC_ENFORCE_EQ(FLAGS_FC_TUNING_MODE == 0 || FLAGS_FC_TUNING_MODE == 1 || FLAGS_FC_TUNING_MODE == 2,
                  true,
                  Unavailable("Unsupported mode: {}, valid modes are 0 (heuristic), 1 (autotuning), 2 (hybrid)", FLAGS_FC_TUNING_MODE));

    // Check if instances already exist for this GEMM kind
    if (instance_map_.find(gemm_problem.kind_) != instance_map_.end() && !instance_map_[gemm_problem.kind_].empty()) {
        VLOG(2) << "Instances already generated for GEMM kind: " << GetGemmKindName(gemm_problem.kind_);
        return;
    }

    // Load legacy GEMM configuration if available
    std::vector<GemmCodeGen> gemm_instances;
    if (FLAGS_FC_ENABLE_CONFIG_JSON) {
        auto base_json_path = std::filesystem::path(FLAGS_FC_CONFIG_JSON_PATH) / GetGemmKindName(gemm_problem.kind_);
        if(FLAGS_FC_ENABLE_JSON_MODE == 0) {
            std::filesystem::path json_path = base_json_path / "default_config.json";
            LegacyGemmConfig config = LoadConfigJson<LegacyGemmConfig>(json_path);
            gemm_instances = GenerateLegacyGemmInstances(config, gemm_problem);
        } else if (FLAGS_FC_ENABLE_JSON_MODE == 1) {
            std::filesystem::path json_path = base_json_path / "user_config.json";
            LegacyGemmConfig config = LoadConfigJson<LegacyGemmConfig>(json_path);
            gemm_instances = GenerateLegacyGemmInstances(config, gemm_problem);
        } else if (FLAGS_FC_ENABLE_JSON_MODE == 2) {
            std::filesystem::path default_json_path = base_json_path / "default_config.json";
            LegacyGemmConfig default_config = LoadConfigJson<LegacyGemmConfig>(default_json_path);
            auto gemm_default_instances = GenerateLegacyGemmInstances(default_config, gemm_problem);

            std::filesystem::path user_json_path = base_json_path / "user_config.json";
            LegacyGemmConfig user_config = LoadConfigJson<LegacyGemmConfig>(user_json_path);
            auto gemm_user_instances = GenerateLegacyGemmInstances(user_config, gemm_problem);

            gemm_instances.insert(gemm_instances.end(), gemm_default_instances.begin(), gemm_default_instances.end());
            gemm_instances.insert(gemm_instances.end(), gemm_user_instances.begin(), gemm_user_instances.end());
        }
    else{
            LOG(WARNING)<< "FC_ENABLE_JSON_MODE is set to an unsupported value: " << FLAGS_FC_ENABLE_JSON_MODE;
        }
    } else {
        LOG(WARNING)<< "FC_ENABLE_CONFIG_JSON is not enabled";
    }

    for (const auto& config : g_backup_legacy_gemm_config) {
        GemmCodeGen gemm;

        gemm.problem_ = gemm_problem;

        gemm.tile_desc_ = GemmTileDesc{
            config.tile.scale_block_size.values[0][0],
            config.tile.block_size.values[0][0],
            config.tile.m_per_block.values[0][0],
            config.tile.n_per_block.values[0][0],
            config.tile.k_per_block.values[0][0],
            config.tile.a_k1.values[0][0],
            config.tile.b_k1.values[0][0],
            config.tile.m_per_xdl.values[0][0],
            config.tile.n_per_xdl.values[0][0],
            config.tile.m_xdl_per_wave.values[0][0],
            config.tile.n_xdl_per_wave.values[0][0]
        };

        gemm.a_block_desc_ = BlockTransferDesc{
            std::vector<int64_t>(config.a_block.thread_cluster_length.values[0].begin(), config.a_block.thread_cluster_length.values[0].end()),
            std::vector<int64_t>(config.a_block.arrange_order.values[0].begin(), config.a_block.arrange_order.values[0].end()),
            std::vector<int64_t>(config.a_block.src_access_order.values[0].begin(), config.a_block.src_access_order.values[0].end()),
            config.a_block.src_vector_dim.values[0][0],
            config.a_block.src_scalar_per_vector.values[0][0],
            config.a_block.dst_scalar_per_vector_k1.values[0][0],
            config.a_block.lds_add_extra_m.values[0][0]
        };

        gemm.b_block_desc_ = BlockTransferDesc{
            std::vector<int64_t>(config.b_block.thread_cluster_length.values[0].begin(), config.b_block.thread_cluster_length.values[0].end()),
            std::vector<int64_t>(config.b_block.arrange_order.values[0].begin(), config.b_block.arrange_order.values[0].end()),
            std::vector<int64_t>(config.b_block.src_access_order.values[0].begin(), config.b_block.src_access_order.values[0].end()),
            config.b_block.src_vector_dim.values[0][0],
            config.b_block.src_scalar_per_vector.values[0][0],
            config.b_block.dst_scalar_per_vector_k1.values[0][0],
            config.b_block.lds_add_extra_m.values[0][0]
        };

        gemm.c_block_desc_ = CBlockTransferDesc{
            config.c_block.m_xdl_per_wave.values[0][0],
            config.c_block.n_xdl_per_wave.values[0][0],
            std::vector<int64_t>(config.c_block.thread_cluster_length.values[0].begin(), config.c_block.thread_cluster_length.values[0].end()),
            config.c_block.scalar_per_vector.values[0][0]
        };

        gemm.gemm_spec_ = DetermineGemmSpecialization(gemm_problem, gemm.tile_desc_);

        gemm.pipeline_version_ = GetPipelineVersionFromString(config.pipeline.version.values[0]);
        gemm.pipeline_scheduler_ = GetPipelineSchedulerFromString(config.pipeline.scheduler.values[0]);

        gemm_instances.push_back(gemm);
    }
    

    // check instances
    std::vector<GemmCodeGen> valid_gemm_instances;
    for (const auto& gemm_instance : gemm_instances) {
        if (IsValidInstance(gemm_instance, gemm_problem)) {
            valid_gemm_instances.push_back(gemm_instance);
        }
    }

    switch (FLAGS_FC_TUNING_MODE) {
        case 0:  // Heuristic mode
            // VLOG(1) << "Generating instances using heuristic mode for GEMM kind: "
            //         << GetGemmKindName(gemm_problem.kind_);
            // tile_descriptors = HeuristicFilter(gemm_instances, gemm_problem);
            // break;

        case 1:  // Autotuning mode
            VLOG(1) << "Generating instances using autotuning mode for GEMM kind: "
                    << GetGemmKindName(gemm_problem.kind_);
            break;

        case 2:  // Hybrid mode
            VLOG(1) << "Generating instances using hybrid mode for GEMM kind: " << GetGemmKindName(gemm_problem.kind_);
            // Start with heuristic filter, then expand if needed
            // tile_descriptors = HeuristicFilter(gemm_instances, gemm_problem);
            // if (tile_descriptors.size() < 3) {  // Expand if too few options
            //     for (const auto& tile_desc : g_backup_gemm_tile_descriptions) {
            //         if (IsValidTile(tile_desc, gemm_problem)) {
            //             auto it = std::find_if(tile_descriptors.begin(),
            //                                    tile_descriptors.end(),
            //                                    [&tile_desc](const GemmTileDesc& existing) {
            //                                        return existing.GetInstanceName() == tile_desc.GetInstanceName();
            //                                    });
            //             if (it == tile_descriptors.end()) {
            //                 tile_descriptors.push_back(tile_desc);
            //             }
            //         }
            //     }
            // }
            break;

        default:
            FC_THROW(Unavailable("Invalid tuning mode: {}", FLAGS_FC_TUNING_MODE));
    }

    if (valid_gemm_instances.empty()) {
        FC_THROW(Unavailable("No valid GEMM instances found for GEMM problem"));
    }

    // Generate instances
    std::map<std::string, GemmCodeGen>& kind_instance_map = instance_map_[gemm_problem.kind_];
    int64_t                             generated_count   = 0;

    for (const auto& instance : valid_gemm_instances) {
        try {
            std::string instance_name = instance.GetInstanceName();

            // Avoid duplicates
            if (kind_instance_map.find(instance_name) == kind_instance_map.end()) {
                kind_instance_map[instance_name] = std::move(instance);
                generated_count++;
                VLOG(2) << "Generated GEMM instance: " << instance_name;
            }
            else {
                VLOG(3) << "Skipped duplicate GEMM instance: " << instance_name;
            }
        }
        catch (const std::exception& e) {
            LOG(WARNING) << "Failed to create GEMM codegen for instance: " << instance.GetInstanceName()
                         << ", error: " << e.what();
        }
    }

    num_instances_ += generated_count;
    VLOG(1) << "Generated " << generated_count << " GEMM instances for kind: " << GetGemmKindName(gemm_problem.kind_)
            << " (total: " << num_instances_ << ")";
}

int64_t GemmEmitter::GetNumInstances() const
{
    return num_instances_;
}

void GemmEmitter::ClearInstances()
{
    instance_map_.clear();
    num_instances_ = 0;
    VLOG(1) << "Cleared all GEMM instances";
}

}  // namespace legacy
}  // namespace flashck