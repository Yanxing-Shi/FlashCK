#pragma once

#include "core/profiling/legacy/gemm/gemm_codegen.h"

#include "core/utils/common.h"

namespace flashck {

namespace legacy{

const std::vector<flashck::LegacyGemmConfig> g_backup_legacy_gemm_config = {flashck::LegacyGemmConfig{
    // tile
    flashck::LegacyTileConfig{
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{256}}}, // scale_block_size_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{256}}}, // block_size_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{128}}}, // m_per_block_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{64}}},  // n_per_block_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{16}}},  // k_per_block_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{16}}},  // a_k1_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{32}}},  // b_k1_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{32}}},  // m_per_xdl_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{4}}},   // n_per_xdl_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{2}}},   // m_xdl_per_wave_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{4}}}    // n_xdl_per_wave_
    },
    flashck::LegacyBlockTransferConfig{
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{4, 64, 1}}},   // thread_cluster_length_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1, 0, 2}}},    // arrange_order_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1, 0, 2}}},    // src_access_order_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{2}}},          // src_vector_dim_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{16}}},         // src_scalar_per_vector_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{16}}},         // dst_scalar_per_vector_k1_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1}}}           // lds_add_extra_m_
    },
    // b_block_transfer
    flashck::LegacyBlockTransferConfig{
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{4, 64, 1}}},   // thread_cluster_length_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1, 0, 2}}},    // arrange_order_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1, 0, 2}}},    // src_access_order_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{2}}},          // src_vector_dim_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{8}}},          // src_scalar_per_vector_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{8}}},          // dst_scalar_per_vector_k1_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1}}}           // lds_add_extra_m_
    },
    // c_block_transfer
    flashck::LegacyCBlockTransferConfig{
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1}}}, // m_xdl_per_wave_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1}}}, // n_xdl_per_wave_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1, 64, 1, 4}}}, // thread_cluster_length_
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{8}}}  // scalar_per_vector_
    },
    // pipeline
    flashck::PipelineConfig{
        flashck::StrEnumConfigParam{{"interwave"}}, // scheduler
        flashck::StrEnumConfigParam{{"V1"}}         // pipeline
    }
}};

GemmSpecialization DetermineGemmSpecialization(const GemmProblem&  gemm_problem,
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

// Generate all possible GemmCodegen instances from a LegacyGemmConfig
inline std::vector<GemmCodegen> GenerateLegacyGemmInstances(const flashck::LegacyGemmConfig& config, const GemmProblem& gemm_problem) {
    std::vector<GemmCodegen> result;
    // Directly use config fields for clarity
    auto flatten = [](const std::vector<std::vector<int>>& v) -> std::vector<int64_t> {
        std::vector<int64_t> out;
        for (const auto& inner : v) out.insert(out.end(), inner.begin(), inner.end());
        return out;
    };
    std::vector<std::vector<int64_t>> all_lists = {
        flatten(config.tile_config_.scale_block_size_.values_),
        flatten(config.tile_config_.block_size_.values_),
        flatten(config.tile_config_.m_per_block_.values_),
        flatten(config.tile_config_.n_per_block_.values_),
        flatten(config.tile_config_.k_per_block_.values_),
        flatten(config.tile_config_.a_k1_.values_),
        flatten(config.tile_config_.b_k1_.values_),
        flatten(config.tile_config_.m_per_xdl_.values_),
        flatten(config.tile_config_.n_per_xdl_.values_),
        flatten(config.tile_config_.m_xdl_per_wave_.values_),
        flatten(config.tile_config_.n_xdl_per_wave_.values_),
        flatten(config.a_block_config_.thread_cluster_length_.values_),
        flatten(config.a_block_config_.arrange_order_.values_),
        flatten(config.a_block_config_.src_access_order_.values_),
        flatten(config.a_block_config_.src_vector_dim_.values_),
        flatten(config.a_block_config_.src_scalar_per_vector_.values_),
        flatten(config.a_block_config_.dst_scalar_per_vector_k1_.values_),
        flatten(config.a_block_config_.lds_add_extra_m_.values_),
        flatten(config.b_block_config_.thread_cluster_length_.values_),
        flatten(config.b_block_config_.arrange_order_.values_),
        flatten(config.b_block_config_.src_access_order_.values_),
        flatten(config.b_block_config_.src_vector_dim_.values_),
        flatten(config.b_block_config_.src_scalar_per_vector_.values_),
        flatten(config.b_block_config_.dst_scalar_per_vector_k1_.values_),
        flatten(config.b_block_config_.lds_add_extra_m_.values_),
        flatten(config.c_block_config_.m_xdl_per_wave_.values_),
        flatten(config.c_block_config_.n_xdl_per_wave_.values_),
        flatten(config.c_block_config_.thread_cluster_length_.values_),
        flatten(config.c_block_config_.scalar_per_vector_.values_)
    };

    // Prepare all string lists
    std::vector<BlockGemmPipelineScheduler> pipeline_schedulers;
    for (const auto& s : config.pipeline_.scheduler_.values_) {
        auto sched = GetPipelineSchedulerFromString(s);
        if (sched != BlockGemmPipelineScheduler::COUNT)
            pipeline_schedulers.push_back(sched);
    }

    std::vector<BlockGemmPipelineVersion> pipeline_versions;
    for (const auto& s : config.pipeline_.version_.values_) {
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
                GemmCodegen gemm;
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

} // namespace legacy
} // namespace flashck