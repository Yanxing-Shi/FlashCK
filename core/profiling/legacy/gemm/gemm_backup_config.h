#pragma once

#include "core/utils/common.h"

namespace flashck {

namespace legacy{

const std::vector<flashck::LegacyGemmConfig> g_backup_legacy_gemm_config = {flashck::LegacyGemmConfig{
    // tile
    flashck::LegacyTileConfig{
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{256}}}, // scale_block_size
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{256}}}, // block_size
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{128}}}, // m_per_block
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{64}}},  // n_per_block
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{16}}},  // k_per_block
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{16}}},  // a_k1
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{32}}},  // b_k1
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{32}}},  // m_per_xdl
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{4}}},   // n_per_xdl
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{2}}},   // m_xdl_per_wave
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{4}}}    // n_xdl_per_wave
    },
    flashck::LegacyBlockTransferConfig{
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{4, 64, 1}}},   // thread_cluster_length
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1, 0, 2}}},    // arrange_order
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1, 0, 2}}},    // src_access_order
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{2}}},          // src_vector_dim
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{16}}},         // src_scalar_per_vector
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{16}}},         // dst_scalar_per_vector_k1
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1}}}           // lds_add_extra_m
    },
    // b_block_transfer
    flashck::LegacyBlockTransferConfig{
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{4, 64, 1}}},   // thread_cluster_length
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1, 0, 2}}},    // arrange_order
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1, 0, 2}}},    // src_access_order
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{2}}},          // src_vector_dim
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{8}}},          // src_scalar_per_vector
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{8}}},          // dst_scalar_per_vector_k1
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1}}}           // lds_add_extra_m
    },
    // c_block_transfer
    flashck::LegacyCBlockTransferConfig{
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1}}}, // m_xdl_per_wave
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1}}}, // n_xdl_per_wave
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{1, 64, 1, 4}}}, // thread_cluster_length
        flashck::IntEnumConfigParam{std::vector<std::vector<int>>{{8}}}  // scalar_per_vector
    },
    // pipeline
    flashck::PipelineConfig{
        flashck::StrEnumConfigParam{{"interwave"}}, // scheduler
        flashck::StrEnumConfigParam{{"V1"}}         // pipeline
    }
}};

} // namespace legacy
} // namespace flashck