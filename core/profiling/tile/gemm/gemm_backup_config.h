#pragma once

#include "core/utils/common.h"

namespace flashck {

namespace tile{

const std::vector<flashck::TileGemmConfig> g_tile_gemm_backup_tile_config = {flashck::TileGemmConfig{
    flashck::TileConfig{
        flashck::BlockConfig{
            flashck::IntEnumConfigParam{{{256}}},
            flashck::IntEnumConfigParam{{{128}}},
            flashck::IntEnumConfigParam{{{128}}}
        },
        flashck::WarpConfig{
            flashck::IntEnumConfigParam{{{4}}},
            flashck::IntEnumConfigParam{{{1}}},
            flashck::IntEnumConfigParam{{{1}}}
        },
        flashck::WarpTileConfig{
            flashck::IntEnumConfigParam{{{64}}},
            flashck::IntEnumConfigParam{{{32}}},
            flashck::IntEnumConfigParam{{{32}}}
        }
    },
    flashck::PaddingConfig{
        flashck::BoolEnumConfigParam{{false}},
        flashck::BoolEnumConfigParam{{false}},
        flashck::BoolEnumConfigParam{{false}}
    },
    flashck::LaunchConfig{
        flashck::IntEnumConfigParam{{{1}}}
    },
    flashck::PartitionConfig{
        flashck::IntEnumConfigParam{{{1}}},
        flashck::IntEnumConfigParam{{{1}}},
        flashck::IntEnumConfigParam{{{1}}},
    },
    flashck::PipelineConfig{
        flashck::StrEnumConfigParam{{"compv3"}},
        flashck::StrEnumConfigParam{{"intrawave"}}
    },
    flashck::StrEnumConfigParam{{"cshuffle"}}
    }
};

} // namespace tile
} // namespace flashck

