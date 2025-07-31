#pragma once

#include "core/utils/common.h"

namespace flashck {

const std::vector<GemmConfig> g_tile_gemm_backup_tile_config = {
    GemmConfig{
        GemmTileConfig{
            GemmBlockConfig{
                IntEnumConfigParam{ {256} },
                IntEnumConfigParam{ {128} },
                IntEnumConfigParam{ {128} }
            },
            GemmWarpConfig{
                IntEnumConfigParam{ {4} },
                IntEnumConfigParam{ {1} },
                IntEnumConfigParam{ {1} }
            },
            GemmWarpTileConfig{
                IntEnumConfigParam{ {64} },
                IntEnumConfigParam{ {32} },
                IntEnumConfigParam{ {32} }
            }
        },
        GemmPaddingConfig{
            BoolEnumConfigParam{ {false} },
            BoolEnumConfigParam{ {false} },
            BoolEnumConfigParam{ {false} }
        },
        GemmLaunchConfig{
            IntEnumConfigParam{ {1} }
        },
        GemmPartitionConfig{
            IntEnumConfigParam{ {1} },
            IntEnumConfigParam{ {1} },
            IntEnumConfigParam{ {1} }
        },
        GemmPipelineConfig{
            StrEnumConfigParam{ {"compv3"} },
            StrEnumConfigParam{ {"intrawave"} },
            StrEnumConfigParam{ {"cshuffle"} }
        }
    }
};

} // namespace flashck

