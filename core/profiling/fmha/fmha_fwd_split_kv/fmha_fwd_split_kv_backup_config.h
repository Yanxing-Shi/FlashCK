#pragma once

#include "core/utils/common.h"

namespace flashck {

const std::vector<FmhaFwdSplitKVConfig> g_backup_fmha_fwd_split_kv_config = {
    FmhaFwdSplitKVConfig{
        FmhaFwdSplitKVTileConfig{
            FmhaFwdSplitKVBlockConfig{
                IntEnumConfigParam{ {256} },
                IntEnumConfigParam{ {128} },
                IntEnumConfigParam{ {128} },
                IntEnumConfigParam{ {256} },
                IntEnumConfigParam{ {128} },
                IntEnumConfigParam{ {128} }
            },
            FmhaFwdSplitKVWarpConfig{
                IntEnumConfigParam{ {4} },
                IntEnumConfigParam{ {1} },
                IntEnumConfigParam{ {1} },
                IntEnumConfigParam{ {4} },
                IntEnumConfigParam{ {1} },
                IntEnumConfigParam{ {1} }
            },
            FmhaFwdSplitKVWarpTileConfig{
                IntEnumConfigParam{ {64} },
                IntEnumConfigParam{ {32} },
                IntEnumConfigParam{ {32} },
                IntEnumConfigParam{ {64} },
                IntEnumConfigParam{ {32} },
                IntEnumConfigParam{ {32} }
            }
        },
        FmhaFwdSplitKVPaddingConfig{
            BoolEnumConfigParam{ {false} },
            BoolEnumConfigParam{ {false} },
            BoolEnumConfigParam{ {false} },
            BoolEnumConfigParam{ {false} }
        },
        FmhaFwdSplitKVLaunchConfig{
            IntEnumConfigParam{ {1} }
        },
        StrEnumConfigParam{ {"qr_ks_vs"} }
    }
};


} // namespace flashck