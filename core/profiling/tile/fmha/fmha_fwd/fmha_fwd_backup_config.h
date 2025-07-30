#pragma once

#include "core/utils/common.h"

namespace flashck {

const std::vector<FmhaFwdConfig> g_backup_fmha_fwd_config = {
    FmhaFwdConfig{FmhaFwdTileConfig{
        FmhaFwdBlockConfig{
            IntEnumConfigParam{{{256}}},
            IntEnumConfigParam{{{128}}},
            IntEnumConfigParam{{{128}}},
            IntEnumConfigParam{{{128}}},
            IntEnumConfigParam{{{128}}},
            IntEnumConfigParam{{{128}}}
        },
        FmhaFwdWarpConfig{
            IntEnumConfigParam{{{4}}},
            IntEnumConfigParam{{{1}}},
            IntEnumConfigParam{{{1}}},
            IntEnumConfigParam{{{1}}},
            IntEnumConfigParam{{{1}}},
            IntEnumConfigParam{{{1}}}
        },
        FmhaFwdWarpTileConfig{
            IntEnumConfigParam{{{64}}},
            IntEnumConfigParam{{{32}}},
            IntEnumConfigParam{{{32}}},
            IntEnumConfigParam{{{32}}},
            IntEnumConfigParam{{{32}}},
            IntEnumConfigParam{{{32}}}

        }
    },
    FmhaFwdPaddingConfig{
        BoolEnumConfigParam{{false}},
        BoolEnumConfigParam{{false}},
        BoolEnumConfigParam{{false}},
        BoolEnumConfigParam{{false}},
    },
    FmhaFwdLaunchConfig{
        IntEnumConfigParam{{{1}}},
    },
    StrEnumConfigParam{{"qr_ks_vs"}},
    }
};


} // namespace flashck