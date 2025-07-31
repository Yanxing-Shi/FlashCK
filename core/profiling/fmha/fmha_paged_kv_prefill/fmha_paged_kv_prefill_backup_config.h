#pragma once

#include "core/profiling/fmha/fmha_paged_kv_prefill/fmha_paged_kv_prefill_codegen.h"

#include "core/utils/common.h"

namespace flashck {

const std::vector<FmhaPagedKVPrefillConfig> g_backup_fmha_paged_kv_prefill_config = {
    FmhaPagedKVPrefillConfig{
        FmhaPagedKVPrefillTileConfig{
            FmhaPagedKVPrefillBlockConfig{
                IntEnumConfigParam{ {256} },
                IntEnumConfigParam{ {128} },
                IntEnumConfigParam{ {128} },
                IntEnumConfigParam{ {256} },
                IntEnumConfigParam{ {128} },
                IntEnumConfigParam{ {128} }
            },
            FmhaPagedKVPrefillWarpConfig{
                IntEnumConfigParam{ {4} },
                IntEnumConfigParam{ {1} },
                IntEnumConfigParam{ {1} },
                IntEnumConfigParam{ {4} },
                IntEnumConfigParam{ {1} },
                IntEnumConfigParam{ {1} }
            },
            FmhaPagedKVPrefillWarpTileConfig{
                IntEnumConfigParam{ {64} },
                IntEnumConfigParam{ {32} },
                IntEnumConfigParam{ {32} },
                IntEnumConfigParam{ {64} },
                IntEnumConfigParam{ {32} },
                IntEnumConfigParam{ {32} }
            }
        },
        FmhaPagedKVPrefillPaddingConfig{
            BoolEnumConfigParam{ {false} },
            BoolEnumConfigParam{ {false} },
            BoolEnumConfigParam{ {false} },
            BoolEnumConfigParam{ {false} }
        },
        FmhaPagedKVPrefillLaunchConfig{
            IntEnumConfigParam{ {1} }
        },
        StrEnumConfigParam{ {"qr_ks_vs"} }
    }
};


} // namespace flashck