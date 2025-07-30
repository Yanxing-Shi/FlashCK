#pragma once

#include "core/profiling/tile/fmha/fmha_batch_prefill/fmha_batch_prefill_codegen.h"

#include "core/utils/common.h"

namespace flashck {

const std::vector<FmhaBatchPrefillConfig> g_backup_fmha_batch_prefill_config = {
    FmhaBatchPrefillConfig{FmhaBatchPrefillTileConfig{
        FmhaBatchPrefillBlockConfig{
            IntEnumConfigParam{{{256}}},
            IntEnumConfigParam{{{128}}},
            IntEnumConfigParam{{{128}}},
            IntEnumConfigParam{{{128}}},
            IntEnumConfigParam{{{128}}},
            IntEnumConfigParam{{{128}}}
        },
        FmhaBatchPrefillWarpConfig{
            IntEnumConfigParam{{{4}}},
            IntEnumConfigParam{{{1}}},
            IntEnumConfigParam{{{1}}},
            IntEnumConfigParam{{{4}}},
            IntEnumConfigParam{{{1}}},
            IntEnumConfigParam{{{1}}}
        },
        FmhaBatchPrefillWarpTileConfig{
            IntEnumConfigParam{{{64}}},
            IntEnumConfigParam{{{32}}},
            IntEnumConfigParam{{{32}}},
            IntEnumConfigParam{{{64}}},
            IntEnumConfigParam{{{32}}},
            IntEnumConfigParam{{{32}}}
        }
    },
    FmhaBatchPrefillPaddingConfig{
        BoolEnumConfigParam{{false}},
        BoolEnumConfigParam{{false}},
        BoolEnumConfigParam{{false}},
        BoolEnumConfigParam{{false}}
    },
    FmhaBatchPrefillLaunchConfig{
        IntEnumConfigParam{{{1}}},
    },
    StrEnumConfigParam{{"qr_ks_vs"}},
    }
};


} // namespace flashck