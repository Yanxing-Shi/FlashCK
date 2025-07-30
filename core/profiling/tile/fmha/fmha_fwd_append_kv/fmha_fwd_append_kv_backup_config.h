#pragma once

#include "core/profiling/tile/fmha/fmha_fwd_append_kv/fmha_fwd_append_kv_codegen.h"

#include "core/utils/common.h"

namespace flashck {

const std::vector<FmhaFwdAppendKVConfig> g_backup_fmha_append_kv_config = {
    FmhaFwdAppendKVConfig{FmhaFwdAppendKVTileConfig{
        FmhaFwdAppendKVBlockConfig{
            IntEnumConfigParam{{{256}}},
            IntEnumConfigParam{{{128}}},
            IntEnumConfigParam{{{128}}},
            IntEnumConfigParam{{{128}}}
        }
    },
    FmhaFwdAppendKVPaddingConfig{
        BoolEnumConfigParam{{false}},
        BoolEnumConfigParam{{false}},
        BoolEnumConfigParam{{false}},
        BoolEnumConfigParam{{false}},
    },
    FmhaFwdAppendKVLaunchConfig{
        IntEnumConfigParam{{{1}}},
    }
    }
};


} // namespace flashck