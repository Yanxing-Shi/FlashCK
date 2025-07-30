#pragma once

#include "core/profiling/tile/fmha/fmha_fwd_split_kv_combine/fmha_fwd_split_kv_combine_codegen.h"

#include "core/utils/common.h"

namespace flashck {

const std::vector<FmhaFwdSplitKVCombineConfig> g_backup_fmha_fwd_split_kv_combine_config = {
    FmhaFwdSplitKVCombineConfig{FmhaFwdSplitKVCombineTileConfig{
        FmhaFwdSplitKVCombineBlockConfig{
            IntEnumConfigParam{{{256}}},
        }
    },
    FmhaFwdSplitKVCombinePaddingConfig{
        BoolEnumConfigParam{{false}},
        BoolEnumConfigParam{{false}}
    },
    FmhaFwdSplitKVCombineLaunchConfig{
        IntEnumConfigParam{{{1}}},
    }
    }
};


} // namespace flashck