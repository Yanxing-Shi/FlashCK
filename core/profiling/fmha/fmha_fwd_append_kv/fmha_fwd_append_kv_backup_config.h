#pragma once

#include "core/profiling/fmha/fmha_fwd_append_kv/fmha_fwd_append_kv_codegen.h"

#include "core/utils/common.h"

namespace flashck {

const std::vector<FmhaFwdAppendKVConfig> g_backup_fmha_append_kv_config = {
    {
        // tile
        {
            // block
            { { {256} }, { {128} }, { {128} } }
        },
        // padding
        { { {false} }, { {false} }, { {false} } },
        // launch
        { { {1} } }
    }
};


} // namespace flashck