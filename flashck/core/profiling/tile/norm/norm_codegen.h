#pragma once

#include "flashck/core/profiling/tile/norm/norm_library.h"

#include "flashck/core/utils/common.h"

namespace flashck {

class NormTileDesc {
public:
    std::string GetConfigName();

    std::string Emit();

    int64_t repeat_m_;            // each thread repeat along M
    int64_t repeat_n_;            // each thread repeat along N
    int64_t thread_per_block_m_;  // num threads along M
    int64_t thread_per_block_n_;  // num threads along N
    int64_t vector_n_;            // vector size along N
};

class NormCodeGen {
public:
    std::string GetConfigName();

    std::string Emit();

    NormKind kind_;

    DataType x_dtype_;
    DataType y_dtype_;
    DataType smooth_scale_dtype_;
    DataType y_scale_dtype_;

    NormTileDesc tile_desc_;

    NormBiasEnum   is_add_bias_;
    FusedAddEnum   fused_add_;
    FusedQuantEnum fused_quant_;
};

}  // namespace flashck