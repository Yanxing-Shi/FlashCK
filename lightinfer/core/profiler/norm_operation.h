#pragma once

#include "lightinfer/core/profiler/library.h"
#include "lightinfer/core/utils/dtype.h"

namespace lightinfer {

struct NormProblem {
    DataType x_dtype_;
    DataType y_dtype_;
    DataType smooth_scale_dtype_;
    DataType y_scale_dtype_;

    int64_t m_;
    int64_t n_;

    NormOperationKind operation_kind_;
    TensorOperation   epilogue_op_;

    NormBiasEnum   is_add_bias_;
    FusedAddEnum   fused_add_;
    FusedQuantEnum fused_quant_;
};

struct NormTileDesc {
    int64_t repeat_m_;            // each thread repeat along M
    int64_t repeat_n_;            // each thread repeat along N
    int64_t thread_per_block_m_;  // num threads along M
    int64_t thread_per_block_n_;  // num threads along N
    int64_t vector_n_;            // vector size along N

    std::string GetConfigName();

    std::string Emit();
};

struct NormOperation {

    std::string GetConfigName();

    std::string Emit();

    NormOperationKind operation_kind_;
    TensorOperation   epilogue_op_;

    NormBiasEnum   is_add_bias_;
    FusedAddEnum   fused_add_;
    FusedQuantEnum fused_quant_;

    DataType x_dtype_;
    DataType y_dtype_;
    DataType smooth_scale_dtype_;
    DataType y_scale_dtype_;

    NormTileDesc tile_desc_;
};

}  // namespace lightinfer