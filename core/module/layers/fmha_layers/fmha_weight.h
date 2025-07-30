#pragma once

#include "core/module/layers/gemm_layers/gemm_weight.h"

namespace flashck {

template<typename T>
struct FmhaWeight {
    GemmWeight<T> query_weight_;
    GemmWeight<T> key_weight_;
    GemmWeight<T> value_weight_;

    GemmWeight<T> query_key_value_weight_;

    GemmWeight<T> attention_output_weight_;
};

}  // namespace flashck