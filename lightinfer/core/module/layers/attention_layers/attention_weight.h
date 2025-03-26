#pragma once

#include "lightinfer/core/module/layers/gemm_layers/linear_weight.h"

namespace lightinfer {

template<typename T>
struct AttentionWeight {
    LinearWeight<T> query_weight_;
    LinearWeight<T> key_weight_;
    LinearWeight<T> value_weight_;

    LinearWeight<T> query_key_value_weight_;

    LinearWeight<T> attention_output_weight_;
};

}  // namespace lightinfer