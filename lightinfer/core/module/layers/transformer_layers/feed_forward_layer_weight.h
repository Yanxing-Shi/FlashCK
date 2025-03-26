#pragma once

#include "lightinfer/core/module/layers/gemm_layers/linear_weight.h"

namespace lightinfer {

template<typename T>
struct FeedForwardWeight {
    LinearWeight<T> intermediate_weight_;
    LinearWeight<T> output_weight_;
};

}  // namespace lightinfer