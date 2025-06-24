#pragma once

#include "flashck/core/module/layers/gemm_layers/linear_weight.h"

namespace flashck {

template<typename T>
struct FeedForwardWeight {
    LinearWeight<T> intermediate_weight_;
    LinearWeight<T> output_weight_;
};

}  // namespace flashck