#pragma once

#include "lightinfer/core/module/layers/gemm_layers/linear_weight.h"

namespace lightinfer {

template<typename T>
struct LlamaMLPWeight {
    LinearWeight<T> gate_weight_;
    LinearWeight<T> up_weight_;
    LinearWeight<T> down_weight_;
};

}  // namespace lightinfer