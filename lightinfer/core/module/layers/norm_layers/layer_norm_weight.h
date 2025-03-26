#pragma once

namespace lightinfer {
template<typename T>
struct LayerNormWeight {
    const T* gamma_ = nullptr;
    const T* beta_  = nullptr;
};
}  // namespace lightinfer