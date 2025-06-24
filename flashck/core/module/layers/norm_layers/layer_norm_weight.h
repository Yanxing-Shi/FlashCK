#pragma once

namespace flashck {
template<typename T>
struct LayerNormWeight {
    const T* gamma_ = nullptr;
    const T* beta_  = nullptr;
};
}  // namespace flashck