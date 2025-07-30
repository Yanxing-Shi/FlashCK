#pragma once

namespace flashck {

template<typename T>
struct GemmWeight {
    const T* weight_ = nullptr;
    const T* bias_   = nullptr;
};
}  // namespace flashck