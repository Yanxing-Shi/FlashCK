#pragma once

namespace flashck {

template<typename T>
struct LinearWeight {
    const T* kernel_ = nullptr;
    const T* bias_   = nullptr;
};
}  // namespace flashck