#pragma once

namespace lightinfer {

template<typename T>
struct LinearWeight {
    const T* kernel_ = nullptr;
    const T* bias_   = nullptr;
};
}  // namespace lightinfer