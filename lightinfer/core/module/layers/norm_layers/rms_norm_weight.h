#pragma once

namespace lightinfer {
template<typename T>
struct RMSNormWeight {
    const T* gamma_ = nullptr;
};
}  // namespace lightinfer