#pragma once

namespace flashck {
template<typename T>
struct RMSNormWeight {
    const T* gamma_ = nullptr;
};
}  // namespace flashck