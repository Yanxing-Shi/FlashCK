#pragma once

#include "hip/hip_runtime.h"

#include "flashck/core/utils/hip_raii.h"
#include "flashck/core/utils/macros.h"

namespace flashck {

template<typename T>
struct IsSupportedType {
    static constexpr bool value = false;
};

template<>
struct IsSupportedType<float> {
    static constexpr bool value = true;
};

template<typename T>
void DeviceMalloc(
    T**                                                                                           ptr,
    size_t                                                                                        count,
    int                                                                                           init_type = 0,
    typename std::conditional_t<IsSupportedType<T>::value && std::is_floating_point_v<T>, T, int> min_val =
        std::is_floating_point_v<T> ? T(0) : 0,
    typename std::conditional_t<IsSupportedType<T>::value && std::is_floating_point_v<T>, T, int> max_val =
        std::is_floating_point_v<T> ? T(1) : 100);

template<typename T>
void DeviceFree(T*& ptr);

template<typename T>
void DeviceFill(T* devptr, size_t size, T value, hipStream_t stream = nullptr);

template<typename T>
void HipD2HCpyAsync(T* tgt, const T* src, size_t size, hipStream_t stream = nullptr);

template<typename T>
void HipH2DCpyAsync(T* tgt, const T* src, size_t size, hipStream_t stream = nullptr);

template<typename T>
void HipD2DCpyAsync(T* tgt, const T* src, size_t size, hipStream_t stream = nullptr);

}  // namespace flashck