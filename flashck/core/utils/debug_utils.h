#pragma once

#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

namespace flashck {

template<typename Y, typename X>
inline Y bit_cast(const X& x)
{
    // static_assert(__has_builtin(__builtin_bit_cast), "");
    // static_assert(sizeof(X) == sizeof(Y), "Do not support cast between different size of type");

    return __builtin_bit_cast(Y, x);
}

inline float bf16_to_float_raw(uint16_t x)
{
    union {
        uint32_t int32;
        float    fp32;
    } u = {uint32_t(x) << 16};
    return u.fp32;
}

template<typename T>
void ResultChecker(const T* tensor, int64_t elem_cnt, const std::string& tensor_name, hipStream_t stream = 0);

template<typename T>
void PrintToFile(const T* result, const int size, const char* file, hipStream_t stream, std::ios::openmode open_mode);

template<typename T>
void CheckMaxVal(const T* result, const int size, hipStream_t stream = nullptr);

template<typename T>
void PrintToScreen(const T* result, const int size, const std::string& name = "");

}  // namespace flashck