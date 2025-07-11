#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>

namespace flashck {

enum class DataType {
    BOOL    = 0,
    FLOAT8  = 1,
    BFLOAT8 = 2,

    FLOAT16  = 3,
    BFLOAT16 = 4,

    UINT32  = 5,
    INT32   = 6,
    FLOAT32 = 7,

    UINT64  = 8,
    INT64   = 9,
    FLOAT64 = 10
};

// GetDataType function template
// Returns the DataType enum corresponding to the given type T
// Supported types: bool, float, double, int32_t, uint32_t, int64_t, uint64_t
// If T is not supported, static_assert will trigger a compile-time error
template<typename T>
constexpr DataType GetDataType()
{
    if constexpr (std::is_same_v<T, bool>) {
        return DataType::BOOL;
    }
    else if constexpr (std::is_same_v<T, float>) {
        return DataType::FLOAT32;
    }
    else if constexpr (std::is_same_v<T, double>) {
        return DataType::FLOAT64;
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        return DataType::INT32;
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        return DataType::UINT32;
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        return DataType::INT64;
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        return DataType::UINT64;
    }
    else {
        static_assert(false, "Unsupported data type");
    }
}

const char* DataTypeToString(DataType type)
{
    switch (type) {
        case DataType::BOOL:
            return "bool";
        case DataType::FLOAT8:
            return "fp8";
        case DataType::BFLOAT8:
            return "bf8";
        case DataType::FLOAT16:
            return "fp16";
        case DataType::BFLOAT16:
            return "bf16";
        case DataType::UINT32:
            return "u32";
        case DataType::INT32:
            return "i32";
        case DataType::FLOAT32:
            return "fp32";
        case DataType::UINT64:
            return "u64";
        case DataType::INT64:
            return "i64";
        case DataType::FLOAT64:
            return "fp64";
        default:
            return "unknown";
    }
}

// Check if a type is supported as a data type
// Supported types: bool, float, double, int32_t, uint32_t, int64_t, uint64_t
template<typename T>
constexpr bool IsSupportedDataType()
{
    return std::is_same_v<T, bool> || std::is_same_v<T, float> || std::is_same_v<T, double>
           || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, int64_t>
           || std::is_same_v<T, uint64_t>;
}

}  // namespace flashck