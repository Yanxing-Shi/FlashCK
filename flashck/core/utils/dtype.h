#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace flashck {

// ==============================================================================
// Data Type Enumeration
// ==============================================================================

enum class DataType : uint8_t {
    BOOL     = 0,
    FLOAT8   = 1,
    BFLOAT8  = 2,
    FLOAT16  = 3,
    BFLOAT16 = 4,
    UINT32   = 5,
    INT32    = 6,
    FLOAT32  = 7,
    UINT64   = 8,
    INT64    = 9,
    FLOAT64  = 10
};

// ==============================================================================
// Type Utilities
// ==============================================================================

// Get data type from C++ type
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
    else if constexpr (std::is_same_v<T, ushort>) {
        return DataType::BFLOAT16;
    }
    else if constexpr (std::is_same_v<T, _Float16>) {
        return DataType::FLOAT16;
    }
    else {
        static_assert(false, "Unsupported data type");
    }
}

// Convert DataType to ck_tile string representation
inline std::string DataTypeToTileString(DataType dtype)
{
    switch (dtype) {
        case DataType::FLOAT32:
            return "ck_tile::fp32_t";
        case DataType::FLOAT16:
            return "ck_tile::half_t";
        case DataType::BFLOAT16:
            return "ck_tile::bf16_t";
        default:
            throw std::runtime_error("Unsupported data type for tile string conversion");
    }
}

// Convert DataType to string representation
inline const char* DataTypeToString(DataType type)
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

// Get size in bytes for a data type
inline constexpr size_t SizeOf(DataType data_type)
{
    switch (data_type) {
        case DataType::BOOL:
        case DataType::FLOAT8:
        case DataType::BFLOAT8:
            return 1;
        case DataType::BFLOAT16:
        case DataType::FLOAT16:
            return 2;
        case DataType::FLOAT32:
        case DataType::INT32:
        case DataType::UINT32:
            return 4;
        case DataType::FLOAT64:
        case DataType::INT64:
        case DataType::UINT64:
            return 8;
        default:
            return 0;
    }
}

// Check if a data type is floating point
inline constexpr bool IsFloatingPoint(DataType data_type)
{
    switch (data_type) {
        case DataType::FLOAT8:
        case DataType::BFLOAT8:
        case DataType::FLOAT16:
        case DataType::BFLOAT16:
        case DataType::FLOAT32:
        case DataType::FLOAT64:
            return true;
        default:
            return false;
    }
}

// Check if a data type is integer
inline constexpr bool IsInteger(DataType data_type)
{
    switch (data_type) {
        case DataType::INT32:
        case DataType::UINT32:
        case DataType::INT64:
        case DataType::UINT64:
            return true;
        default:
            return false;
    }
}

// ==============================================================================
// Type Mapping Templates
// ==============================================================================

// Check if a type is supported
template<typename T>
constexpr bool IsSupportedDataType()
{
    return std::is_same_v<T, bool> || std::is_same_v<T, float> || std::is_same_v<T, double>
           || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, int64_t>
           || std::is_same_v<T, uint64_t> || std::is_same_v<T, ushort> || std::is_same_v<T, _Float16>;
}

// Optimized type mapping for HIP
#define FC_FOR_EACH_DATA_TYPE(_)                                                                                       \
    _(bool, DataType::BOOL)                                                                                            \
    _(int32_t, DataType::INT32)                                                                                        \
    _(uint32_t, DataType::UINT32)                                                                                      \
    _(int64_t, DataType::INT64)                                                                                        \
    _(uint64_t, DataType::UINT64)                                                                                      \
    _(float, DataType::FLOAT32)                                                                                        \
    _(double, DataType::FLOAT64)                                                                                       \
    _(ushort, DataType::BFLOAT16)                                                                                      \
    _(_Float16, DataType::FLOAT16)

template<DataType T>
struct DataTypeToCppType;

template<typename T>
struct CppTypeToDataType;

#define FC_SPECIALIZE_DataTypeToCppType(cpp_type, data_type)                                                           \
    template<>                                                                                                         \
    struct DataTypeToCppType<data_type> {                                                                              \
        using type = cpp_type;                                                                                         \
    };

FC_FOR_EACH_DATA_TYPE(FC_SPECIALIZE_DataTypeToCppType)

#undef FC_SPECIALIZE_DataTypeToCppType

#define FC_SPECIALIZE_CppTypeToDataType(cpp_type, data_type)                                                           \
    template<>                                                                                                         \
    struct CppTypeToDataType<cpp_type> {                                                                               \
        static constexpr DataType value = data_type;                                                                   \
    };

FC_FOR_EACH_DATA_TYPE(FC_SPECIALIZE_CppTypeToDataType)

#undef FC_SPECIALIZE_CppTypeToDataType

inline uint16_t float2bhalf(float x)
{
    uint32_t tmp = __builtin_bit_cast(uint32_t, x) >> 16;
    return static_cast<uint16_t>(tmp);
}

inline float bhalf2float(uint16_t x)
{
    uint32_t extended = static_cast<uint32_t>(x) << 16;
    return __builtin_bit_cast(float, extended);
}

inline uint16_t float2half(float x)
{
    return static_cast<uint16_t>(x);  // Simplified fallback
}

inline float half2float(uint16_t x)
{
    return static_cast<float>(x);  // Simplified fallback
}

// ==============================================================================
// Utility Functions for Type Validation
// ==============================================================================

// Validate if a DataType is supported in current compilation context
inline bool IsDataTypeSupported(DataType type)
{
    switch (type) {
        case DataType::BOOL:
        case DataType::INT32:
        case DataType::UINT32:
        case DataType::INT64:
        case DataType::UINT64:
        case DataType::FLOAT32:
        case DataType::FLOAT64:
            return true;
        case DataType::FLOAT16:
        case DataType::BFLOAT16:
            return true;
        case DataType::FLOAT8:
        case DataType::BFLOAT8:
            return false;  // Not yet supported
        default:
            return false;
    }
}

// Get alignment requirement for a data type
inline constexpr size_t AlignmentOf(DataType data_type)
{
    return SizeOf(data_type);  // Simple alignment = size for most types
}

}  // namespace flashck