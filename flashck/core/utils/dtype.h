#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>
#include <ostream>

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

// Stream operator for DataType enum
inline std::ostream& operator<<(std::ostream& os, DataType type)
{
    return os << DataTypeToString(type);
}

inline size_t SizeOf(DataType data_type)
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
            throw std::runtime_error("Unsupported data type");
    }
    return 0;
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

#define FC_FOR_EACH_DATA_TYPE(_)                                                                                       \
    _(bool, DataType::BOOL)                                                                                            \
    _(int32_t, DataType::INT32)                                                                                        \
    _(uint32_t, DataType::UINT32)                                                                                      \
    _(int64_t, DataType::INT64)                                                                                        \
    _(uint64_t, DataType::UINT64)                                                                                      \
    _(_BitInt(8), DataType::FLOAT8)                                                                                    \
    _(ushort, DataType::BFLOAT16)                                                                                      \
    _(_Float16, DataType::FLOAT16)                                                                                     \
    _(float, DataType::FLOAT32)                                                                                        \
    _(double, DataType::FLOAT64)

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
        constexpr static DataType Type()                                                                               \
        {                                                                                                              \
            return data_type;                                                                                          \
        }                                                                                                              \
    };

FC_FOR_EACH_DATA_TYPE(FC_SPECIALIZE_CppTypeToDataType)

#undef FC_SPECIALIZE_CppTypeToDataType

}  // namespace flashck