#pragma once

#include <cstdint>
#include <ostream>

#include "lightinfer/core/utils/enforce.h"

namespace lightinfer {

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

enum class DataType {
    UNDEFINED = 0,
    BOOL      = 1,
    INT4      = 2,  // Nibble
    UINT8     = 3,  // Byte
    INT8      = 4,  // Char
    UINT32    = 5,
    INT32     = 6,
    UINT64    = 7,
    INT64     = 8,
    FLOAT8    = 9,
    BFLOAT8   = 10,
    FLOAT16   = 11,
    BFLOAT16  = 12,
    FLOAT32   = 13,
    FLOAT64   = 14,
    BYTE      = 15,
    ALL_DTYPE = UNDEFINED,
    UINT16    = 16,
    INT16     = 17,
};

inline size_t SizeOf(DataType data_type)
{
    switch (data_type) {
        case DataType::INT4:
            return 1;
        case DataType::BOOL:
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::BYTE:
        case DataType::FLOAT8:
        case DataType::BFLOAT8:
            return 1;
        case DataType::BFLOAT16:
        case DataType::FLOAT16:
        case DataType::UINT16:
        case DataType::INT16:
            return 2;
        case DataType::FLOAT32:
        case DataType::INT32:
        case DataType::UINT32:
            return 4;
        case DataType::FLOAT64:
        case DataType::INT64:
        case DataType::UINT64:
            return 8;
        case DataType::UNDEFINED:
            LI_THROW(Unavailable("Data type {} is not supported by tensor.", static_cast<int>(data_type)));
    }
    return 0;
}

#define LI_FOR_EACH_DATA_TYPE(_)                                                                                       \
    _(bool, DataType::BOOL)                                                                                            \
    _(_BitInt(4), DataType::INT4)                                                                                      \
    _(int8_t, DataType::INT8)                                                                                          \
    _(int32_t, DataType::INT32)                                                                                        \
    _(uint32_t, DataType::UINT32)                                                                                      \
    _(int64_t, DataType::INT64)                                                                                        \
    _(uint64_t, DataType::UINT64)                                                                                      \
    _(unsigned _BitInt(8), DataType::BFLOAT8)                                                                          \
    _(_BitInt(8), DataType::FLOAT8)                                                                                    \
    _(ushort, DataType::BFLOAT16)                                                                                      \
    _(_Float16, DataType::FLOAT16)                                                                                     \
    _(float, DataType::FLOAT32)                                                                                        \
    _(double, DataType::FLOAT64)

template<DataType T>
struct DataTypeToCppType;

template<typename T>
struct CppTypeToDataType;

#define LI_SPECIALIZE_DataTypeToCppType(cpp_type, data_type)                                                           \
    template<>                                                                                                         \
    struct DataTypeToCppType<data_type> {                                                                              \
        using type = cpp_type;                                                                                         \
    };

LI_FOR_EACH_DATA_TYPE(LI_SPECIALIZE_DataTypeToCppType)

#undef LI_SPECIALIZE_DataTypeToCppType

#define LI_SPECIALIZE_CppTypeToDataType(cpp_type, data_type)                                                           \
    template<>                                                                                                         \
    struct CppTypeToDataType<cpp_type> {                                                                               \
        constexpr static DataType Type()                                                                               \
        {                                                                                                              \
            return data_type;                                                                                          \
        }                                                                                                              \
    };

LI_FOR_EACH_DATA_TYPE(LI_SPECIALIZE_CppTypeToDataType)

#undef LI_SPECIALIZE_CppTypeToDataType

inline std::ostream& operator<<(std::ostream& os, DataType dtype)
{
    switch (dtype) {
        case DataType::UNDEFINED:
            os << "Undefined";
            break;
        case DataType::BOOL:
            os << "bool";
            break;
        case DataType::INT4:
            os << "int4";
            break;
        case DataType::INT8:
            os << "int8";
            break;
        case DataType::INT32:
            os << "int32";
            break;
        case DataType::UINT32:
            os << "uint32";
            break;
        case DataType::INT64:
            os << "int64";
            break;
        case DataType::UINT64:
            os << "uint64";
            break;
        case DataType::FLOAT8:
            os << "float8";
            break;
        case DataType::BFLOAT8:
            os << "bfloat8";
            break;
        case DataType::BFLOAT16:
            os << "bfloat16";
            break;
        case DataType::FLOAT16:
            os << "float16";
            break;
        case DataType::FLOAT32:
            os << "float32";
            break;
        case DataType::FLOAT64:
            os << "float64";
            break;

        default:
            LI_THROW(Unimplemented("Invalid enum data type {}", static_cast<int>(dtype)));
    }
    return os;
}

inline std::string DataTypeToShortString(const DataType& dtype)
{
    switch (dtype) {
        case DataType::UNDEFINED:
            return "Undefined";
        case DataType::BOOL:
            return "bool";
        case DataType::INT4:
            return "i4";
        case DataType::INT8:
            return "i8";
        case DataType::INT32:
            return "i32";
        case DataType::UINT32:
            return "u32";
        case DataType::INT64:
            return "i64";
        case DataType::UINT64:
            return "u64";
        case DataType::FLOAT8:
            return "fp8";
        case DataType::BFLOAT8:
            return "bf8";
        case DataType::BFLOAT16:
            return "bf16";
        case DataType::FLOAT16:
            return "fp16";
        case DataType::FLOAT32:
            return "fp32";
        case DataType::FLOAT64:
            return "fp64";
        default:
            LI_THROW(Unimplemented("Invalid enum data type {}", static_cast<int>(dtype)));
    }
}

inline std::string DataTypeToString(const DataType& dtype)
{
    switch (dtype) {
        case DataType::UNDEFINED:
            return "Undefined(ALL_DTYPE)";
        case DataType::BOOL:
            return "bool";
        case DataType::INT4:
            return "_BitInt(4)";
        case DataType::INT8:
            return "int8_t";
        case DataType::UINT8:
            return "uint8_t";
        case DataType::INT32:
            return "int32_t";
        case DataType::UINT32:
            return "uint32_t";
        case DataType::INT64:
            return "int64_t";
        case DataType::UINT64:
            return "uint64_t";
        case DataType::BFLOAT8:
            return "unsigned _BitInt(8)";
        case DataType::FLOAT8:
            return "_BitInt(8)";
        case DataType::BFLOAT16:
            return "ushort";
        case DataType::FLOAT16:
            return "_Float16";
        case DataType::FLOAT32:
            return "float";
        case DataType::FLOAT64:
            return "double";
        default:
            LI_THROW(Unimplemented("Invalid enum data type {}", static_cast<int>(dtype)));
    }
}

inline std::string TileDataTypeToString(const DataType& dtype)
{
    switch (dtype) {
        case DataType::FLOAT32:
            return "ck_tile::fp32_t";
        case DataType::FLOAT16:
            return "ck_tile::half_t";
        case DataType::BFLOAT16:
            return "ck_tile::bf16_t";
        case DataType::INT8:
            return "ck_tile::int8_t";
        default:
            LI_THROW(Unimplemented("Invalid enum data type {}", static_cast<int>(dtype)));
    }
}

// used for numpy type description
inline std::string GetNumpyTypeDesc(DataType type)
{
    static const std::unordered_map<DataType, std::string> type_map{{DataType::UNDEFINED, "x"},
                                                                    {DataType::BOOL, "?"},
                                                                    {DataType::BYTE, "b"},
                                                                    {DataType::UINT8, "u1"},
                                                                    {DataType::UINT16, "u2"},
                                                                    {DataType::UINT32, "u4"},
                                                                    {DataType::UINT64, "u8"},
                                                                    {DataType::INT8, "i1"},
                                                                    {DataType::INT16, "i2"},
                                                                    {DataType::INT32, "i4"},
                                                                    {DataType::INT64, "i8"},
                                                                    {DataType::FLOAT16, "f2"},
                                                                    {DataType::FLOAT32, "f4"},
                                                                    {DataType::FLOAT64, "f8"}};

    if (type == DataType::BFLOAT16) {
        LI_THROW(
            Unimplemented("Numpy doesn't support bfloat16 as of now, it will be properly extended if numpy "
                          "supports. Please refer for the discussions https://github.com/numpy/numpy/issues/19808."));
    }

    return type_map.count(type) > 0 ? type_map.at(type) : "x";
}

// used for numpy type description
inline DataType GetTypeFromNumpyDesc(std::string type)
{
    static const std::unordered_map<std::string, DataType> type_map{{"?", DataType::BOOL},
                                                                    {"b", DataType::BYTE},
                                                                    {"u1", DataType::UINT8},
                                                                    {"u2", DataType::UINT16},
                                                                    {"u4", DataType::UINT32},
                                                                    {"u8", DataType::UINT64},
                                                                    {"i1", DataType::INT8},
                                                                    {"i2", DataType::INT16},
                                                                    {"i4", DataType::INT32},
                                                                    {"i8", DataType::INT64},
                                                                    {"f2", DataType::FLOAT16},
                                                                    {"f4", DataType::FLOAT32},
                                                                    {"f8", DataType::FLOAT64}};
    return type_map.at(type);
}

}  // namespace lightinfer