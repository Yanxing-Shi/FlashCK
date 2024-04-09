#pragma once

#include <cstdint>
#include <ostream>

#include "ater/core/utils/enforce.h"

namespace ater {

enum class DataType {
    UNDEFINED = 0,
    BOOL      = 1,
    UINT8     = 2,  // Byte
    INT8      = 3,  // Char
    UINT32    = 4,
    INT32     = 5,
    UINT64    = 6,
    INT64     = 7,
    FLOAT32   = 8,
    FLOAT64   = 9,
    FLOAT16   = 10,
    BFLOAT16  = 11,
    ALL_DTYPE = UNDEFINED,
};

inline size_t SizeOf(DataType data_type)
{
    switch (data_type) {
        case DataType::BOOL:
        case DataType::UINT8:
        case DataType::INT8:
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
        case DataType::UNDEFINED:
            ATER_THROW(Unavailable("Data type {} is not supported by tensor.", static_cast<int>(data_type)));
    }
    return 0;
}

#define ATER_FOR_EACH_DATA_TYPE(_)                                                                                     \
    _(bool, DataType::BOOL)                                                                                            \
    _(int8_t, DataType::INT8)                                                                                          \
    _(int32_t, DataType::INT32)                                                                                        \
    _(uint32_t, DataType::UINT32)                                                                                      \
    _(int64_t, DataType::INT64)                                                                                        \
    _(uint64_t, DataType::UINT64)                                                                                      \
    _(ushort, DataType::BFLOAT16)                                                                                      \
    _(_Float16, DataType::FLOAT16)                                                                                     \
    _(float, DataType::FLOAT32)                                                                                        \
    _(double, DataType::FLOAT64)

template<DataType T>
struct DataTypeToCppType;

template<typename T>
struct CppTypeToDataType;

#define ATER_SPECIALIZE_DataTypeToCppType(cpp_type, data_type)                                                         \
    template<>                                                                                                         \
    struct DataTypeToCppType<data_type> {                                                                              \
        using type = cpp_type;                                                                                         \
    };

ATER_FOR_EACH_DATA_TYPE(ATER_SPECIALIZE_DataTypeToCppType)

#undef ATER_SPECIALIZE_DataTypeToCppType

#define ATER_SPECIALIZE_CppTypeToDataType(cpp_type, data_type)                                                         \
    template<>                                                                                                         \
    struct CppTypeToDataType<cpp_type> {                                                                               \
        constexpr static DataType Type()                                                                               \
        {                                                                                                              \
            return data_type;                                                                                          \
        }                                                                                                              \
    };

ATER_FOR_EACH_DATA_TYPE(ATER_SPECIALIZE_CppTypeToDataType)

#undef ATER_SPECIALIZE_CppTypeToDataType

inline std::ostream& operator<<(std::ostream& os, DataType dtype)
{
    switch (dtype) {
        case DataType::UNDEFINED:
            os << "Undefined";
            break;
        case DataType::BOOL:
            os << "bool";
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
            ATER_THROW(Unimplemented("Invalid enum data type {}", static_cast<int>(dtype)));
    }
    return os;
}

inline std::string DataTypeToString(const DataType& dtype)
{
    switch (dtype) {
        case DataType::UNDEFINED:
            return "Undefined(ALL_DTYPE)";
        case DataType::BOOL:
            return "bool";
        case DataType::INT8:
            return "int8";
        case DataType::UINT8:
            return "uint8";
        case DataType::INT32:
            return "int32";
        case DataType::UINT32:
            return "uint32";
        case DataType::INT64:
            return "int64";
        case DataType::UINT64:
            return "uint64";
        case DataType::BFLOAT16:
            return "bfloat16";
        case DataType::FLOAT16:
            return "float16";
        case DataType::FLOAT32:
            return "float32";
        case DataType::FLOAT64:
            return "float64";
        default:
            ATER_THROW(Unimplemented("Invalid enum data type {}", static_cast<int>(dtype)));
    }
}

}  // namespace ater