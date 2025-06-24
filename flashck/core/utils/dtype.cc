#include "flashck/core/utils/dtype.h"

namespace flashck {

template<>
struct DataTypeTraits<DataType::BOOL> {
    using cpp_type                          = bool;
    static constexpr const char* full_name  = "bool";
    static constexpr const char* short_name = "b";
    static constexpr size_t      size       = sizeof(cpp_type);
    static constexpr size_t      alignment  = alignof(cpp_type);
};

template<>
struct DataTypeTraits<DataType::FLOAT8> {
    using cpp_type                          = _BitInt(8);  // sync with CK tile
    static constexpr const char* full_name  = "float8";
    static constexpr const char* short_name = "fp8";
    static constexpr size_t      size       = sizeof(cpp_type);
    static constexpr size_t      alignment  = alignof(cpp_type);
};

template<>
struct DataTypeTraits<DataType::BFLOAT8> {
    using cpp_type                          = unsigned _BitInt(8);  // sync with CK tile
    static constexpr const char* full_name  = "bfloat8";
    static constexpr const char* short_name = "bf8";
    static constexpr size_t      size       = sizeof(cpp_type);
    static constexpr size_t      alignment  = alignof(cpp_type);
};

template<>
struct DataTypeTraits<DataType::FLOAT16> {
    using cpp_type                          = _Float16;  // sync with CK tile
    static constexpr const char* full_name  = "float16";
    static constexpr const char* short_name = "fp16";
    static constexpr size_t      size       = sizeof(cpp_type);
    static constexpr size_t      alignment  = alignof(cpp_type);
};

template<>
struct DataTypeTraits<DataType::BFLOAT16> {
    using cpp_type                          = ushort;  // sync with CK tile
    static constexpr const char* full_name  = "bfloat16";
    static constexpr const char* short_name = "bf16";
    static constexpr size_t      size       = sizeof(cpp_type);
    static constexpr size_t      alignment  = alignof(cpp_type);
};

template<>
struct DataTypeTraits<DataType::UINT32> {
    using cpp_type                          = uint32_t;
    static constexpr const char* full_name  = "uint32";
    static constexpr const char* short_name = "u32";
    static constexpr size_t      size       = sizeof(cpp_type);
    static constexpr size_t      alignment  = alignof(cpp_type);
};

template<>
struct DataTypeTraits<DataType::INT32> {
    using cpp_type                          = int32_t;
    static constexpr const char* full_name  = "int32";
    static constexpr const char* short_name = "i32";
    static constexpr size_t      size       = sizeof(cpp_type);
    static constexpr size_t      alignment  = alignof(cpp_type);
};

template<>
struct DataTypeTraits<DataType::UINT64> {
    using cpp_type                          = uint64_t;
    static constexpr const char* full_name  = "uint64";
    static constexpr const char* short_name = "u64";
    static constexpr size_t      size       = sizeof(cpp_type);
    static constexpr size_t      alignment  = alignof(cpp_type);
};

template<>
struct DataTypeTraits<DataType::INT64> {
    using cpp_type                          = int64_t;
    static constexpr const char* full_name  = "int64";
    static constexpr const char* short_name = "i64";
    static constexpr size_t      size       = sizeof(cpp_type);
    static constexpr size_t      alignment  = alignof(cpp_type);
};

template<>
struct DataTypeTraits<DataType::FLOAT64> {
    using cpp_type                          = double;
    static constexpr const char* full_name  = "float64";
    static constexpr const char* short_name = "fp64";
    static constexpr size_t      size       = sizeof(cpp_type);
    static constexpr size_t      alignment  = alignof(cpp_type);
};

}  // namespace flashck