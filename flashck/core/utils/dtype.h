#pragma once

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

template<DataType DT>
struct DataTypeTraits {
    static_assert(sizeof(DT) == 0, "Unsupported data type");
};

class TypeSystem {
public:
    template<DataType DT>
    static constexpr void ValidateType()
    {
        static_assert(DataTypeTraits<DT>::size == sizeof(typename DataTypeTraits<DT>::cpp_type), "Type size mismatch");
        static_assert(DataTypeTraits<DT>::alignment == alignof(typename DataTypeTraits<DT>::cpp_type),
                      "Alignment mismatch");
    }

    // static void ValidateConversion(DataType src, DataType dst)
    // {
    //     const size_t src_size = GetTypeSize(src);
    //     const size_t dst_size = GetTypeSize(dst);

    //     if (src_size < dst_size) {
    //         throw std::runtime_error("Potential precision loss in type conversion");
    //     }
    // }

    template<DataType DT>
    static constexpr const char* GetTypeFullName()
    {
        ValidateType<DT>();
        return DataTypeTraits<DT>::full_name;
    }

    template<DataType DT>
    static constexpr const char* GetTypeShortName()
    {
        ValidateType<DT>();
        return DataTypeTraits<DT>::short_name;
    }

    // static const char* GetTypeName(DataType dt)
    // {
    //     switch (dt) {
    //         case DataType::FLOAT8:
    //             return "float8";
    //         case DataType::BFLOAT8:
    //             return "bfloat8";
    //         default:
    //             throw std::invalid_argument("Unknown data type");
    //     }
    // }

    template<DataType DT>
    static constexpr size_t GetTypeSize()
    {
        validate_type<DT>();
        return DataTypeTraits<DT>::size;
    }

    // static size_t get_type_size(DataType dt)
    // {
    //     switch (dt) {
    //         case DataType::FLOAT8:
    //             return sizeof(float8);
    //         case DataType::BFLOAT8:
    //             return sizeof(bfloat8);

    //         default:
    //             throw std::invalid_argument("Unknown data type");
    //     }
    // }
};

}  // namespace flashck