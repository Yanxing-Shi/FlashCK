#pragma once

#include "lightinfer/core/graph/shape.h"
#include "lightinfer/core/utils/backend.h"
#include "lightinfer/core/utils/dtype.h"
#include "lightinfer/core/utils/enforce.h"

#include "lightinfer/core/utils/log.h"

namespace lightinfer {

class FeedData {
public:
    FeedData() = default;
    FeedData(const BackendType backend_type,
             const DataType    dtype,
             const Shape       shape,
             const void*       data,
             const hipStream_t stream = nullptr);
    FeedData(const BackendType         backend_type,
             const DataType            dtype,
             const Shape               shape,
             const void*               data,
             const std::vector<size_t> offset,
             const hipStream_t         stream = nullptr);

    size_t GetSize() const;
    size_t GetSizeBytes() const;

    std::string GetBackendTypeStr() const;
    std::string ToString() const;

    void            SaveNpy(const std::string& filename) const;
    static FeedData LoadNpy(const std::string& npy_file, const BackendType backend_type);

    template<typename T>
    inline T GetValue(size_t index) const
    {
        LI_ENFORCE_EQ(
            BackendTypeToString(backend_type_),
            std::string("CPU"),
            Unavailable("GetValue is only supported for CPU tensors, but got:{} ", BackendTypeToString(backend_type_)));

        LI_ENFORCE_NOT_NULL(data_, Unavailable("data is nullptr."));
        LI_ENFORCE_LT(
            index, std::get<1>(shape_.GetElementSizeTuple()), Unavailable("index is larger than buffer size, "));

        if (CppTypeToDataType<T>::Type() != dtype_) {
            LI_THROW(Unavailable("getVal with type {}, but data type is: {}",
                                 DataTypeToString(CppTypeToDataType<T>::Type()),
                                 DataTypeToString(dtype_)));
        }

        return ((T*)data_)[index];
    }

    template<typename T>
    inline T GetValue() const
    {
        if (CppTypeToDataType<T>::Type() != dtype_) {
            LI_THROW(Unavailable("getVal with type {}, but data type is: {}",
                                 DataTypeToString(CppTypeToDataType<T>::Type()),
                                 DataTypeToString(dtype_)));
        }
        return GetValue<T>(0);
    }

    template<typename T>
    inline T* GetPtr() const
    {
        if (CppTypeToDataType<T>::Type() != dtype_) {
            LI_THROW(Unavailable("getVal with type {}, but data type is: {}",
                                 DataTypeToString(CppTypeToDataType<T>::Type()),
                                 DataTypeToString(dtype_)));
        }
        return (T*)data_;
    }

    template<typename T>
    inline void SetPtr(T* data)
    {
        if (CppTypeToDataType<T>::Type() != dtype_) {
            LI_THROW(Unavailable("getVal with type {}, but data type is: {}",
                                 DataTypeToString(CppTypeToDataType<T>::Type()),
                                 DataTypeToString(dtype_)));
        }
        data_ = (void*)data;
    }

    inline void* GetPtrWithOffset(size_t offset) const
    {
        if (data_ == nullptr) {
            return (void*)data_;
        }
        else {
            LI_ENFORCE_LT(offset, GetSize(), Unavailable("offset {} is larger than buffer size{}", offset, GetSize()));

            return (void*)((char*)data_ + offset * SizeOf(dtype_));
        }
    }

    template<typename T>
    inline T* GetPtrWithOffset(size_t offset) const
    {
        if (CppTypeToDataType<T>::Type() != dtype_) {
            LI_THROW(Unavailable("getVal with type {}, but data type is: {}",
                                 DataTypeToString(CppTypeToDataType<T>::Type()),
                                 DataTypeToString(dtype_)));
        }
        if (data_ == nullptr) {
            return (T*)data_;
        }
        else {
            LI_ENFORCE_LT(offset, GetSize(), Unavailable("offset {} is larger than buffer size{}", offset, GetSize()));
            return ((T*)data_) + offset;
        }
    }

    template<typename T>
    inline T GetMaxValue() const
    {
        if (CppTypeToDataType<T>::Type() != dtype_) {
            LI_THROW(Unavailable("getVal with type {}, but data type is: {}",
                                 DataTypeToString(CppTypeToDataType<T>::Type()),
                                 DataTypeToString(dtype_)));
        }

        LI_ENFORCE_GT(shape_.GetNumDim(), 0, Unavailable("Should be a non-empty tensor."));
        LI_ENFORCE_EQ(backend_type_ == BackendType::CPU || backend_type_ == BackendType::CPU_PINNED,
                      true,
                      Unavailable("GetMinValue() supports CPU or CPU_PINNED tensor."));

        size_t max_idx = 0;
        T      max_val = GetValue<T>(max_idx);
        for (size_t i = 1; i < GetSize(); ++i) {
            T val = GetValue<T>(i);
            if (val > max_val) {
                max_idx = i;
                max_val = val;
            }
        }
        return max_val;
    }

    template<typename T>
    inline T GetMinValue() const
    {
        if (CppTypeToDataType<T>::Type() != dtype_) {
            LI_THROW(Unavailable("getVal with type {}, but data type is: {}",
                                 DataTypeToString(CppTypeToDataType<T>::Type()),
                                 DataTypeToString(dtype_)));
        }

        LI_ENFORCE_GT(shape_.GetNumDim(), 0, Unavailable("Should be a non-empty tensor."));
        LI_ENFORCE_EQ(backend_type_ == BackendType::CPU || backend_type_ == BackendType::CPU_PINNED,
                      true,
                      Unavailable("GetMinValue() supports CPU or CPU_PINNED tensor."));

        size_t min_idx = 0;
        T      min_val = GetValue<T>(min_idx);
        for (size_t i = 1; i < GetSize(); ++i) {
            T val = GetValue<T>(i);
            if (val < min_val) {
                min_idx = i;
                min_val = val;
            }
        }
        return min_val;
    }

    template<typename T>
    inline T Any(T val) const
    {
        if (CppTypeToDataType<T>::Type() != dtype_) {
            LI_THROW(Unavailable("getVal with type {}, but data type is: {}",
                                 DataTypeToString(CppTypeToDataType<T>::Type()),
                                 DataTypeToString(dtype_)));
        }

        LI_ENFORCE_GT(shape_.GetNumDim(), 0, Unavailable("Should be a non-empty tensor."));
        LI_ENFORCE_EQ(backend_type_ == BackendType::CPU || backend_type_ == BackendType::CPU_PINNED,
                      true,
                      Unavailable("All() supports CPU or CPU_PINNED tensor."));

        for (size_t i = 0; i < GetSize(); ++i) {
            if (GetValue<T>(i) == val) {
                return true;
            }
        }
        return false;
    }

    template<typename T>
    inline T All(T val) const
    {
        if (CppTypeToDataType<T>::Type() != dtype_) {
            LI_THROW(Unavailable("getVal with type {}, but data type is: {}",
                                 DataTypeToString(CppTypeToDataType<T>::Type()),
                                 DataTypeToString(dtype_)));
        }

        LI_ENFORCE_GT(shape_.GetNumDim(), 0, Unavailable("Should be a non-empty tensor."));
        LI_ENFORCE_EQ(backend_type_ == BackendType::CPU || backend_type_ == BackendType::CPU_PINNED,
                      true,
                      Unavailable("All() supports CPU or CPU_PINNED tensor."));

        for (size_t i = 0; i < GetSize(); ++i) {
            if (GetValue<T>(i) != val) {
                return false;
            }
        }
        return true;
    }

    FeedData Slice(const Shape& shape, const size_t offset) const;

    const BackendType         backend_type_ = BackendType::GPU;
    const DataType            dtype_        = DataType::UNDEFINED;
    const Shape               shape_;
    const void*               data_;
    const std::vector<size_t> offsets_{};
    const hipStream_t         stream_ = nullptr;

private:
    static void ParseNpyIntro(FILE*& f_ptr, uint32_t& header_len, uint32_t& start_data);
    static int  ParseNpyHeader(FILE*& f_ptr, uint32_t header_len, DataType& type, Shape& shape);
};

}  // namespace lightinfer