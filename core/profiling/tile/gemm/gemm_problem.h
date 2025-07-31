#pragma once

#include <sstream>

#include "core/profiling/problem_base.h"
#include "core/profiling/tile/gemm/gemm_library.h"
#include "core/utils/dtype.h"

namespace flashck {
namespace tile {

/**
 * @class GemmProblem
 * @brief Represents a GEMM (General Matrix Multiply) operation problem configuration
 *
 * This class encapsulates all the parameters needed to define a GEMM operation,
 * including data types, dimensions, fusion options, and bias settings. It provides
 * serialization capabilities for problem representation and comparison.
 */
class GemmProblem: public ProblemBase<GemmProblem> {
public:
    /**
     * @brief Get the type name of this problem
     * @return String identifier for GEMM problems
     */
    std::string GetTypeImpl()
    {
        return "GemmProblem";
    }


    /**
     * @brief Get the type name of this problem
     * @return String identifier for GEMM problems
     */
    std::string SerializeImpl()
    {
        std::ostringstream oss;
        oss << "{";
        oss << "\"kind\": \"" << GetGemmKindName(kind_) << "\", ";
        oss << "\"elementwise_kind\": \"" << GetElementwiseKindName(elementwise_kind_) << "\", ";
        oss << "\"a_dtype\": \"" << DataTypeToString(a_dtype_) << "\", ";
        oss << "\"b_dtype\": \"" << DataTypeToString(b_dtype_) << "\", ";
        oss << "\"c_dtype\": \"" << DataTypeToString(c_dtype_) << "\", ";
        oss << "\"acc_dtype\": \"" << DataTypeToString(acc_dtype_) << "\", ";
        oss << "\"ds_dtype\": [";
        for (size_t i = 0; i < ds_dtype_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "\"" << DataTypeToString(ds_dtype_[i]) << "\"";
        }
        oss << "], ";
        oss << "\"a_layout\": \"" << GetLayoutTypeName(a_layout_) << "\", ";
        oss << "\"b_layout\": \"" << GetLayoutTypeName(b_layout_) << "\", ";
        oss << "\"c_layout\": \"" << GetLayoutTypeName(c_layout_) << "\", ";
        oss << "\"ds_layouts\": [";
        for (size_t i = 0; i < ds_layouts_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "\"" << GetLayoutTypeName(ds_layouts_[i]) << "\"";
        }
        oss << "], ";
        oss << "\"a_permute\": " << (a_permute_ ? "true" : "false") << ", ";
        oss << "\"b_permute\": " << (b_permute_ ? "true" : "false") << ", ";
        oss << "\"c_permute\": " << (c_permute_ ? "true" : "false") << ", ";
        oss << "\"use_structured_sparsity\": " << (use_structured_sparsity_ ? "true" : "false") << ", ";
        oss << "\"is_preshuffle\": " << (is_preshuffle_ ? "true" : "false") << ", ";
        oss << "\"batch_count\": " << batch_count_ << ", ";
        oss << "\"split_k\": " << split_k_ << ", ";
        oss << "\"group_count\": " << group_count_ << ", ";
        oss << "\"m\": " << m_ << ", ";
        oss << "\"n\": " << n_ << ", ";
        oss << "\"k\": " << k_ << ", ";
        oss << "\"a_stride\": " << a_stride_ << ", ";
        oss << "\"b_stride\": " << b_stride_ << ", ";
        oss << "\"c_stride\": " << c_stride_ << ", ";
        oss << "\"ds_stride\": [";
        for (size_t i = 0; i < ds_stride_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << ds_stride_[i];
        }
        oss << "]";
        oss << "}";

        return oss.str();
    }
          

    // ====================== Problem Configuration ======================

    GemmKind kind_;  ///< Type of gemm operation (Gemm, GemmMultiD, etc.)

    ElementwiseKind elementwise_kind_;  ///< Type of elementwise operation (PassThrough, Add, etc.)

    // Data type specifications
    DataType a_dtype_;             ///< Tensor a data type
    DataType b_dtype_;             ///< Tensor b data type
    DataType c_dtype_;             ///< Tensor c data type
    DataType acc_dtype_ = DataType::FLOAT32;           ///< Tensor accumulator data type
    std::vector<DataType> ds_dtype_;            ///< Tensor ds data type (Special for GemmMultiD)


    // Layout specifications
    LayoutType a_layout_;           ///< Tensor a layout
    LayoutType b_layout_;           ///< Tensor b layout
    LayoutType c_layout_;           ///< Tensor c layout
    std::vector<LayoutType> ds_layouts_;  ///< Tensor ds layouts (Special for GemmMultiD)

    // Other configurations
    bool a_permute_ = false;  ///< Whether to permute tensor a
    bool b_permute_ = false;  ///< Whether to permute tensor b
    bool c_permute_ = false;  ///< Whether to permute tensor c
    bool use_structured_sparsity_ = false;  ///< Whether to use structured sparsity
    bool is_preshuffle_ = false;  ///< Whether to use preshuffle
    bool is_persistent_ = false;  ///< Whether to use persistent kernel

    // Tensor dimensions
    int64_t batch_count_; ///< (special for BatchGemm)
    int64_t split_k_;
    int64_t group_count_ = 1; ///< (special for GroupGemm)
    int64_t m_;  
    int64_t n_;
    int64_t k_;  

    // Tensor stride
    int64_t a_stride_;             ///< Tensor a stride
    int64_t b_stride_;             ///< Tensor b stride
    int64_t c_stride_;             ///< Tensor c stride
    std::vector<int64_t> ds_stride_;            ///< Tensor ds stride (Special for GemmMultiD)

    // Batch Stride
    int64_t a_batch_stride_;
    int64_t b_batch_stride_;
    int64_t c_batch_stride_;

};

struct BatchGemmProblem: public ProblemBase<GemmProblem> {
public:
    /**
     * @brief Get the type name of this problem
     * @return String identifier for GEMM problems
     */
    std::string GetTypeImpl()
    {
        return "GemmProblem";
    }


    /**
     * @brief Get the type name of this problem
     * @return String identifier for GEMM problems
     */
    std::string SerializeImpl()
    {
        std::ostringstream oss;
        oss << "{";
        oss << "\"kind\": \"" << GetGemmKindName(kind_) << "\", ";
        oss << "\"elementwise_kind\": \"" << GetElementwiseKindName(elementwise_kind_) << "\", ";
        oss << "\"a_dtype\": \"" << DataTypeToString(a_dtype_) << "\", ";
        oss << "\"b_dtype\": \"" << DataTypeToString(b_dtype_) << "\", ";
        oss << "\"c_dtype\": \"" << DataTypeToString(c_dtype_) << "\", ";
        oss << "\"acc_dtype\": \"" << DataTypeToString(acc_dtype_) << "\", ";
        oss << "\"ds_dtype\": [";
        for (size_t i = 0; i < ds_dtype_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "\"" << DataTypeToString(ds_dtype_[i]) << "\"";
        }
        oss << "], ";
        oss << "\"a_layout\": \"" << GetLayoutTypeName(a_layout_) << "\", ";
        oss << "\"b_layout\": \"" << GetLayoutTypeName(b_layout_) << "\", ";
        oss << "\"c_layout\": \"" << GetLayoutTypeName(c_layout_) << "\", ";
        oss << "\"ds_layouts\": [";
        for (size_t i = 0; i < ds_layouts_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "\"" << GetLayoutTypeName(ds_layouts_[i]) << "\"";
        }
        oss << "], ";
        oss << "\"a_permute\": " << (a_permute_ ? "true" : "false") << ", ";
        oss << "\"b_permute\": " << (b_permute_ ? "true" : "false") << ", ";
        oss << "\"c_permute\": " << (c_permute_ ? "true" : "false") << ", ";
        oss << "\"use_structured_sparsity\": " << (use_structured_sparsity_ ? "true" : "false") << ", ";
        oss << "\"is_preshuffle\": " << (is_preshuffle_ ? "true" : "false") << ", ";
        oss << "\"batch_count\": " << batch_count_ << ", ";
        oss << "\"split_k\": " << split_k_ << ", ";
        oss << "\"group_count\": " << group_count_ << ", ";
        oss << "\"m\": " << m_ << ", ";
        oss << "\"n\": " << n_ << ", ";
        oss << "\"k\": " << k_ << ", ";
        oss << "\"a_stride\": " << a_stride_ << ", ";
        oss << "\"b_stride\": " << b_stride_ << ", ";
        oss << "\"c_stride\": " << c_stride_ << ", ";
        oss << "\"ds_stride\": [";
        for (size_t i = 0; i < ds_stride_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << ds_stride_[i];
        }
        oss << "]";
        oss << "}";

        return oss.str();
    }
          

    // ====================== Problem Configuration ======================

    GemmKind kind_;  ///< Type of gemm operation (Gemm, GemmMultiD, etc.)

    ElementwiseKind elementwise_kind_;  ///< Type of elementwise operation (PassThrough, Add, etc.)

    // Data type specifications
    DataType a_dtype_;             ///< Tensor a data type
    DataType b_dtype_;             ///< Tensor b data type
    DataType c_dtype_;             ///< Tensor c data type
    DataType acc_dtype_ = DataType::FLOAT32;           ///< Tensor accumulator data type
    std::vector<DataType> ds_dtype_;            ///< Tensor ds data type (Special for GemmMultiD)


    // Layout specifications
    LayoutType a_layout_;           ///< Tensor a layout
    LayoutType b_layout_;           ///< Tensor b layout
    LayoutType c_layout_;           ///< Tensor c layout
    std::vector<LayoutType> ds_layouts_;  ///< Tensor ds layouts (Special for GemmMultiD)

    // Other configurations
    bool a_permute_ = false;  ///< Whether to permute tensor a
    bool b_permute_ = false;  ///< Whether to permute tensor b
    bool c_permute_ = false;  ///< Whether to permute tensor c
    bool use_structured_sparsity_ = false;  ///< Whether to use structured sparsity
    bool is_preshuffle_ = false;  ///< Whether to use preshuffle
    bool is_persistent_ = false;  ///< Whether to use persistent kernel

    // Tensor dimensions
    int64_t batch_count_; ///< (special for BatchGemm)
    int64_t split_k_;
    int64_t group_count_ = 1; ///< (special for GroupGemm)
    int64_t m_;  
    int64_t n_;
    int64_t k_;  

    // Tensor stride
    int64_t a_stride_;             ///< Tensor a stride
    int64_t b_stride_;             ///< Tensor b stride
    int64_t c_stride_;             ///< Tensor c stride
    std::vector<int64_t> ds_stride_;            ///< Tensor ds stride (Special for GemmMultiD)

    // Batch Stride
    int64_t a_batch_stride_;
    int64_t b_batch_stride_;
    int64_t c_batch_stride_;

};

struct FlatmmProblem: public ProblemBase<GemmProblem> {
public:
    /**
     * @brief Get the type name of this problem
     * @return String identifier for GEMM problems
     */
    std::string GetTypeImpl()
    {
        return "GemmProblem";
    }


    /**
     * @brief Get the type name of this problem
     * @return String identifier for GEMM problems
     */
    std::string SerializeImpl()
    {
        std::ostringstream oss;
        oss << "{";
        oss << "\"kind\": \"" << GetGemmKindName(kind_) << "\", ";
        oss << "\"elementwise_kind\": \"" << GetElementwiseKindName(elementwise_kind_) << "\", ";
        oss << "\"a_dtype\": \"" << DataTypeToString(a_dtype_) << "\", ";
        oss << "\"b_dtype\": \"" << DataTypeToString(b_dtype_) << "\", ";
        oss << "\"c_dtype\": \"" << DataTypeToString(c_dtype_) << "\", ";
        oss << "\"acc_dtype\": \"" << DataTypeToString(acc_dtype_) << "\", ";
        oss << "\"ds_dtype\": [";
        for (size_t i = 0; i < ds_dtype_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "\"" << DataTypeToString(ds_dtype_[i]) << "\"";
        }
        oss << "], ";
        oss << "\"a_layout\": \"" << GetLayoutTypeName(a_layout_) << "\", ";
        oss << "\"b_layout\": \"" << GetLayoutTypeName(b_layout_) << "\", ";
        oss << "\"c_layout\": \"" << GetLayoutTypeName(c_layout_) << "\", ";
        oss << "\"ds_layouts\": [";
        for (size_t i = 0; i < ds_layouts_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "\"" << GetLayoutTypeName(ds_layouts_[i]) << "\"";
        }
        oss << "], ";
        oss << "\"a_permute\": " << (a_permute_ ? "true" : "false") << ", ";
        oss << "\"b_permute\": " << (b_permute_ ? "true" : "false") << ", ";
        oss << "\"c_permute\": " << (c_permute_ ? "true" : "false") << ", ";
        oss << "\"use_structured_sparsity\": " << (use_structured_sparsity_ ? "true" : "false") << ", ";
        oss << "\"is_preshuffle\": " << (is_preshuffle_ ? "true" : "false") << ", ";
        oss << "\"batch_count\": " << batch_count_ << ", ";
        oss << "\"split_k\": " << split_k_ << ", ";
        oss << "\"group_count\": " << group_count_ << ", ";
        oss << "\"m\": " << m_ << ", ";
        oss << "\"n\": " << n_ << ", ";
        oss << "\"k\": " << k_ << ", ";
        oss << "\"a_stride\": " << a_stride_ << ", ";
        oss << "\"b_stride\": " << b_stride_ << ", ";
        oss << "\"c_stride\": " << c_stride_ << ", ";
        oss << "\"ds_stride\": [";
        for (size_t i = 0; i < ds_stride_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << ds_stride_[i];
        }
        oss << "]";
        oss << "}";

        return oss.str();
    }
          

    // ====================== Problem Configuration ======================

    GemmKind kind_;  ///< Type of gemm operation (Gemm, GemmMultiD, etc.)

    ElementwiseKind elementwise_kind_;  ///< Type of elementwise operation (PassThrough, Add, etc.)

    // Data type specifications
    DataType a_dtype_;             ///< Tensor a data type
    DataType b_dtype_;             ///< Tensor b data type
    DataType c_dtype_;             ///< Tensor c data type
    DataType acc_dtype_ = DataType::FLOAT32;           ///< Tensor accumulator data type
    std::vector<DataType> ds_dtype_;            ///< Tensor ds data type (Special for GemmMultiD)


    // Layout specifications
    LayoutType a_layout_;           ///< Tensor a layout
    LayoutType b_layout_;           ///< Tensor b layout
    LayoutType c_layout_;           ///< Tensor c layout
    std::vector<LayoutType> ds_layouts_;  ///< Tensor ds layouts (Special for GemmMultiD)

    // Other configurations
    bool a_permute_ = false;  ///< Whether to permute tensor a
    bool b_permute_ = false;  ///< Whether to permute tensor b
    bool c_permute_ = false;  ///< Whether to permute tensor c
    bool use_structured_sparsity_ = false;  ///< Whether to use structured sparsity
    bool is_preshuffle_ = false;  ///< Whether to use preshuffle
    bool is_persistent_ = false;  ///< Whether to use persistent kernel

    // Tensor dimensions
    int64_t batch_count_; ///< (special for BatchGemm)
    int64_t split_k_;
    int64_t group_count_ = 1; ///< (special for GroupGemm)
    int64_t m_;  
    int64_t n_;
    int64_t k_;  

    // Tensor stride
    int64_t a_stride_;             ///< Tensor a stride
    int64_t b_stride_;             ///< Tensor b stride
    int64_t c_stride_;             ///< Tensor c stride
    std::vector<int64_t> ds_stride_;            ///< Tensor ds stride (Special for GemmMultiD)

    // Batch Stride
    int64_t a_batch_stride_;
    int64_t b_batch_stride_;
    int64_t c_batch_stride_;

};

struct GemmMultiDProblem: public ProblemBase<GemmProblem> {
public:
    /**
     * @brief Get the type name of this problem
     * @return String identifier for GEMM problems
     */
    std::string GetTypeImpl()
    {
        return "GemmMultiDProblem";
    }


    /**
     * @brief Get the type name of this problem
     * @return String identifier for GEMM problems
     */
    std::string SerializeImpl()
    {
        std::ostringstream oss;
        oss << "{";
        oss << "\"kind\": \"" << GetGemmKindName(kind_) << "\", ";
        oss << "\"elementwise_kind\": \"" << GetElementwiseKindName(elementwise_kind_) << "\", ";
        oss << "\"a_dtype\": \"" << DataTypeToString(a_dtype_) << "\", ";
        oss << "\"b_dtype\": \"" << DataTypeToString(b_dtype_) << "\", ";
        oss << "\"c_dtype\": \"" << DataTypeToString(c_dtype_) << "\", ";
        oss << "\"acc_dtype\": \"" << DataTypeToString(acc_dtype_) << "\", ";
        oss << "\"ds_dtype\": [";
        for (size_t i = 0; i < ds_dtype_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "\"" << DataTypeToString(ds_dtype_[i]) << "\"";
        }
        oss << "], ";
        oss << "\"a_layout\": \"" << GetLayoutTypeName(a_layout_) << "\", ";
        oss << "\"b_layout\": \"" << GetLayoutTypeName(b_layout_) << "\", ";
        oss << "\"c_layout\": \"" << GetLayoutTypeName(c_layout_) << "\", ";
        oss << "\"ds_layouts\": [";
        for (size_t i = 0; i < ds_layouts_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "\"" << GetLayoutTypeName(ds_layouts_[i]) << "\"";
        }
        oss << "], ";
        oss << "\"a_permute\": " << (a_permute_ ? "true" : "false") << ", ";
        oss << "\"b_permute\": " << (b_permute_ ? "true" : "false") << ", ";
        oss << "\"c_permute\": " << (c_permute_ ? "true" : "false") << ", ";
        oss << "\"use_structured_sparsity\": " << (use_structured_sparsity_ ? "true" : "false") << ", ";
        oss << "\"is_preshuffle\": " << (is_preshuffle_ ? "true" : "false") << ", ";
        oss << "\"batch_count\": " << batch_count_ << ", ";
        oss << "\"split_k\": " << split_k_ << ", ";
        oss << "\"group_count\": " << group_count_ << ", ";
        oss << "\"m\": " << m_ << ", ";
        oss << "\"n\": " << n_ << ", ";
        oss << "\"k\": " << k_ << ", ";
        oss << "\"a_stride\": " << a_stride_ << ", ";
        oss << "\"b_stride\": " << b_stride_ << ", ";
        oss << "\"c_stride\": " << c_stride_ << ", ";
        oss << "\"ds_stride\": [";
        for (size_t i = 0; i < ds_stride_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << ds_stride_[i];
        }
        oss << "]";
        oss << "}";

        return oss.str();
    }
          

    // ====================== Problem Configuration ======================

    GemmKind kind_;  ///< Type of gemm operation (Gemm, GemmMultiD, etc.)

    ElementwiseKind elementwise_kind_;  ///< Type of elementwise operation (PassThrough, Add, etc.)

    // Data type specifications
    DataType a_dtype_;             ///< Tensor a data type
    DataType b_dtype_;             ///< Tensor b data type
    DataType c_dtype_;             ///< Tensor c data type
    DataType acc_dtype_ = DataType::FLOAT32;           ///< Tensor accumulator data type
    std::vector<DataType> ds_dtype_;            ///< Tensor ds data type (Special for GemmMultiD)


    // Layout specifications
    LayoutType a_layout_;           ///< Tensor a layout
    LayoutType b_layout_;           ///< Tensor b layout
    LayoutType c_layout_;           ///< Tensor c layout
    std::vector<LayoutType> ds_layouts_;  ///< Tensor ds layouts (Special for GemmMultiD)

    // Other configurations
    bool a_permute_ = false;  ///< Whether to permute tensor a
    bool b_permute_ = false;  ///< Whether to permute tensor b
    bool c_permute_ = false;  ///< Whether to permute tensor c
    bool use_structured_sparsity_ = false;  ///< Whether to use structured sparsity
    bool is_preshuffle_ = false;  ///< Whether to use preshuffle
    bool is_persistent_ = false;  ///< Whether to use persistent kernel

    // Tensor dimensions
    int64_t batch_count_; ///< (special for BatchGemm)
    int64_t split_k_;
    int64_t group_count_ = 1; ///< (special for GroupGemm)
    int64_t m_;  
    int64_t n_;
    int64_t k_;  

    // Tensor stride
    int64_t a_stride_;             ///< Tensor a stride
    int64_t b_stride_;             ///< Tensor b stride
    int64_t c_stride_;             ///< Tensor c stride
    std::vector<int64_t> ds_stride_;            ///< Tensor ds stride (Special for GemmMultiD)

    // Batch Stride
    int64_t a_batch_stride_;
    int64_t b_batch_stride_;
    int64_t c_batch_stride_;

};


} // namespace tile
}  // namespace flashck