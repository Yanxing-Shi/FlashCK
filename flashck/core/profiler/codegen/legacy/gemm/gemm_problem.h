#pragma once

namespace flashck {

struct GemmProblem {
    GemmOperationKind operation_kind_;
    // element-wise operation
    TensorOperation epilogue_op_ = TensorOperation::PassThrough;

    // shape
    int64_t batch_;
    int64_t m_;
    int64_t n_;
    int64_t k_;
    int64_t lda_;
    int64_t ldb_;
    int64_t ldc_;
    int64_t ldd_;

    // data type
    DataType              a_dtype_   = DataType::UNDEFINED;
    DataType              b_dtype_   = DataType::UNDEFINED;
    DataType              c_dtype_   = DataType::UNDEFINED;
    DataType              acc_dtype_ = DataType::UNDEFINED;
    std::vector<DataType> ds_dtype_  = {};
    DataType              e_dtype_   = DataType::UNDEFINED;

    // layout
    DataLayout              layout_    = DataLayout::UNDEFINED;
    std::vector<LayoutType> ds_layout_ = {};
};
}  // namespace flashck