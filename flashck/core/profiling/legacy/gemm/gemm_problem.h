#pragma once

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "flashck/core/profiling/legacy/gemm/gemm_library.h"
#include "flashck/core/profiling/problem_base.h"
#include "flashck/core/utils/dtype.h"

namespace flashck {

/**
 * @class GemmProblem
 * @brief Represents a General Matrix Multiplication (GEMM) operation problem configuration
 *
 * This class encapsulates all the parameters needed to define a GEMM operation,
 * including matrix dimensions, data types, layouts, epilogue operations, and
 * leading dimensions. It provides serialization capabilities for problem
 * representation and comparison.
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
     * @brief Serialize the problem configuration to JSON format
     * @return JSON string representation of the problem parameters
     */

    std::string SerializeImpl()
    {
        std::ostringstream oss;
        oss << "{"
            << "\"kind\": \"" << GetGemmKindName(kind_) << "\", "
            << "\"epilogue\": \"" << GetEpilogueName(epilogue_) << "\", "
            << "\"a_dtype\": \"" << DataTypeToString(a_dtype_) << "\", "
            << "\"b_dtype\": \"" << DataTypeToString(b_dtype_) << "\", "
            << "\"c_dtype\": \"" << DataTypeToString(c_dtype_) << "\", "
            << "\"acc_dtype\": \"" << DataTypeToString(acc_dtype_) << "\", "
            << "\"e_dtype\": \"" << DataTypeToString(e_dtype_) << "\", "
            << "\"ds_dtype\": [";
        for (size_t i = 0; i < ds_dtype_.size(); ++i) {
            oss << "\"" << DataTypeToString(ds_dtype_[i]) << "\"";
            if (i < ds_dtype_.size() - 1) {
                oss << ", ";
            }
        }
        oss << "], "
            << "\"a_layout\": \"" << GetLayoutName(a_layout_) << "\", "
            << "\"b_layout\": \"" << GetLayoutName(b_layout_) << "\", "
            << "\"c_layout\": \"" << GetLayoutName(c_layout_) << "\", "
            << "\"ds_layout\": [";
        for (size_t i = 0; i < ds_layout_.size(); ++i) {
            oss << "\"" << GetLayoutName(ds_layout_[i]) << "\"";
            if (i < ds_layout_.size() - 1) {
                oss << ", ";
            }
        }
        oss << "], "
            << "\"batch\": " << batch_ << ", "
            << "\"m\": " << m_ << ", "
            << "\"n\": " << n_ << ", "
            << "\"k\": " << k_ << ", "
            << "\"lda\": " << lda_ << ", "
            << "\"ldb\": " << ldb_ << ", "
            << "\"ldc\": " << ldc_ << ", "
            << "\"ldd\": " << ldd_ << "}";
        return oss.str();
    }

    // ====================== GEMM Operation Configuration ======================

    GemmKind     kind_;      ///< Type of GEMM operation (standard, multiple-D, batched)
    EpilogueType epilogue_;  ///< Epilogue operation applied after GEMM

    // Data type specifications
    DataType              a_dtype_;    ///< Matrix A data type
    DataType              b_dtype_;    ///< Matrix B data type
    DataType              c_dtype_;    ///< Matrix C (output) data type
    DataType              acc_dtype_;  ///< Accumulator data type for intermediate results
    DataType              e_dtype_;    ///< Epilogue tensor data type
    std::vector<DataType> ds_dtype_;   ///< Additional D tensor data types (for multiple-D GEMM)

    // Layout specifications
    LayoutType              a_layout_;   ///< Matrix A memory layout (row/column major)
    LayoutType              b_layout_;   ///< Matrix B memory layout (row/column major)
    LayoutType              c_layout_;   ///< Matrix C memory layout (row/column major)
    std::vector<LayoutType> ds_layout_;  ///< Additional D tensor layouts

    // Matrix dimensions and batch configuration
    int64_t batch_;  ///< Batch size (for batched GEMM operations)
    int64_t m_;      ///< Number of rows in matrix A and C
    int64_t n_;      ///< Number of columns in matrix B and C
    int64_t k_;      ///< Number of columns in matrix A / rows in matrix B

    // Leading dimensions (stride information)
    int64_t lda_;  ///< Leading dimension of matrix A
    int64_t ldb_;  ///< Leading dimension of matrix B
    int64_t ldc_;  ///< Leading dimension of matrix C
    int64_t ldd_;  ///< Leading dimension of additional D tensors
};
}  // namespace flashck