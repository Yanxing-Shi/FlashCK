#pragma once

#include <string>
#include <unordered_map>
#include <variant>

// GEMM Profiling Cache entry

namespace flashck {

// GEMM query parameters for cache lookup. Represents a unique combination of
// matrix operation characteristics that identifies a specific GEMM configuration.
struct GemmQueryEntry {
    // Matrix data types
    std::string a_dtype_;    // Data type of left-hand matrix operand (A)
    std::string b_dtype_;    // Data type of right-hand matrix operand (B)
    std::string c_dtype_;    // Data type of output matrix (C)
    std::string acc_dtype_;  // Accumulator data type for intermediate calculations

    // Matrix layout characteristics
    std::string layout_;  // Memory layout (e.g., row-major, column-major)

    // Operation metadata
    std::string op_name_;      // Name of the GEMM operation variant
    std::string device_name_;  // Target execution device (e.g., "CUDA:0")
    std::string epilogue_;     // Post-processing operation (e.g., relu, bias_add)

    // Execution fingerprint
    std::string exec_entry_sha1_;  // SHA-1 hash of critical execution parameters
    std::string pshape_;           // Problem shape (M, N, K dimensions)
};

// Record of executed GEMM operation with performance characteristics. Stores
// both configuration parameters and runtime results for performance analysis.
struct GemmRecordEntry {
    // Execution identifiers
    std::string exec_entry_;       // Unique identifier for execution entry
    std::string exec_entry_sha1_;  // SHA-1 hash of execution parameters

    // Matrix characteristics
    std::string a_dtype_;    // Data type of matrix A
    std::string b_dtype_;    // Data type of matrix B
    std::string c_dtype_;    // Data type of matrix C
    std::string acc_dtype_;  // Accumulator data type
    std::string layout_;     // Memory layout

    // Operation metadata
    std::string op_name_;        // GEMM operation variant name
    std::string epilogue_;       // Post-processing operation
    std::string permute_shape_;  // Tensor permutation dimensions
    std::string device_name_;    // Execution device identifier

    // Algorithm selection
    std::string algo_;     // Specific algorithm implementation used
    int64_t     split_k_;  // K-dimension splitting factor for parallel execution
};

}  // namespace flashck