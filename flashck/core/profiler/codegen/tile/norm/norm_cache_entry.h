#pragma once
#include <string>
#include <unordered_map>
#include <variant>

// Normalization layer profiling cache data structures

namespace flashck {

// Query parameters for normalization layer performance lookup
struct NormQueryEntry {
    // Data type configurations
    std::string x_dtype_;             // Input tensor data type (e.g. "float32")
    std::string y_dtype_;             // Output tensor data type
    std::string smooth_scale_dtype_;  // Smoothing scale data type
    std::string y_scale_dtype_;       // Output scaling factor data type

    // Operation configuration
    std::string op_name_;          // Normalization type (e.g. "layer_norm")
    std::string device_;           // Execution device (e.g. "CUDA:0")
    std::string epilogue_;         // Post-processing operations (e.g. "relu")
    std::string exec_entry_sha1_;  // Config parameters hash (SHA-1)

    // Fusion flags
    std::string fused_add_;    // Fused add operation status ("enabled"/"disabled")
    std::string fused_quant_;  // Quantization fusion status ("enabled"/"disabled")
};

// Cached normalization layer performance results
struct NormRecordEntry {
    // Data type configurations
    std::string x_dtype_;             // Input tensor data type
    std::string y_dtype_;             // Output tensor data type
    std::string smooth_scale_dtype_;  // Smoothing scale data type
    std::string y_scale_dtype_;       // Output scaling factor data type

    // Operation configuration
    std::string op_name_;          // Normalization operator variant
    std::string device_;           // Hardware accelerator type
    std::string epilogue_;         // Combined post-processing ops
    std::string exec_entry_;       // Unique execution configuration ID
    std::string exec_entry_sha1_;  // Hashed execution parameters

    // Fusion features
    std::string fused_add_;    // Fused element-wise addition flag
    std::string fused_quant_;  // Integrated quantization flag

    // Optimization parameters
    std::string algo_;  // Implementation algorithm (e.g. "group_norm_split")
};

}  // namespace flashck
