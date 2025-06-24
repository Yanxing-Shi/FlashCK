#pragma once
#include <string>

// FMHA profiling cache data structures for performance analysis

namespace flashck {

// Query entry for looking up cached FMHA performance results
struct FmhaQueryEntry {
    // Data type configuration
    std::string dtype_;  // Base data type (e.g. "float16", "bfloat16")

    // Attention configuration
    std::string mask_type_;  // Attention mask type (e.g. "causal", "padding")
    std::string bias_type_;  // Positional bias type (e.g. "alibi", "relative")
    std::string mode_;       // Execution mode (e.g. "group", "batch")

    // Positional encoding
    int64_t rotary_dim_;  // Rotary position embedding dimensions (0 = disabled)

    // Memory optimization parameters
    int64_t paged_block_size_;     // Block size for paged memory (in tokens)
    bool    use_batch_cache_idx_;  // Enable batch cache index optimization

    // Operation metadata
    std::string op_name_;      // Operator variant name (e.g. "decoder_layer")
    std::string device_name_;  // Execution device (e.g. "CUDA:0")
    std::string epilogue_;     // Post-processing operations (e.g. "dropout")

    // Execution fingerprint
    std::string exec_entry_sha1_;  // SHA-1 hash of execution parameters
};

// Record entry storing actual FMHA performance results
struct FmhaRecordEntry {
    // Data type configuration
    std::string dtype_;  // Base data type

    // Attention configuration
    std::string mask_type_;  // Attention mask type
    std::string bias_type_;  // Positional bias type
    std::string mode_;       // Execution mode

    // Positional encoding
    int64_t rotary_dim_;  // Rotary position embedding dimensions

    // Memory optimization parameters
    int64_t paged_block_size_;     // Paged memory block size
    bool    use_batch_cache_idx_;  // Batch cache index status

    // Operation metadata
    std::string op_name_;      // Operator variant name
    std::string device_name_;  // Execution device
    std::string epilogue_;     // Post-processing chain

    // Execution identifiers
    std::string exec_entry_;       // Unique execution entry ID
    std::string exec_entry_sha1_;  // Hashed execution config

    // Optimization parameters
    int64_t     num_splits_;  // Number of splits for parallel processing
    std::string algo_;        // Algorithm implementation (e.g. "flash_v2")
};

}  // namespace flashck
