#pragma once

#include <string>

/*
fmha profiling cache entries
*/

namespace lightinfer {

struct FmhaQueryEntry {
    std::string dtype_;

    std::string mask_;
    std::string bias_;
    std::string mode_;

    int64_t rotary_dim_;

    int64_t paged_block_size_;
    bool    use_batch_cache_idx_;

    std::string op_name_;
    std::string device_;
    std::string epilogue_;
    std::string exec_entry_sha1_;
};

struct FmhaRecordEntry {
    std::string dtype_;

    std::string mask_;
    std::string bias_;
    std::string mode_;

    int64_t rotary_dim_;

    int64_t paged_block_size_;
    bool    use_batch_cache_idx_;

    std::string op_name_;
    std::string device_;
    std::string epilogue_;
    std::string exec_entry_;
    std::string exec_entry_sha1_;

    int64_t     num_splits_;
    std::string algo_;
};

}  // namespace lightinfer