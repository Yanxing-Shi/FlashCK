#pragma once

#include <string>
#include <unordered_map>
#include <variant>

/*
norm profiling cache entries
*/

namespace lightinfer {

struct NormQueryEntry {
    std::string x_dtype_;
    std::string y_dtype_;
    std::string smooth_scale_dtype_;
    std::string y_scale_dtype_;

    std::string op_name_;
    std::string device_;
    std::string epilogue_;
    std::string exec_entry_sha1_;

    std::string fused_add_;
    std::string fused_quant_;
};

// norm cache entry
struct NormRecordEntry {
    std::string x_dtype_;
    std::string y_dtype_;
    std::string smooth_scale_dtype_;
    std::string y_scale_dtype_;

    std::string op_name_;
    std::string device_;
    std::string epilogue_;
    std::string exec_entry_;
    std::string exec_entry_sha1_;

    std::string fused_add_;
    std::string fused_quant_;

    std::string algo_;
};

}  // namespace lightinfer