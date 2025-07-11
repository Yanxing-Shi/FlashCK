#pragma once

namespace flashck {

struct GemmQueryEntry {
    std::string a_dtype_;
    std::string b_dtype_;
    std::string c_dtype_;
    std::string acc_dtype_;

    std::string layout_;

    std::string op_name_;
    std::string device_name_;
    std::string epilogue_;

    std::string exec_entry_sha1_;
    std::string pshape_;
};

struct GemmRecordEntry {
    std::string exec_entry_;
    std::string exec_entry_sha1_;

    std::string a_dtype_;
    std::string b_dtype_;
    std::string c_dtype_;
    std::string acc_dtype_;
    std::string layout_;

    std::string op_name_;
    std::string epilogue_;
    std::string permute_shape_;
    std::string device_name_;

    std::string algo_;
    int64_t     split_k_;
};

}  // namespace flashck