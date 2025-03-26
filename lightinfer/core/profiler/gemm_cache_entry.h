#pragma once

#include <string>
#include <unordered_map>
#include <variant>

/*
GEMM profiling cache entries
*/

namespace lightinfer {

// Gemm query entry
struct GemmQueryEntry {

    GemmQueryEntry(std::string a_dtype,
                   std::string b_dtype,
                   std::string c_dtype,
                   std::string acc_dtype,
                   std::string layout,
                   std::string op_name,
                   std::string device,
                   std::string epilogue,
                   std::string exec_entry_sha1,
                   std::string pshape):
        a_dtype_(a_dtype),
        b_dtype_(b_dtype),
        c_dtype_(c_dtype),
        acc_dtype_(acc_dtype),
        layout_(layout),
        op_name_(op_name),
        device_(device),
        epilogue_(epilogue),
        exec_entry_sha1_(exec_entry_sha1),
        pshape_(pshape)
    {
    }

    std::string a_dtype_;
    std::string b_dtype_;
    std::string c_dtype_;
    std::string acc_dtype_;
    std::string layout_;

    std::string op_name_;
    std::string device_;
    std::string epilogue_;
    std::string exec_entry_sha1_;
    std::string pshape_;
};

struct GemmRecordEntry {

    GemmRecordEntry(std::string exec_entry,
                    std::string exec_entry_sha1,
                    std::string a_dtype,
                    std::string b_dtype,
                    std::string c_dtype,
                    std::string acc_dtype,
                    std::string layout,
                    std::string op_name,
                    std::string epilogue,
                    std::string pshape,
                    std::string device,
                    std::string algo,
                    int64_t     split_k):
        exec_entry_(exec_entry),
        exec_entry_sha1_(exec_entry_sha1),
        a_dtype_(a_dtype),
        b_dtype_(b_dtype),
        c_dtype_(c_dtype),
        acc_dtype_(acc_dtype),
        layout_(layout),
        op_name_(op_name),
        epilogue_(epilogue),
        pshape_(pshape),
        device_(device),
        algo_(algo),
        split_k_(split_k)
    {
    }

    std::string exec_entry_;
    std::string exec_entry_sha1_;
    std::string a_dtype_;
    std::string b_dtype_;
    std::string c_dtype_;
    std::string acc_dtype_;
    std::string layout_;
    std::string op_name_;
    std::string epilogue_;
    std::string pshape_;
    std::string device_;
    std::string algo_;
    int64_t     split_k_;
};

}  // namespace lightinfer