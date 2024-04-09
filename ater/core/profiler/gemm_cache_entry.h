#pragma once

#include <string>
#include <unordered_map>
#include <variant>

/*
GEMM profiling cache entries
*/

namespace ater {

// Gemm query entry
struct GemmQueryEntry {
    GemmQueryEntry(int         dtype_a,
                   int         dtype_b,
                   int         dtype_c,
                   int         dtype_acc,
                   int         major_a,
                   int         major_b,
                   int         major_c,
                   std::string op_name,
                   std::string device,
                   int         epilogue,
                   std::string exec_entry_sha1,
                   std::string pshape):
        dtype_a_(dtype_a),
        dtype_b_(dtype_b),
        dtype_c_(dtype_c),
        dtype_acc_(dtype_acc),
        major_a_(major_a),
        major_b_(major_b),
        major_c_(major_c),
        op_name_(op_name),
        device_(device),
        epilogue_(epilogue),
        exec_entry_sha1_(exec_entry_sha1),
        pshape_(pshape)
    {
    }

    std::unordered_map<std::string, std::variant<int, std::string>> GetAttrsMap()
    {
        std::unordered_map<std::string, std::variant<int, std::string>> instance_map{
            {"dtype_a", dtype_a_},
            {"dtype_b", dtype_b_},
            {"dtype_c", dtype_c_},
            {"dtype_acc", dtype_acc_},
            {"major_a", major_a_},
            {"major_b", major_b_},
            {"major_c", major_c_},
            {"op_name", op_name_},
            {"device", device_},
            {"epilogue", epilogue_},
            {"exec_entry_sha1", exec_entry_sha1_},
            {"pshape", pshape_}};

        return instance_map;
    }

    int         dtype_a_;
    int         dtype_b_;
    int         dtype_c_;
    int         dtype_acc_;
    int         major_a_;
    int         major_b_;
    int         major_c_;
    std::string op_name_;
    std::string device_;
    int         epilogue_;
    std::string exec_entry_sha1_;
    std::string pshape_;
};

// Profile result record entry
struct GemmRecordEntry {
    GemmRecordEntry(std::string exec_entry,
                    std::string exec_entry_sha1,
                    int         dtype_a,
                    int         dtype_b,
                    int         dtype_c,
                    int         dtype_acc,
                    int         major_a,
                    int         major_b,
                    int         major_c,
                    std::string op_name,
                    int         epilogue,
                    std::string pshape,
                    std::string device,
                    std::string algo,
                    int         workspace,
                    int         split_k):
        exec_entry_(exec_entry),
        exec_entry_sha1_(exec_entry_sha1),
        dtype_a_(dtype_a),
        dtype_b_(dtype_b),
        dtype_c_(dtype_c),
        dtype_acc_(dtype_acc),
        major_a_(major_a),
        major_b_(major_b),
        major_c_(major_c),
        op_name_(op_name),
        epilogue_(epilogue),
        pshape_(pshape),
        device_(device),
        algo_(algo),
        workspace_(workspace),
        split_k_(split_k)
    {
    }

    std::unordered_map<std::string, std::variant<int, std::string>> GetAttrsMap()
    {
        std::unordered_map<std::string, std::variant<int, std::string>> instance_map{
            {"exec_entry", exec_entry_},
            {"exec_entry_sha1", exec_entry_sha1_},
            {"dtype_a", dtype_a_},
            {"dtype_b", dtype_b_},
            {"dtype_c", dtype_c_},
            {"dtype_acc", dtype_acc_},
            {"major_a", major_a_},
            {"major_b", major_b_},
            {"major_c", major_c_},
            {"op_name", op_name_},
            {"device", device_},
            {"epilogue", epilogue_},
            {"pshape", pshape_},
            {"device", device_},
            {"algo", algo_},
            {"workspace", workspace_},
            {"split_k", split_k_}};

        return instance_map;
    }

    std::string exec_entry_;
    std::string exec_entry_sha1_;
    int         dtype_a_;
    int         dtype_b_;
    int         dtype_c_;
    int         dtype_acc_;
    int         major_a_;
    int         major_b_;
    int         major_c_;
    std::string op_name_;
    int         epilogue_;
    std::string pshape_;
    std::string device_;
    std::string algo_;
    int         workspace_;
    int         split_k_;
};

}  // namespace ater