#pragma once

namespace flash_ck {

// This structure is used to store the execution items for profiling
class ExecItem {
public:
    std::string profiling_key_;

    std::string exec_cond_;
    std::string instance_name_;
    PerfResult  perf_result_;
};

std::vector<int64_t> InvertExecKey(const std::string& key)
{
    std::vector<int64_t> tmp;
    std::regex           pattern("(\\d+)");
    std::smatch          m;
    std::string          s = key;
    while (std::regex_search(s, m, pattern)) {
        tmp.push_back(std::stoi(m[0]));
        s = m.suffix().str();
    }
    return tmp;
}

std::string GenExecKey(const std::map<std::string, std::vector<int64_t>>& name_value_mapping)
{
    std::vector<std::string> key_strs;
    for (auto& [name, values] : name_value_mapping) {
        if (values.size() == 1) {
            key_strs.emplace_back(Sprintf("{} == {}", name, values[0]));
        }
        else if (values.size() > 1) {
            key_strs.emplace_back(Sprintf("{} >= {} && {} <= {}", name, values[0], name, values.back()));
        }
        else {
            FC_THROW(Unavailable("norm input has empty dim values: {}", values[0]));
        }
    }

    return JoinStrings(key_strs, " && ");
}

}  // namespace flash_ck