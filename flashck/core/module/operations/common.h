#pragma once

#include <map>
#include <regex>
#include <string>
#include <vector>

namespace flash_ck {

std::map<std::string, int> ExtractWorkLoad(const std::string& key)
{
    std::map<std::string, int> result;

    // Handle equality expressions like "M == 2"
    std::regex  eq_pattern(R"((\w+)\s*==\s*(\d+))");
    std::smatch match;
    std::string remaining = key;

    while (std::regex_search(remaining, match, eq_pattern)) {
        std::string var_name = match[1].str();
        int         value    = std::stoi(match[2].str());
        result[var_name]     = value;
        remaining            = match.suffix().str();
    }

    // Handle range expressions like "M>=2 && M<=9"
    // Extract the minimum value as representative for ranges
    std::regex range_pattern(R"((\w+)\s*>=\s*(\d+)\s*&&\s*\1\s*<=\s*(\d+))");
    remaining = key;

    while (std::regex_search(remaining, match, range_pattern)) {
        std::string var_name = match[1].str();
        int         min_val  = std::stoi(match[2].str());
        int         max_val  = std::stoi(match[3].str());
        // For ranges, we'll take the minimum value as the representative
        result[var_name] = min_val;
        remaining        = match.suffix().str();
    }

    return result;
}

std::string GenWorkLoad(const std::map<std::string, std::vector<int64_t>>& name_value_mapping)
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