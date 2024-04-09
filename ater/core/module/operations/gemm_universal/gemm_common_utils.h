#pragma once

#include <filesystem>
#include <functional>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "ater/core/utils/log.h"

namespace ater {

// inline std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> GroupGemmInverseKeyFunc(const std::string&
// key)
// {
//     std::vector<int> m, n, k;
//     std::regex       pattern("(\\d+)");
//     std::smatch      m_;
//     std::string      s = key;
//     while (std::regex_search(s, m_, pattern)) {
//         m.push_back(std::stoi(m_[0]));
//         s = m_.suffix().str();
//         n.push_back(std::stoi(m_[0]));
//         s = m_.suffix().str();
//         k.push_back(std::stoi(m_[0]));
//         s = m_.suffix().str();
//     }
//     return std::make_tuple(m, n, k);
// }

inline const std::vector<int> GemmInverseKeyFunc(const std::string& key)
{
    std::vector<int> tmp;
    std::regex       pattern("(\\d+)");
    std::smatch      m;
    std::string      s = key;
    while (std::regex_search(s, m, pattern)) {
        tmp.push_back(std::stoi(m[0]));
        s = m.suffix().str();
    }
    return tmp;
}

inline bool
CheckWithRetries(const std::filesystem::path& exe_path, const int max_attempts = 3, const int delay_seconds = 5)
{
    int attempts = 0;
    while (attempts < max_attempts) {
        if (std::filesystem::exists(exe_path)) {
            return true;
        }
        attempts++;
        VLOG(1) << "Attempt " << attempts << " of " << max_attempts << " failed. Retrying in " << delay_seconds
                << " seconds.";
        std::this_thread::sleep_for(std::chrono::seconds(delay_seconds));
    }
    return false;
}

}  // namespace ater