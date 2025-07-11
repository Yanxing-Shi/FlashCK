#pragma once

#include <iomanip>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <fmt/format.h>

namespace flashck {

template<typename... Args>
inline void Fprintf(std::string& s, fmt::format_string<Args...> format, Args&&... args)
{
    fmt::vformat_to(std::back_inserter(s), format, fmt::make_format_args(args...));
}

inline std::string Sprintf()
{
    return "";
}

template<typename... Args>
inline std::string Sprintf(fmt::format_string<Args...> format, Args&&... args)
{
    std::string s;
    Fprintf(s, format, std::forward<Args>(args)...);
    return s;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec);

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& lod);

std::string HumanReadableSize(uint64_t bytes, int precision = 2);

}  // namespace flashck