#include "core/utils/printf.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace flashck {

// ==============================================================================
// Container Stream Operators Implementation
// ==============================================================================

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0)
            os << ", ";
        os << vec[i];
    }
    os << "]";
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& lod)
{
    os << "[";
    for (size_t i = 0; i < lod.size(); ++i) {
        if (i > 0)
            os << ", ";
        os << lod[i];  // Reuse 1D vector formatting
    }
    os << "]";
    return os;
}

template<typename K, typename V>
std::ostream& operator<<(std::ostream& os, const std::map<K, V>& map)
{
    os << "{";
    bool first = true;
    for (const auto& [key, value] : map) {
        if (!first)
            os << ", ";
        os << key << ": " << value;
        first = false;
    }
    os << "}";
    return os;
}

template<typename K, typename V>
std::ostream& operator<<(std::ostream& os, const std::unordered_map<K, V>& map)
{
    os << "{";
    bool first = true;
    for (const auto& [key, value] : map) {
        if (!first)
            os << ", ";
        os << key << ": " << value;
        first = false;
    }
    os << "}";
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::set<T>& set)
{
    os << "{";
    bool first = true;
    for (const auto& elem : set) {
        if (!first)
            os << ", ";
        os << elem;
        first = false;
    }
    os << "}";
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::unordered_set<T>& set)
{
    os << "{";
    bool first = true;
    for (const auto& elem : set) {
        if (!first)
            os << ", ";
        os << elem;
        first = false;
    }
    os << "}";
    return os;
}

template<typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, const std::pair<T1, T2>& pair)
{
    os << "(" << pair.first << ", " << pair.second << ")";
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::optional<T>& opt)
{
    if (opt.has_value()) {
        os << "Some(" << opt.value() << ")";
    }
    else {
        os << "None";
    }
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::shared_ptr<T>& ptr)
{
    if (ptr) {
        os << "shared_ptr(" << *ptr << ")";
    }
    else {
        os << "shared_ptr(nullptr)";
    }
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::unique_ptr<T>& ptr)
{
    if (ptr) {
        os << "unique_ptr(" << *ptr << ")";
    }
    else {
        os << "unique_ptr(nullptr)";
    }
    return os;
}

// ==============================================================================
// Explicit Template Instantiations
// ==============================================================================

// Vector instantiations
template std::ostream& operator<<(std::ostream& os, const std::vector<int>& vec);
template std::ostream& operator<<(std::ostream& os, const std::vector<int64_t>& vec);
template std::ostream& operator<<(std::ostream& os, const std::vector<float>& vec);
template std::ostream& operator<<(std::ostream& os, const std::vector<double>& vec);
template std::ostream& operator<<(std::ostream& os, const std::vector<std::string>& vec);
template std::ostream& operator<<(std::ostream& os, const std::vector<bool>& vec);

// Nested vector instantiations
template std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<int>>& lod);
template std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<int64_t>>& lod);
template std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<float>>& lod);
template std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<std::string>>& lod);

// Map instantiations
template std::ostream& operator<<(std::ostream& os, const std::map<std::string, int>& map);
template std::ostream& operator<<(std::ostream& os, const std::map<std::string, std::string>& map);
template std::ostream& operator<<(std::ostream& os, const std::unordered_map<std::string, int>& map);
template std::ostream& operator<<(std::ostream& os, const std::unordered_map<std::string, std::string>& map);

// Set instantiations
template std::ostream& operator<<(std::ostream& os, const std::set<int>& set);
template std::ostream& operator<<(std::ostream& os, const std::set<std::string>& set);
template std::ostream& operator<<(std::ostream& os, const std::unordered_set<int>& set);
template std::ostream& operator<<(std::ostream& os, const std::unordered_set<std::string>& set);

// Pair instantiations
template std::ostream& operator<<(std::ostream& os, const std::pair<int, int>& pair);
template std::ostream& operator<<(std::ostream& os, const std::pair<std::string, int>& pair);
template std::ostream& operator<<(std::ostream& os, const std::pair<std::string, std::string>& pair);

// Optional instantiations
template std::ostream& operator<<(std::ostream& os, const std::optional<int>& opt);
template std::ostream& operator<<(std::ostream& os, const std::optional<std::string>& opt);

// Pointer instantiations
template std::ostream& operator<<(std::ostream& os, const std::shared_ptr<int>& ptr);
template std::ostream& operator<<(std::ostream& os, const std::shared_ptr<std::string>& ptr);
template std::ostream& operator<<(std::ostream& os, const std::unique_ptr<int>& ptr);
template std::ostream& operator<<(std::ostream& os, const std::unique_ptr<std::string>& ptr);

// ==============================================================================
// Utility Functions Implementation
// ==============================================================================

std::string HumanReadableSize(uint64_t bytes, int precision)
{
    const char* units[]    = {"B", "KB", "MB", "GB", "TB", "PB"};
    int         unit_index = 0;
    double      size       = static_cast<double>(bytes);

    while (size >= 1024 && unit_index < 5) {
        size /= 1024;
        unit_index++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << size << " " << units[unit_index];
    return oss.str();
}

}  // namespace flashck