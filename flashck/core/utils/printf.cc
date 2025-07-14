#include "flashck/core/utils/printf.h"

namespace flashck {

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    os << "{";
    const char* sep = "";
    for (const auto& elem : vec) {
        os << sep << elem;
        sep = ", ";
    }
    os << "}";
    return os;
}

template std::ostream& operator<<(std::ostream& os, const std::vector<int64_t>& vec);
template std::ostream& operator<<(std::ostream& os, const std::vector<std::string>& vec);

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& lod)
{
    os << "{";
    const char* outer_sep = "";
    for (const auto& inner : lod) {
        os << outer_sep << inner;  // Reuse 1D vector formatting
        outer_sep = ", ";
    }
    os << "}";
    return os;
}

template std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<int64_t>>& lod);
template std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<std::string>>& lod);

std::string HumanReadableSize(uint64_t bytes, int precision)
{
    const char* units[]   = {"B", "KB", "MB", "GB", "TB"};
    int         unitIndex = 0;
    double      size      = static_cast<double>(bytes);

    while (size >= 1024 && unitIndex < 4) {
        size /= 1024;
        unitIndex++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << size << " " << units[unitIndex];
    return oss.str();
}

}  // namespace flashck