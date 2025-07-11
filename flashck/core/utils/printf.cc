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

}  // namespace flashck