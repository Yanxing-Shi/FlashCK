#include "flashck/core/utils/printf.h"

namespace flashck {

/**
 * @brief Overloaded stream insertion operator for 1D vectors.
 *
 * @tparam T Type of elements in the vector (must support stream insertion)
 * @param os Output stream to write to
 * @param vec Vector to be formatted
 * @return Reference to the output stream
 *
 * @note Formats vector as {elem1, elem2, ...}. Handles empty vectors correctly.
 * @example
 * std::vector<int> v{1, 2, 3};
 * std::cout << v; // Outputs: {1, 2, 3}
 *
 * std::vector<std::string> empty;
 * std::cout << empty; // Outputs: {}
 */
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

/**
 * @brief Overloaded stream insertion operator for 2D vectors.
 *
 * @tparam T Type of elements in the inner vectors (must support stream insertion)
 * @param os Output stream to write to
 * @param lod 2D vector to be formatted
 * @return Reference to the output stream
 *
 * @note Formats as {{row1}, {row2}, ...}. Uses 1D vector formatting for rows.
 * @example
 * std::vector<std::vector<int>> m{{1, 2}, {3}, {}};
 * std::cout << m; // Outputs: {{1, 2}, {3}, {}}
 *
 * std::vector<std::vector<char>> empty_2d;
 * std::cout << empty_2d; // Outputs: {}
 */
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