#pragma once

#include <iterator>
#include <ostream>
#include <string>
#include <vector>

#include <fmt/args.h>
#include <fmt/compile.h>  // For compile-time format support
#include <fmt/format.h>

namespace flashck {

/**
 * @brief Formats arguments using type-erased argument store
 * @tparam Args Argument types compatible with format specifiers
 * @param format Compile-time validated format string
 * @param args Arguments to format
 * @return Formatted string with type-safe conversion
 *
 * @example Basic usage:
 * auto s1 = Sprintf("π ≈ {:.2f}", 3.14159);  // Returns "π ≈ 3.14"
 *
 * @example Named arguments with type safety:
 * auto s2 = Sprintf("Name: {name}, Age: {age}",
 *     fmt::arg("name", "Alice"),
 *     fmt::arg("age", 30));  // Returns "Name: Alice, Age: 30"
 *
 * @note Uses type-erased argument storage for runtime formatting
 * while maintaining compile-time format validation
 */
template<typename... Args>
std::string Sprintf(fmt::format_string<Args...> format, const Args&... args)
{
    // Create named argument store
    auto args_store = fmt::make_format_args(args...);

    fmt::memory_buffer buffer;
    fmt::vformat_to(std::back_inserter(buffer), format, args_store);

    return {buffer.data(), buffer.size()};
}

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
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec);

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
std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& lod);

}  // namespace flashck