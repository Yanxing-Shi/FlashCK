#pragma once

#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>

namespace flashck {

// ==============================================================================
// String Formatting Functions
// ==============================================================================

/**
 * @brief Format string and append to existing string
 * @param s Target string to append to
 * @param format Format string
 * @param args Arguments to format
 */
template<typename... Args>
inline void Fprintf(std::string& s, fmt::format_string<Args...> format, Args&&... args)
{
    fmt::vformat_to(std::back_inserter(s), format, fmt::make_format_args(args...));
}

/**
 * @brief Format string and return result
 * @param format Format string
 * @param args Arguments to format
 * @return Formatted string
 */
template<typename... Args>
inline std::string Sprintf(fmt::format_string<Args...> format, Args&&... args)
{
    return fmt::vformat(format, fmt::make_format_args(args...));
}

/**
 * @brief Return empty string (overload for no arguments)
 * @return Empty string
 */
inline std::string Sprintf()
{
    return "";
}

/**
 * @brief Print formatted string to stdout
 * @param format Format string
 * @param args Arguments to format
 */
template<typename... Args>
inline void Printf(fmt::format_string<Args...> format, Args&&... args)
{
    fmt::vprint(format, fmt::make_format_args(args...));
}

/**
 * @brief Print formatted string to stderr
 * @param format Format string
 * @param args Arguments to format
 */
template<typename... Args>
inline void Eprintf(fmt::format_string<Args...> format, Args&&... args)
{
    fmt::vprint(stderr, format, fmt::make_format_args(args...));
}

// ==============================================================================
// Container Stream Operators
// ==============================================================================

/**
 * @brief Stream operator for std::vector
 * @param os Output stream
 * @param vec Vector to print
 * @return Output stream reference
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec);

/**
 * @brief Stream operator for nested std::vector
 * @param os Output stream
 * @param lod Vector of vectors to print
 * @return Output stream reference
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& lod);

/**
 * @brief Stream operator for std::map
 * @param os Output stream
 * @param map Map to print
 * @return Output stream reference
 */
template<typename K, typename V>
std::ostream& operator<<(std::ostream& os, const std::map<K, V>& map);

/**
 * @brief Stream operator for std::unordered_map
 * @param os Output stream
 * @param map Unordered map to print
 * @return Output stream reference
 */
template<typename K, typename V>
std::ostream& operator<<(std::ostream& os, const std::unordered_map<K, V>& map);

/**
 * @brief Stream operator for std::set
 * @param os Output stream
 * @param set Set to print
 * @return Output stream reference
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::set<T>& set);

/**
 * @brief Stream operator for std::unordered_set
 * @param os Output stream
 * @param set Unordered set to print
 * @return Output stream reference
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::unordered_set<T>& set);

/**
 * @brief Stream operator for std::pair
 * @param os Output stream
 * @param pair Pair to print
 * @return Output stream reference
 */
template<typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, const std::pair<T1, T2>& pair);

/**
 * @brief Stream operator for std::optional
 * @param os Output stream
 * @param opt Optional to print
 * @return Output stream reference
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::optional<T>& opt);

/**
 * @brief Stream operator for std::shared_ptr
 * @param os Output stream
 * @param ptr Shared pointer to print
 * @return Output stream reference
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::shared_ptr<T>& ptr);

/**
 * @brief Stream operator for std::unique_ptr
 * @param os Output stream
 * @param ptr Unique pointer to print
 * @return Output stream reference
 */
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::unique_ptr<T>& ptr);

// ==============================================================================
// Utility Functions
// ==============================================================================

/**
 * @brief Convert bytes to human-readable size string
 * @param bytes Size in bytes
 * @param precision Number of decimal places
 * @return Human-readable size string
 */
std::string HumanReadableSize(uint64_t bytes, int precision = 2);

}  // namespace flashck