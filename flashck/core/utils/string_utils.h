#pragma once

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "flashck/core/utils/enforce.h"
#include "flashck/core/utils/errors.h"
#include "flashck/core/utils/printf.h"

namespace flashck {

/*!
 * @brief Check if a string starts with a given prefix
 * @param str The string to check
 * @param prefix The prefix to look for
 * @return true if str starts with prefix, false otherwise
 * @note Uses string_view for zero-copy operations
 */
bool StartsWith(std::string_view str, std::string_view prefix);

/*!
 * @brief Check if a string ends with a given suffix
 * @param str The string to check
 * @param suffix The suffix to look for
 * @return true if str ends with suffix, false otherwise
 * @note Uses string_view for zero-copy operations
 */
bool EndsWith(std::string_view str, std::string_view suffix);

/*!
 * @brief Type trait to detect if a type is streamable to std::ostream
 * @tparam T The type to check
 * @note Used for SFINAE in ToString template functions
 */
template<typename T, typename = void>
struct is_streamable: std::false_type {};

/*!
 * @brief Specialization for streamable types
 * @tparam T The type that can be streamed to std::ostream
 */
template<typename T>
struct is_streamable<T, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<const T&>())>>:
    std::true_type {};

/*!
 * @brief Convert integral types to string (except char)
 * @tparam T Integral type
 * @param value The value to convert
 * @return String representation of the value
 * @note Uses std::to_string for optimal performance
 */
template<typename T>
std::enable_if_t<std::is_integral<T>::value && !std::is_same<T, char>::value, std::string> ToString(T value)
{
    return std::to_string(value);
}

/*!
 * @brief Convert floating point types to string
 * @tparam T Floating point type
 * @param value The value to convert
 * @return String representation of the value
 * @note Uses std::to_string for optimal performance
 */
template<typename T>
std::enable_if_t<std::is_floating_point<T>::value, std::string> ToString(T value)
{
    return std::to_string(value);
}

/*!
 * @brief Convert string_view to string
 * @param value The string_view to convert
 * @return String copy of the string_view
 * @note Efficient conversion from string_view to string
 */
inline std::string ToString(std::string_view value)
{
    return std::string(value);
}

/*!
 * @brief Convert C-style string to string
 * @param value The C-style string to convert
 * @return String representation, empty string if value is null
 * @note Includes null pointer safety check
 */
inline std::string ToString(const char* value)
{
    return value ? std::string(value) : std::string();
}

/*!
 * @brief Convert streamable types to string using operator<<
 * @tparam T Type that supports streaming to std::ostream
 * @param value The value to convert
 * @return String representation of the value
 * @note Uses SFINAE to enable only for streamable non-arithmetic types
 */
template<typename T>
std::enable_if_t<is_streamable<T>::value && !std::is_arithmetic<T>::value, std::string> ToString(const T& value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

/*!
 * @brief Join container elements into a single string with delimiter
 * @tparam Container Any container type with begin()/end() iterators
 * @param container The container to join
 * @param delimiter String to insert between elements (default: empty)
 * @return Joined string
 * @note Optimized to avoid unnecessary operations on empty containers
 */
template<typename Container>
inline std::string JoinStrings(const Container& container, std::string_view delimiter = "")
{
    if (container.empty()) {
        return {};
    }

    std::ostringstream oss;
    auto               it = container.begin();
    oss << *it++;

    for (; it != container.end(); ++it) {
        oss << delimiter << *it;
    }

    return oss.str();
}

/*!
 * @brief Specialization for std::set<std::filesystem::path>
 * @param container Set of filesystem paths
 * @param delimiter String to insert between paths
 * @return Joined string of path strings
 * @note Converts each path to string representation
 */
template<>
inline std::string JoinStrings(const std::set<std::filesystem::path>& container, std::string_view delimiter)
{
    if (container.empty()) {
        return {};
    }

    std::ostringstream oss;
    auto               it = container.begin();
    oss << it->string();
    ++it;

    for (; it != container.end(); ++it) {
        oss << delimiter << it->string();
    }

    return oss.str();
}

/*!
 * @brief Split a string into tokens using a delimiter
 * @param str The string to split
 * @param delimiter The delimiter to use for splitting
 * @return Vector of string tokens (empty tokens are filtered out)
 * @note Optimized to avoid unnecessary copies using string_view
 */
std::vector<std::string> SplitStrings(std::string_view str, std::string_view delimiter);

/*!
 * @brief Extract substring before or after a separator
 * @param str The input string
 * @param separator The separator to search for
 * @param direction "left" for substring before separator, "right" for after
 * @return Substring based on direction, empty string if separator not found
 * @throws std::invalid_argument if direction is invalid or separator is empty
 */
std::string
SliceAroundSubstring(std::string_view str, std::string_view separator, std::string_view direction = "right");

/*!
 * @brief Generate hexadecimal hash string from input
 * @param input_str The input string to hash
 * @return Hexadecimal representation of the hash
 * @note Uses std::hash for consistent hashing
 */
std::string HashToHexString(std::string_view input_str);

/*!
 * @brief Generate combined hexadecimal hash string from input
 * @param input_str The input string to hash
 * @return Combined hexadecimal representation of two different hashes
 * @note Uses both string and string_view hashing for added uniqueness
 */
std::string CombinedHashToHexString(std::string_view input_str);

/*!
 * @brief Replace all occurrences of a substring with another string
 * @param s The string to modify (modified in-place)
 * @param search The substring to search for
 * @param replacement The string to replace with
 * @note Optimized to prevent infinite loops when replacement contains search
 */
void ReplaceAll(std::string& s, std::string_view search, std::string_view replacement);

/*!
 * @brief Extract workload parameters from a key string
 * @param key The key string containing expressions like "M == 2" or "N >= 1 && N <= 10"
 * @return Map of parameter names to their values
 * @note For ranges, returns the minimum value as representative
 */
std::map<std::string, int> ExtractWorkLoad(std::string_view key);

/*!
 * @brief Generate workload string from parameter mappings
 * @param name_value_mapping Map of parameter names to value vectors
 * @return Generated workload string with appropriate expressions
 * @note Handles single values, ranges, and empty values appropriately
 */
std::string GenWorkLoad(const std::map<std::string, std::vector<int64_t>>& name_value_mapping);

/*!
 * @brief Template class for grouping consecutive elements by a key function
 * @tparam InputIterator Iterator type (must be at least forward iterator)
 * @tparam KeyFunc Function type that extracts key from iterator element
 * @note Provides lazy evaluation and efficient grouping of consecutive elements
 */
template<typename InputIterator, typename KeyFunc>
class GroupBy {
public:
    //! Type of the key returned by the key function
    using KeyType = decltype(std::declval<KeyFunc>()(*std::declval<InputIterator>()));

    //! Ensure we have at least forward iterators for proper functionality
    static_assert(std::is_convertible_v<typename std::iterator_traits<InputIterator>::iterator_category,
                                        std::forward_iterator_tag>,
                  "Requires at least forward iterators");

    /*!
     * @brief Represents a group of consecutive elements with the same key
     */
    class Group {
    public:
        /*!
         * @brief Construct a group with key and iterator range
         * @param key The common key for all elements in this group
         * @param begin Iterator to the first element in the group
         * @param end Iterator to one past the last element in the group
         */
        Group(KeyType key, InputIterator begin, InputIterator end): key_(key), begin_(begin), end_(end) {}

        /*!
         * @brief Get the key for this group
         * @return Reference to the key
         */
        const KeyType& key() const
        {
            return key_;
        }

        /*!
         * @brief Get iterator to the first element in the group
         * @return Iterator to the beginning of the group
         */
        InputIterator begin() const
        {
            return begin_;
        }

        /*!
         * @brief Get iterator to one past the last element in the group
         * @return Iterator to the end of the group
         */
        InputIterator end() const
        {
            return end_;
        }

    private:
        const KeyType       key_;    //!< The common key for this group
        const InputIterator begin_;  //!< Iterator to first element
        const InputIterator end_;    //!< Iterator to one past last element
    };

    /*!
     * @brief Iterator over groups of consecutive elements with the same key
     * @note Provides input iterator semantics for lazy group evaluation
     */
    class Iterator {
    public:
        //! Iterator traits
        using iterator_category = std::input_iterator_tag;
        using value_type        = Group;
        using difference_type   = std::ptrdiff_t;
        using pointer           = value_type*;
        using reference         = value_type&;

        /*!
         * @brief Construct iterator starting at current position
         * @param current Current position in the sequence
         * @param end End of the sequence
         * @param key_func Function to extract key from elements
         */
        Iterator(InputIterator current, InputIterator end, KeyFunc key_func):
            current_(current), sequence_end_(end), key_func_(key_func)
        {
            AdvanceToNextGroup();
        }

        /*!
         * @brief Check if two iterators are equal
         * @param other The other iterator to compare with
         * @return true if iterators point to the same position
         */
        bool operator==(const Iterator& other) const
        {
            return current_ == other.current_;
        }

        /*!
         * @brief Check if two iterators are not equal
         * @param other The other iterator to compare with
         * @return true if iterators point to different positions
         */
        bool operator!=(const Iterator& other) const
        {
            return !(*this == other);
        }

        /*!
         * @brief Advance to the next group
         * @return Reference to this iterator
         */
        Iterator& operator++()
        {
            current_ = group_end_;
            AdvanceToNextGroup();
            return *this;
        }

        /*!
         * @brief Dereference iterator to get current group
         * @return Current group object
         */
        value_type operator*() const
        {
            return Group(current_key_, current_, group_end_);
        }

    private:
        /*!
         * @brief Advance to the next group of consecutive elements with same key
         * @note Updates current_key_ and group_end_ for the next group
         */
        void AdvanceToNextGroup()
        {
            if (current_ == sequence_end_)
                return;

            group_end_   = current_;
            current_key_ = key_func_(*group_end_);

            // Find the end of current group (consecutive elements with same key)
            while (group_end_ != sequence_end_ && key_func_(*group_end_) == current_key_) {
                ++group_end_;
            }
        }

        InputIterator       current_;       //!< Current position in sequence
        InputIterator       group_end_;     //!< End of current group
        const InputIterator sequence_end_;  //!< End of entire sequence
        KeyFunc             key_func_;      //!< Function to extract key from elements
        KeyType             current_key_;   //!< Key of current group
    };

    /*!
     * @brief Construct GroupBy object with iterator range and key function
     * @param begin Iterator to the beginning of the sequence
     * @param end Iterator to the end of the sequence
     * @param key_func Function to extract key from elements
     */
    explicit GroupBy(InputIterator begin, InputIterator end, KeyFunc key_func):
        begin_(begin), end_(end), key_func_(key_func)
    {
    }

    /*!
     * @brief Get iterator to the first group
     * @return Iterator to the beginning of groups
     */
    Iterator begin()
    {
        return Iterator(begin_, end_, key_func_);
    }

    /*!
     * @brief Get iterator to one past the last group
     * @return Iterator to the end of groups
     */
    Iterator end()
    {
        return Iterator(end_, end_, key_func_);
    }

private:
    const InputIterator begin_;     //!< Beginning of the sequence
    const InputIterator end_;       //!< End of the sequence
    KeyFunc             key_func_;  //!< Function to extract key from elements
};

/*!
 * @brief Convenience function to create GroupBy objects with template argument deduction
 * @tparam Iterator Iterator type
 * @tparam KeyFunc Key function type
 * @param begin Iterator to the beginning of the sequence
 * @param end Iterator to the end of the sequence
 * @param func Function to extract key from elements
 * @return GroupBy object configured with the given parameters
 */
template<typename Iterator, typename KeyFunc>
auto GroupByFunc(Iterator begin, Iterator end, KeyFunc func)
{
    return GroupBy<Iterator, KeyFunc>(begin, end, func);
}

}  // namespace flashck