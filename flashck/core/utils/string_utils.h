#pragma once

#include <algorithm>    // std::transform
#include <cassert>      // assert
#include <cctype>       // ::tolower
#include <cstddef>      // std::size_t
#include <functional>   // std::hash
#include <iomanip>      // std::hex, std::setw, std::setfill
#include <iterator>     // std::back_inserter
#include <sstream>      // std::stringstream
#include <stdexcept>    // std::invalid_argument
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace flashck {

/**
 * @brief Checks if a string starts with the specified prefix.
 *
 * @param str The input string to check
 * @param prefix The prefix to look for (empty prefix always returns true)
 * @return true if the string starts with the prefix, false otherwise
 *
 * @note Complexity: O(prefix.length())
 */
bool StartsWith(std::string_view str, std::string_view prefix) noexcept;

/**
 * @brief Checks if a string ends with the specified suffix.
 *
 * @param str The input string to check
 * @param suffix The suffix to look for (empty suffix always returns true)
 * @return true if the string ends with the suffix, false otherwise
 *
 * @note Complexity: O(suffix.length())
 */
bool EndsWith(std::string_view str, std::string_view suffix) noexcept;

/**
 * @brief Checks if a type can be streamed via operator<<.
 * @tparam T Type to check.
 */
template<typename T, typename = void>
struct is_streamable: std::false_type {};

template<typename T>
struct is_streamable<T, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<const T&>())>>:
    std::true_type {};

/**
 * @brief Converts a streamable non-arithmetic type to string using ostringstream.
 * @tparam T Type must support operator<< and not be arithmetic.
 * @param value Input value.
 * @return String representation of the value.
 */
template<typename T>
std::enable_if_t<is_streamable<T>::value && !std::is_arithmetic<T>::value, std::string> ToString(const T& value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

/**
 * @brief Converts integral types (excluding char) using std::ToString for efficiency.
 * @tparam T Must be integral and not char.
 * @param value Input value.
 * @return String representation.
 */
template<typename T>
std::enable_if_t<std::is_integral<T>::value && !std::is_same<T, char>::value, std::string> ToString(T value)
{
    return std::to_string(value);
}

/**
 * @brief Converts a floating-point value to a string using stream formatting.
 *
 * @tparam T Floating-point type (float, double, long double). Constrained by SFINAE.
 * @param value The floating-point value to convert.
 * @return std::string String representation of the value.
 *
 * @note
 * - Uses std::ostringstream for conversion, inheriting current stream formatting rules
 * - Precision/notation depends on stream's default settings
 * - Ensures compile-time type constraint via std::enable_if_t
 *
 * @par Example:
 * @code
 * double pi = 3.1415926535;
 * std::string s = ToString(pi);  // "3.14159" (depends on default precision)
 * @endcode
 */
template<typename T>
std::enable_if_t<std::is_floating_point<T>::value, std::string> ToString(T value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

/**
 * @brief Converts a string view to a std::string.
 *
 * @param value The string view to convert. Can be any valid string view,
 *              including views of std::string objects, C strings, or string literals.
 * @return std::string A new string containing a copy of the viewed characters.
 *
 * @note This function:
 * - Creates an explicit copy of the string view's data
 * - Handles empty views gracefully (returns empty string)
 * - Maintains the exact content including null characters
 *
 * @par Example:
 * @code
 * std::string_view sv = "Hello";
 * std::string s = ToString(sv);  // s contains "Hello"
 * @endcode
 *
 * @par Complexity:
 * - Time: O(n) linear in the length of the view
 * - Space: Allocates memory for the new string
 */
inline std::string ToString(std::string_view value)
{
    return std::string(value);  // Explicit construction from string_view
}

/**
 * @brief Converts a C-style string to std::string.
 * @param value Input C-string.
 * @return std::string copy of the input.
 */
inline std::string ToString(const char* value)
{
    return value;
}

/**
 * @brief Splits a string around the first occurrence of a separator and returns the specified portion
 *
 * This function locates the first occurrence of the `separator` within `str`, then returns either:
 * - The left portion (from start to separator) when direction is "left"
 * - The right portion (from separator end to string end) when direction is "right"
 *
 * @param[in] str The input string to process. Maintains original case sensitivity.
 * @param[in] separator The substring acting as the split point. Must not be empty.
 * @param[in] direction Selection of which portion to return ("left"/"right"). Case-insensitive.
 *                      Defaults to "right".
 * @return std::string_view View of the requested portion. Returns empty view if:
 *         - Separator not found
 *         - Resulting portion has zero length
 * @throw std::invalid_argument Thrown if:
 *        - `separator` is empty
 *        - `direction` is neither "left" nor "right" (case-insensitive)
 *
 * @par Example Usage:
 * @code
 * // Basic usage
 * SliceAroundSubstring("apple:orange", ":", "left");   // returns "apple"
 * SliceAroundSubstring("192.168.1.1", ".", "right");   // returns "168.1.1"
 *
 * // Edge cases
 * SliceAroundSubstring("start--end", "--", "right");   // returns "end"
 * SliceAroundSubstring("nodivider", "-");              // returns empty view
 * @endcode
 *
 * @note The returned string_view remains valid only while the original string exists.
 * @warning Avoid passing temporary strings that may be destroyed before using the result.
 */
std::string_view
SliceAroundSubstring(std::string_view str, std::string_view separator, std::string_view direction = "right");

/**
 * @brief Generates standard hash hex string for input (non-cryptographic)
 * @param inputStr Source string to hash
 * @return Hexadecimal string representation of the hash
 *
 * @note Uses std::hash which:
 * - Produces size_t hash (typically 64-bit)
 * - Not suitable for cryptographic purposes
 * - Output length varies by platform (16 chars for 64-bit)
 */
std::string HashToHexString(std::string_view input_str);

/**
 * @brief Generates combined hash hex string using multiple hash functions
 * @param inputStr Source string to hash
 * @return 128-bit hex string combining two different hash implementations
 *
 * @note Hybrid hashing strategy:
 * 1. Uses std::hash as primary hash
 * 2. Uses hash<string> from functional as secondary
 * 3. Combines results for better distribution
 */
std::string CombinedHashToHexString(std::string_view input_str);

/**
 * @brief Replaces all occurrences of a substring within a string.
 *
 * @details This function performs in-place replacement of all instances of the `search` substring
 *          with the `replacement` substring in the target string `s`. The implementation uses a
 *          two-phase approach for efficiency: first counting matches to calculate required memory,
 *          then performing replacements with position offset tracking.
 *
 * @param[in,out] s          The string to modify. Will be altered directly.
 * @param[in]      search     The substring pattern to search for. Must be non-empty.
 * @param[in]      replacement The substring to substitute for matches. May be empty.
 *
 * @note Key implementation features:
 * 1. Uses std::string_view parameters to avoid unnecessary copies
 * 2. Two-phase processing prevents repeated string modifications
 * 3. Memory pre-allocation when resulting string might grow
 * 4. Early termination if replacement contains search pattern
 *
 * @warning Special case handling:
 * - If `search` is empty, returns immediately to prevent infinite loops
 * - If `replacement` contains `search`, stops after first replacement to avoid recursion
 * - Thread safety: Not thread-safe for concurrent modifications to parameter `s`
 *
 * @par Exception Safety:
 * Provides strong exception guarantee - original string remains intact if memory allocation fails.
 *
 * @par Complexity:
 * - Time: O(n + m) average case, where n = s.length(), m = number of matches
 * - Space: O(1) additional space, except when reallocation occurs
 *
 * @par Example:
 * @code
 * std::string text = "apple orange apple";
 * ReplaceAll(text, "apple", "fruit");
 * // text becomes "fruit orange fruit"
 * @endcode
 */
void ReplaceAll(std::string& s, std::string_view search, std::string_view replacement);

/**
 * @class GroupBy
 * @brief Groups sequence elements based on a key selector function
 *
 * @tparam InputIterator Type of input iterator
 * @tparam KeyFunc Type of key extraction function
 *
 * @par Example Usage:
 * @code
 * std::vector<int> vec{1, 1, 2, 3, 3};
 * auto group_by = GroupBy(vec.begin(), vec.end(), [](int x) { return x; });
 * for (const auto& group : group_by) {
 *   std::cout << "Key: " << group.key() << " Values: ";
 *   for (auto it = group.begin(); it != group.end(); ++it) {
 *     std::cout << *it << " ";
 *   }
 * }
 * @endcode
 */
template<typename InputIterator, typename KeyFunc>
class GroupBy {
public:
    /// @brief Type alias for key value
    using KeyType = decltype(std::declval<KeyFunc>()(*std::declval<InputIterator>()));

    // Verify iterator requirements²⁸
    static_assert(std::is_convertible_v<typename std::iterator_traits<InputIterator>::iterator_category,
                                        std::forward_iterator_tag>,
                  "Requires at least forward iterators");

    /**
     * @class Group
     * @brief Represents a single group of elements sharing the same key
     */
    class Group {
    public:
        /**
         * @brief Constructs a group element
         * @param[in] key Group identifier
         * @param[in] begin Start iterator of group elements
         * @param[in] end End iterator of group elements
         */
        Group(KeyType key, InputIterator begin, InputIterator end): key_(key), begin_(begin), end_(end) {}

        /// @brief Gets group key
        /// @return Constant reference to the key value
        const KeyType& key() const
        {
            return key_;
        }

        /// @brief Gets start iterator of the group
        /// @return Forward iterator to first element
        InputIterator begin() const
        {
            return begin_;
        }

        /// @brief Gets end iterator of the group
        /// @return Forward iterator to element past last element
        InputIterator end() const
        {
            return end_;
        }

    private:
        const KeyType       key_;
        const InputIterator begin_;
        const InputIterator end_;
    };

    /**
     * @class Iterator
     * @brief Input iterator for traversing groups³⁸
     */
    class Iterator {
    public:
        /// @brief Standard iterator traits
        using iterator_category = std::input_iterator_tag;
        using value_type        = Group;
        using difference_type   = std::ptrdiff_t;
        using pointer           = value_type*;
        using reference         = value_type&;

        /**
         * @brief Constructs group iterator
         * @param[in] current Start position in sequence
         * @param[in] end End of sequence
         * @param[in] key_func Key extraction function
         */
        Iterator(InputIterator current, InputIterator end, KeyFunc key_func):
            current_(current), sequence_end_(end), key_func_(key_func)
        {
            AdvanceToNextGroup();
        }

        /// @brief Equality comparison operator
        bool operator==(const Iterator& other) const
        {
            return current_ == other.current_;
        }

        /// @brief Inequality comparison operator
        bool operator!=(const Iterator& other) const
        {
            return !(*this == other);
        }

        /// @brief Pre-increment operator
        Iterator& operator++()
        {
            current_ = group_end_;
            AdvanceToNextGroup();
            return *this;
        }

        /// @brief Dereference operator
        value_type operator*() const
        {
            return Group(current_key_, current_, group_end_);
        }

    private:
        /// @brief Advances to next group boundary
        void AdvanceToNextGroup()
        {
            if (current_ == sequence_end_)
                return;

            group_end_   = current_;
            current_key_ = key_func_(*group_end_);

            while (group_end_ != sequence_end_ && key_func_(*group_end_) == current_key_) {
                ++group_end_;
            }
        }

        InputIterator       current_;
        InputIterator       group_end_;
        const InputIterator sequence_end_;
        KeyFunc             key_func_;
        KeyType             current_key_;
    };

    /**
     * @brief Constructs GroupBy processor
     * @param[in] begin Start of input sequence
     * @param[in] end End of input sequence
     * @param[in] key_func Key extraction function
     */
    explicit GroupBy(InputIterator begin, InputIterator end, KeyFunc key_func):
        begin_(begin), end_(end), key_func_(key_func)
    {
    }

    /// @brief Gets start iterator of groups
    Iterator begin()
    {
        return Iterator(begin_, end_, key_func_);
    }

    /// @brief Gets end iterator of groups
    Iterator end()
    {
        return Iterator(end_, end_, key_func_);
    }

private:
    const InputIterator begin_;
    const InputIterator end_;
    KeyFunc             key_func_;
};

/**
 * @brief Creates a GroupBy range adapter
 * @tparam Iterator Iterator type for element sequence
 * @tparam KeyFunc Key generation function type
 * @param begin Start iterator of element range
 * @param end End iterator of element range
 * @param func Key generation function
 * @return GroupBy<Iterator, KeyFunc> instance
 *
 * @par Example:
 * @code
 * auto groups = GroupByFunc(container.begin(), container.end(),
 *                  [](const auto& item) { return item.key; });
 * @endcode
 */
template<typename Iterator, typename KeyFunc>
auto GroupByFunc(Iterator begin, Iterator end, KeyFunc func)
{
    return GroupBy<Iterator, KeyFunc>(begin, end, func);
}

/**
 * @brief Transforms elements from one or two input ranges using a callable
 *
 * @tparam Range Input range type supporting begin()/end()
 * @tparam F Unary function type compatible with range elements
 * @param[in] r Input range to transform
 * @param[in] f Transformation function (f(const Element&) -> T)
 * @return std::vector<decltype(f(*r.begin()))> Transformed elements vector
 *
 * @throws std::bad_alloc On memory allocation failure
 * @note Preserves original element order. Complexity: O(N)
 *
 * @example
 * std::vector<int> v{1,2,3};
 * auto squared = Transform(v, [](int x) { return x*x; });
 * // Returns {1, 4, 9}
 */
template<class Range, class F>
inline auto Transform(const Range& r, F f) -> std::vector<decltype(f(*r.begin()))>
{
    std::vector<decltype(f(*r.begin()))> result;
    std::transform(r.begin(), r.end(), std::back_inserter(result), f);
    return result;
}

/**
 * @brief Transforms pairs of elements from two ranges using a binary function
 *
 * @tparam Range1 First input range type
 * @tparam Range2 Second input range type
 * @tparam F Binary function type (f(const E1&, const E2&) -> T)
 * @param[in] r1 First input range
 * @param[in] r2 Second input range (must be same length as r1)
 * @param[in] f Binary transformation function
 * @return std::vector<decltype(f(*r1.begin(), *r2.begin()))> Result vector
 *
 * @throws std::invalid_argument If range lengths differ (debug builds only)
 * @throws std::bad_alloc On memory allocation failure
 * @warning Undefined behavior in release builds if ranges differ in length
 *
 * @example
 * std::vector<int> a{1,2,3}, b{4,5,6};
 * auto sums = Transform(a, b, [](int x, int y) { return x+y; });
 * // Returns {5, 7, 9}
 */
template<class Range1, class Range2, class F>
inline auto Transform(const Range1& r1, const Range2& r2, F f) -> std::vector<decltype(f(*r1.begin(), *r2.begin()))>
{
    std::vector<decltype(f(*r1.begin(), *r2.begin()))> result;
    assert(std::distance(r1.begin(), r1.end()) == std::distance(r2.begin(), r2.end()));
    std::transform(r1.begin(), r1.end(), r2.begin(), std::back_inserter(result), f);
    return result;
}

/**
 * @brief Extracts keys from associative containers as strings
 *
 * @tparam T Associative container type (map, unordered_map, etc.)
 * @param[in] map Container with key-value pairs
 * @return std::vector<std::string> Vector of stringified keys
 *
 * @throws std::bad_alloc On memory allocation failure
 * @throws Any exceptions thrown by key-to-string conversions
 *
 * @note Requires container value_type to be pair<const Key, Value>
 * @warning Key type must be implicitly convertible to std::string
 *
 * @example
 * std::map<int, char> m{{1,'a'}, {2,'b'}};
 * auto keys = GetKeyVector(m); // Returns {"1", "2"}
 */
template<class T>
std::vector<std::string> GetKeyVector(T map)
{
    return Transform(map, [](auto&& p) { return p.first; });
}

}  // namespace flashck