#include "flashck/core/utils/string_utils.h"

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
bool StartsWith(std::string_view str, std::string_view prefix) noexcept
{
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

/**
 * @brief Checks if a string ends with the specified suffix.
 *
 * @param str The input string to check
 * @param suffix The suffix to look for (empty suffix always returns true)
 * @return true if the string ends with the suffix, false otherwise
 *
 * @note Complexity: O(suffix.length())
 */
bool EndsWith(std::string_view str, std::string_view suffix) noexcept
{
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
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
std::string_view SliceAroundSubstring(std::string_view str, std::string_view separator, std::string_view direction)
{
    // Validate separator
    if (separator.empty()) {
        throw std::invalid_argument("Separator cannot be empty");
    }

    // Normalize direction to lowercase
    std::string dir_lower;
    dir_lower.reserve(direction.size());
    std::transform(direction.begin(), direction.end(), std::back_inserter(dir_lower), ::tolower);

    // Find separator position
    const size_t pos = str.find(separator);
    if (pos == std::string_view::npos) {
        return {};
    }

    // Calculate substring based on direction
    if (dir_lower == "left") {
        return str.substr(0, pos);
    }
    else if (dir_lower == "right") {
        const size_t start = pos + separator.length();
        return start <= str.length() ? str.substr(start) : std::string_view{};
    }

    // Handle invalid direction
    throw std::invalid_argument("Direction must be 'left' or 'right'");
}

/**
 * @brief Generates standard hash hex string for input (non-cryptographic)
 * @param input_str Source string to hash
 * @return Hexadecimal string representation of the hash
 *
 * @note Uses std::hash which:
 * - Produces size_t hash (typically 64-bit)
 * - Not suitable for cryptographic purposes
 * - Output length varies by platform (16 chars for 64-bit)
 */

std::string HashToHexString(std::string_view input_str)
{
    const std::size_t hash_value = std::hash<std::string>{}(std::string(input_str));

    std::stringstream ss;
    ss << std::hex << std::setw(sizeof(std::size_t) * 2) << std::setfill('0') << hash_value;

    return ss.str();
}

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
std::string CombinedHashToHexString(std::string_view input_str)
{
    const std::string str_copy(input_str);
    const auto        hash1 = std::hash<std::string>{}(str_copy);
    const auto        hash2 = std::_Hash_impl::hash(input_str.data(), input_str.size());

    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(sizeof(std::size_t) * 2) << hash1
       << std::setw(sizeof(std::size_t) * 2) << hash2;

    return ss.str();

    return ss.str();
}

/**
 * @brief Replaces all occurrences of a search substring with a replacement substring within a string.
 *
 * @param[in,out] s The target string to modify. Content will be changed in-place.
 * @param[in] search The substring pattern to search for. Must not be empty.
 * @param[in] replacement The substring to substitute for found patterns. May be empty.
 *
 * @note Implementation details:
 * - Two-phase operation: Count matches → Perform replacements
 * - Memory pre-allocation when resultant string needs expansion
 * - Offset tracking for consistent position calculation
 * - Early termination when replacement contains search pattern
 *
 * @warning Special case handling:
 * - Returns immediately if search is empty to prevent infinite loops
 * - Breaks replacement loop if replacement contains search pattern
 * - Not thread-safe for concurrent modifications to target string
 *
 * @par Algorithm Complexity:
 * - Time: O(n + m*k) where:
 *   - n = s.length()
 *   - m = number of matches
 *   - k = replacement operation cost
 * - Space: O(1) except during reallocation
 *
 * @par Exception Safety:
 * Provides strong guarantee - original string preserved if memory allocation fails.
 *
 * @par Example:
 * @code
 * std::string text = "Hello world! Hello universe!";
 * ReplaceAll(text, "Hello", "Hi");
 * // text becomes "Hi world! Hi universe!"
 * @endcode
 */
void ReplaceAll(std::string& s, std::string_view search, std::string_view replacement)
{
    if (search.empty())
        return;  // Prevent infinite loop for empty search pattern

    size_t       count       = 0;
    const size_t search_len  = search.length();
    const size_t replace_len = replacement.length();

    // Phase 1: Count matches and calculate new length
    for (size_t pos = s.find(search); pos != std::string::npos; pos = s.find(search, pos + search_len)) {
        ++count;
    }
    if (count == 0)
        return;

    // Pre-allocate memory if resultant string needs expansion
    const size_t new_len = s.length() + count * (replace_len - search_len);
    if (replace_len > search_len) {
        s.reserve(new_len + 1);  // +1 accounts for potential implementation-specific alignment
    }

    // Phase 2: Perform replacements with offset tracking
    size_t offset = 0;
    for (size_t pos = s.find(search); pos != std::string::npos; pos = s.find(search, pos + search_len)) {
        s.replace(pos + offset, search_len, replacement);
        offset += replace_len - search_len;

        // Early termination if replacement contains search pattern
        if (replacement.find(search) != std::string_view::npos) {
            break;  // Prevent infinite replacement recursion
        }
    }
}

}  // namespace flashck