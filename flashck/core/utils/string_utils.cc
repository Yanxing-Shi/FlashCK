#include "flashck/core/utils/string_utils.h"

namespace flashck {

bool StartsWith(const std::string& str, const std::string& prefix)
{
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

bool EndsWith(const std::string& str, const std::string& suffix)
{
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> SplitStrings(const std::string& str, const std::string& delimiter)
{
    std::vector<std::string> tokens;
    size_t                   start = 0;
    size_t                   end   = str.find(delimiter);

    while (end != std::string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end   = str.find(delimiter, start);
    }
    tokens.push_back(str.substr(start));
    return tokens;
}

std::string SliceAroundSubstring(const std::string& str, const std::string& separator, const std::string& direction)
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
    if (pos == std::string::npos) {
        return {};
    }

    // Calculate substring based on direction
    if (dir_lower == "left") {
        return str.substr(0, pos);
    }
    else if (dir_lower == "right") {
        const size_t start = pos + separator.length();
        return start <= str.length() ? str.substr(start) : std::string{};
    }

    // Handle invalid direction
    throw std::invalid_argument("Direction must be 'left' or 'right'");
}

std::string HashToHexString(const std::string& input_str)
{
    const std::size_t hash_value = std::hash<std::string>{}(input_str);

    std::stringstream ss;
    ss << std::hex << std::setw(sizeof(std::size_t) * 2) << std::setfill('0') << hash_value;

    return ss.str();
}

std::string CombinedHashToHexString(const std::string& input_str)
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

void ReplaceAll(std::string& s, const std::string& search, const std::string& replacement)
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
        if (replacement.find(search) != std::string::npos) {
            break;  // Prevent infinite replacement recursion
        }
    }
}

}  // namespace flashck