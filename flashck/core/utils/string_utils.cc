#include "flashck/core/utils/string_utils.h"

namespace flashck {

bool StartsWith(std::string_view str, std::string_view prefix)
{
    // Optimized: Use string_view::substr for zero-copy operation
    return str.size() >= prefix.size() && str.substr(0, prefix.size()) == prefix;
}

bool EndsWith(std::string_view str, std::string_view suffix)
{
    // Optimized: Use string_view::substr for zero-copy operation
    return str.size() >= suffix.size() && str.substr(str.size() - suffix.size()) == suffix;
}

std::vector<std::string> SplitStrings(std::string_view str, std::string_view delimiter)
{
    // Handle edge cases early for better performance
    if (str.empty() || delimiter.empty()) {
        return {std::string(str)};
    }

    std::vector<std::string> tokens;
    size_t                   start = 0;
    size_t                   end   = str.find(delimiter);

    // Process each token found by delimiter
    while (end != std::string_view::npos) {
        if (end > start) {  // Only add non-empty tokens to filter out empty strings
            tokens.emplace_back(str.substr(start, end - start));
        }
        start = end + delimiter.length();
        end   = str.find(delimiter, start);
    }

    // Add the last token if it's not empty
    if (start < str.length()) {
        tokens.emplace_back(str.substr(start));
    }

    return tokens;
}

std::string SliceAroundSubstring(std::string_view str, std::string_view separator, std::string_view direction)
{
    // Validate separator early to avoid unnecessary work
    if (separator.empty()) {
        throw std::invalid_argument("Separator cannot be empty");
    }

    // Find separator position
    const size_t pos = str.find(separator);
    if (pos == std::string_view::npos) {
        return {};  // Separator not found, return empty string
    }

    // Extract substring based on direction (optimized: no case conversion)
    if (direction == "left") {
        return std::string(str.substr(0, pos));
    }
    else if (direction == "right") {
        const size_t start = pos + separator.length();
        return start <= str.length() ? std::string(str.substr(start)) : std::string{};
    }

    // Handle invalid direction
    throw std::invalid_argument("Direction must be 'left' or 'right'");
}

std::string HashToHexString(std::string_view input_str)
{
    // Use string_view hash for better performance (no string copy)
    const std::size_t hash_value = std::hash<std::string_view>{}(input_str);

    // Convert to hex string with proper formatting
    std::ostringstream ss;
    ss << std::hex << std::setw(sizeof(std::size_t) * 2) << std::setfill('0') << hash_value;

    return ss.str();
}

std::string CombinedHashToHexString(std::string_view input_str)
{
    // Use two different hash functions for better uniqueness
    const std::string str_copy(input_str);
    const auto        hash1 = std::hash<std::string>{}(str_copy);        // String hash
    const auto        hash2 = std::hash<std::string_view>{}(input_str);  // String_view hash

    // Combine both hashes into a single hex string
    std::ostringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(sizeof(std::size_t) * 2) << hash1
       << std::setw(sizeof(std::size_t) * 2) << hash2;

    return ss.str();
}

void ReplaceAll(std::string& s, std::string_view search, std::string_view replacement)
{
    if (search.empty()) {
        return;  // Prevent infinite loop for empty search pattern
    }

    size_t pos = 0;
    // Optimized: Single-pass replacement without pre-counting
    while ((pos = s.find(search, pos)) != std::string::npos) {
        s.replace(pos, search.length(), replacement);
        pos += replacement.length();

        // Prevent infinite replacement if replacement contains search pattern
        if (replacement.find(search) != std::string_view::npos) {
            break;
        }
    }
}

std::map<std::string, int> ExtractWorkLoad(std::string_view key)
{
    std::map<std::string, int> result;
    const std::string          key_str(key);  // Convert to string for regex compatibility

    // Handle equality expressions like "M == 2"
    // Use static regex for better performance (compiled once)
    static const std::regex eq_pattern(R"((\w+)\s*==\s*(\d+))");
    std::smatch             match;
    std::string             remaining = key_str;

    while (std::regex_search(remaining, match, eq_pattern)) {
        result[match[1].str()] = std::stoi(match[2].str());
        remaining              = match.suffix().str();
    }

    // Handle range expressions like "M>=2 && M<=9"
    // Use static regex for better performance (compiled once)
    static const std::regex range_pattern(R"((\w+)\s*>=\s*(\d+)\s*&&\s*\1\s*<=\s*(\d+))");
    remaining = key_str;

    while (std::regex_search(remaining, match, range_pattern)) {
        // For ranges, we'll take the minimum value as the representative
        result[match[1].str()] = std::stoi(match[2].str());
        remaining              = match.suffix().str();
    }

    return result;
}

std::string GenWorkLoad(const std::map<std::string, std::vector<int64_t>>& name_value_mapping)
{
    std::vector<std::string> key_strs;

    // Generate appropriate expressions for each parameter
    for (auto& [name, values] : name_value_mapping) {
        if (values.size() == 1) {
            // Single value: generate equality expression
            key_strs.emplace_back(Sprintf("{} == {}", name, values[0]));
        }
        else if (values.size() > 1) {
            // Multiple values: generate range expression (min to max)
            key_strs.emplace_back(Sprintf("{} >= {} && {} <= {}", name, values[0], name, values.back()));
        }
        else {
            // Handle empty values case - use default value of 0
            key_strs.emplace_back(Sprintf("{} == {}", name, 0));
        }
    }

    // Join all expressions with logical AND
    return JoinStrings(key_strs, " && ");
}

}  // namespace flashck