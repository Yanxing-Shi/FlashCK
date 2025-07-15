#pragma once

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "flashck/core/utils/enforce.h"
#include "flashck/core/utils/errors.h"
#include "flashck/core/utils/printf.h"

namespace flashck {

bool StartsWith(const std::string& str, const std::string& prefix);

bool EndsWith(const std::string& str, const std::string& suffix);

template<typename T, typename = void>
struct is_streamable: std::false_type {};

template<typename T>
struct is_streamable<T, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<const T&>())>>:
    std::true_type {};

template<typename T>
std::enable_if_t<is_streamable<T>::value && !std::is_arithmetic<T>::value, std::string> ToString(const T& value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

template<typename T>
std::enable_if_t<std::is_integral<T>::value && !std::is_same<T, char>::value, std::string> ToString(T value)
{
    return std::to_string(value);
}

template<typename T>
std::enable_if_t<std::is_floating_point<T>::value, std::string> ToString(T value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

inline std::string ToString(const char* value)
{
    return value;
}

template<typename Container>
inline std::string JoinStrings(const Container& container, const std::string& delimiter = "")
{
    std::ostringstream oss;
    auto               it = container.begin();
    if (it != container.end()) {
        oss << *it++;
        for (; it != container.end(); ++it) {
            oss << delimiter << *it;
        }
    }
    return oss.str();
}

// Specialization for std::set<std::filesystem::path>
template<>
inline std::string JoinStrings(const std::set<std::filesystem::path>& container, const std::string& delimiter)
{
    std::ostringstream oss;
    auto               it = container.begin();
    if (it != container.end()) {
        oss << it->string();
        ++it;
        for (; it != container.end(); ++it) {
            oss << delimiter << it->string();
        }
    }
    return oss.str();
}

std::vector<std::string> SplitStrings(const std::string& str, const std::string& delimiter);

std::string
SliceAroundSubstring(const std::string& str, const std::string& separator, const std::string& direction = "right");

std::string HashToHexString(const std::string& input_str);

std::string CombinedHashToHexString(const std::string& input_str);

void ReplaceAll(std::string& s, const std::string& search, const std::string& replacement);

std::map<std::string, int> ExtractWorkLoad(const std::string& key);

std::string GenWorkLoad(const std::map<std::string, std::vector<int64_t>>& name_value_mapping);

template<typename InputIterator, typename KeyFunc>
class GroupBy {
public:
    using KeyType = decltype(std::declval<KeyFunc>()(*std::declval<InputIterator>()));

    static_assert(std::is_convertible_v<typename std::iterator_traits<InputIterator>::iterator_category,
                                        std::forward_iterator_tag>,
                  "Requires at least forward iterators");

    class Group {
    public:
        Group(KeyType key, InputIterator begin, InputIterator end): key_(key), begin_(begin), end_(end) {}

        const KeyType& key() const
        {
            return key_;
        }

        InputIterator begin() const
        {
            return begin_;
        }

        InputIterator end() const
        {
            return end_;
        }

    private:
        const KeyType       key_;
        const InputIterator begin_;
        const InputIterator end_;
    };

    class Iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type        = Group;
        using difference_type   = std::ptrdiff_t;
        using pointer           = value_type*;
        using reference         = value_type&;

        Iterator(InputIterator current, InputIterator end, KeyFunc key_func):
            current_(current), sequence_end_(end), key_func_(key_func)
        {
            AdvanceToNextGroup();
        }

        bool operator==(const Iterator& other) const
        {
            return current_ == other.current_;
        }

        bool operator!=(const Iterator& other) const
        {
            return !(*this == other);
        }

        Iterator& operator++()
        {
            current_ = group_end_;
            AdvanceToNextGroup();
            return *this;
        }

        value_type operator*() const
        {
            return Group(current_key_, current_, group_end_);
        }

    private:
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

    explicit GroupBy(InputIterator begin, InputIterator end, KeyFunc key_func):
        begin_(begin), end_(end), key_func_(key_func)
    {
    }

    Iterator begin()
    {
        return Iterator(begin_, end_, key_func_);
    }

    Iterator end()
    {
        return Iterator(end_, end_, key_func_);
    }

private:
    const InputIterator begin_;
    const InputIterator end_;
    KeyFunc             key_func_;
};

template<typename Iterator, typename KeyFunc>
auto GroupByFunc(Iterator begin, Iterator end, KeyFunc func)
{
    return GroupBy<Iterator, KeyFunc>(begin, end, func);
}

template<class Range, class F>
inline auto Transform(const Range& r, F f) -> std::vector<decltype(f(*r.begin()))>
{
    std::vector<decltype(f(*r.begin()))> result;
    std::transform(r.begin(), r.end(), std::back_inserter(result), f);
    return result;
}

template<class Range1, class Range2, class F>
inline auto Transform(const Range1& r1, const Range2& r2, F f) -> std::vector<decltype(f(*r1.begin(), *r2.begin()))>
{
    std::vector<decltype(f(*r1.begin(), *r2.begin()))> result;
    assert(std::distance(r1.begin(), r1.end()) == std::distance(r2.begin(), r2.end()));
    std::transform(r1.begin(), r1.end(), r2.begin(), std::back_inserter(result), f);
    return result;
}

template<class T>
std::vector<std::string> GetKeyVector(T map)
{
    return Transform(map, [](auto&& p) { return p.first; });
}

}  // namespace flashck