#pragma once

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <openssl/sha.h>

#define SHA256_DIGEST_LENGTH 32
#define SHA1_DIGEST_LENGTH 16

namespace lightinfer {

inline size_t CountSpaces(const char* s)
{
    size_t count = 0;

    while (*s != 0 && isspace(*s++)) {
        count++;
    }

    return count;
}

inline size_t CountNonspaces(const char* s)
{
    size_t count = 0;

    while (*s != 0 && !isspace(*s++)) {
        count++;
    }

    return count;
}

// remove leading and tailing spaces
inline std::string TrimSpaces(const std::string& str)
{
    const char* p = str.c_str();

    while (*p != 0 && isspace(*p)) {
        p++;
    }

    size_t len = strlen(p);

    while (len > 0 && isspace(p[len - 1])) {
        len--;
    }

    return std::string(p, len);
}

inline std::string erase_spaces(const std::string& str)
{
    std::string result;
    result.reserve(str.size());
    const char* p = str.c_str();
    while (*p != 0) {
        if (!isspace(*p)) {
            result.append(p, 1);
        }
        ++p;
    }
    return result;
}

// cast string to float
inline int StrToFloat(const char* str, float* v)
{
    const char* head   = str;
    char*       cursor = NULL;
    int         index  = 0;
    while (*(head += CountSpaces(head)) != 0) {
        v[index++] = std::strtof(head, &cursor);
        if (head == cursor) {
            break;
        }
        head = cursor;
    }
    return index;
}

inline float* StrToFloat(std::string& str)
{
    return (float*)const_cast<char*>(str.c_str());
}

inline float* StrToFloat(const char* str)
{
    return (float*)const_cast<char*>(str);
}

// checks whether the test string is a prefix of the input string.
inline bool StartsWith(const std::string& str, const std::string& prefix)
{
    return (str.rfind(prefix, 0) == 0);
}

// checks whether the test string is a suffix of the input string.
inline bool EndsWith(const std::string& str, const std::string& suffix)
{
    if (suffix.length() > str.length()) {
        return false;
    }

    return (str.rfind(suffix) == (str.length() - suffix.length()));
}

// split string by delim
// For string delimiter
template<class T = std::string>
inline std::vector<T> SplitString(const std::string& str, const std::string& delim = ",")
{
    size_t         pos_start = 0, pos_end, delim_len = delim.length();
    std::string    token;
    std::vector<T> res;

    while ((pos_end = str.find(delim, pos_start)) != std::string::npos) {
        token     = str.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(str.substr(pos_start));
    return res;
}

inline std::string
SliceString(const std::string& str, const std::string& given_str, const std::string& direction = "right")
{
    size_t start = str.find(given_str);
    if (start == std::string::npos) {
        return "";
    }
    if (direction == "left")
        return str.substr(0, start);
    else {
        return str.substr(start + 1);
    }
}

template<class Container>
inline std::string JoinStrings(const Container& strs, char delim)
{
    std::string str;

    size_t i = 0;
    for (auto& elem : strs) {
        if (i > 0) {
            str += delim;
        }

        std::stringstream ss;
        ss << elem;
        str += ss.str();
        ++i;
    }

    return str;
}

template<class Container>
std::string JoinStrings(const Container& strs, const std::string& delim)
{
    std::string str;

    size_t i = 0;
    for (auto& elem : strs) {
        if (i > 0) {
            str += delim;
        }

        std::stringstream ss;
        ss << elem;
        str += ss.str();
        ++i;
    }

    return str;
}

template<class Container, class DelimT, class ConvertFunc>
std::string JoinStrings(const Container& strs, DelimT&& delim, ConvertFunc&& func)
{
    std::stringstream ss;
    size_t            i = 0;
    for (const auto& elem : strs) {
        if (i > 0) {
            ss << delim;
        }
        ss << func(elem);
        ++i;
    }

    return ss.str();
}

template<typename T>
inline const std::string JoinToString(const T& sequence, const std::string& separator = ", ")
{
    std::string result;
    for (size_t i = 0; i < sequence.size(); ++i)
        result += static_cast<std::string>(sequence[i]) + ((i != sequence.size() - 1) ? separator : "");
    return result;
}

template const std::string JoinToString(const std::vector<std::string>& sequence, const std::string& separator);
template const std::string JoinToString(const std::vector<std::filesystem::path>& sequence,
                                        const std::string&                        separator);
template<>
inline const std::string JoinToString(const std::vector<int>& sequence, const std::string& separator)
{
    std::string result;
    for (size_t i = 0; i < sequence.size(); ++i)
        result += std::to_string(sequence[i]) + ((i != sequence.size() - 1) ? separator : "");
    return result;
}

template<>
inline const std::string JoinToString(const std::vector<int64_t>& sequence, const std::string& separator)
{
    std::string result;
    for (size_t i = 0; i < sequence.size(); ++i)
        result += std::to_string(sequence[i]) + ((i != sequence.size() - 1) ? separator : "");
    return result;
}

template<>
inline const std::string JoinToString(const std::set<std::filesystem::path>& sequence, const std::string& separator)
{
    std::string result;
    for (size_t i = 0; i < sequence.size(); ++i)
        result +=
            static_cast<std::string>(*std::next(sequence.begin(), i)) + ((i != sequence.size() - 1) ? separator : "");
    return result;
}

template<>
inline const std::string JoinToString(const std::set<std::string>& sequence, const std::string& separator)
{
    std::string result;
    for (size_t i = 0; i < sequence.size(); ++i)
        result +=
            static_cast<std::string>(*std::next(sequence.begin(), i)) + ((i != sequence.size() - 1) ? separator : "");
    return result;
}

inline std::string SHA256ToHexString(const std::string& inputStr)
{
    unsigned char        hash[SHA256_DIGEST_LENGTH];
    const unsigned char* data = (const unsigned char*)inputStr.c_str();
    SHA256(data, inputStr.size(), hash);
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return ss.str();
}

inline std::string SHA1ToHexString(const std::string& inputStr)
{
    unsigned char        hash[SHA256_DIGEST_LENGTH];
    const unsigned char* data = (const unsigned char*)inputStr.c_str();
    SHA1(data, inputStr.size(), hash);
    std::stringstream ss;
    for (int i = 0; i < SHA1_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return ss.str();
}

inline std::string Replace(const std::string& str, const std::string& from, const std::string& to)
{
    std::string str_copy(str);
    size_t      start_pos = str.find(from);
    if (start_pos == std::string::npos)
        return "";
    str_copy.replace(start_pos, from.length(), to);
    return str_copy;
}

// inline void ReplaceAll(std::string& str, const std::string& from, const std::string& to)
// {
//     if (from.empty())
//         return;
//     size_t start_pos = 0;
//     while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
//         str.replace(start_pos, from.length(), to);
//         start_pos += to.length();  // In case 'to' contains 'from', like replacing 'x' with 'yx'
//     }
// }

// inline std::vector<std::string> ToVector(const std::string& str, const std::string& separator = ",")
// {
//     std::vector<std::string> result;
//     std::string              tmp_str(str);
//     size_t                   pos = 0;
//     while ((pos = tmp_str.find(separator)) != std::string::npos) {
//         result.push_back(tmp_str.substr(0, pos));
//         tmp_str.erase(0, pos + separator.length());
//     }
//     result.push_back(tmp_str);
//     return result;
// }

/*---------------------------------To string---------------------------------------*/
template<typename T, typename std::enable_if<!std::is_enum<T>::value, int>::type = 0>
inline std::string ToString(T v)
{
    std::ostringstream sout;
    sout << v;
    return sout.str();
}

template<typename T, typename std::enable_if<std::is_enum<T>::value, int>::type = 0>
inline std::string ToString(T v)
{
    return std::to_string(static_cast<int>(v));
}

// Faster std::string/const char* type
template<>
inline std::string ToString(std::string v)
{
    return v;
}

template<>
inline std::string ToString(const char* v)
{
    return std::string(v);
}

// bug output append ""
// template<>
// inline std::string ToString(const std::filesystem::path& v)
// {
//     return v.string();
// }

// groupby func utils
template<typename Iterator, typename KeyFunc>
struct GroupBy {
    typedef typename KeyFunc::value_type KeyValue;

    struct Range {
        Range(Iterator begin, Iterator end): iter_pair(begin, end) {}

        Iterator begin() const
        {
            return iter_pair.first;
        }
        Iterator end() const
        {
            return iter_pair.second;
        }

    private:
        std::pair<Iterator, Iterator> iter_pair;
    };

    struct Group {
        KeyValue value;
        Range    range;

        Group(KeyValue value, Range range): value(value), range(range) {}
    };

    struct GroupIterator {
        typedef Group value_type;

        GroupIterator(Iterator iter, Iterator end, KeyFunc key_func):
            range_begin(iter), range_end(iter), end(end), key_func(key_func)
        {
            advance_range_end();
        }

        bool operator==(const GroupIterator& that) const
        {
            return range_begin == that.range_begin;
        }

        bool operator!=(const GroupIterator& that) const
        {
            return !(*this == that);
        }

        GroupIterator operator++()
        {
            range_begin = range_end;
            advance_range_end();
            return *this;
        }

        value_type operator*() const
        {
            return value_type(key_func(*range_begin), Range(range_begin, range_end));
        }

    private:
        void advance_range_end()
        {
            if (range_end != end) {
                typename KeyFunc::value_type value = key_func(*range_end++);
                while (range_end != end && key_func(*range_end) == value) {
                    ++range_end;
                }
            }
        }

        Iterator range_begin;
        Iterator range_end;
        Iterator end;
        KeyFunc  key_func;
    };

    GroupBy(Iterator begin_iter, Iterator end_iter, KeyFunc key_func):
        begin_iter(begin_iter), end_iter(end_iter), key_func(key_func)
    {
    }

    GroupIterator begin()
    {
        return GroupIterator(begin_iter, end_iter, key_func);
    }

    GroupIterator end()
    {
        return GroupIterator(end_iter, end_iter, key_func);
    }

private:
    Iterator begin_iter;
    Iterator end_iter;
    KeyFunc  key_func;
};

template<typename Iterator, typename KeyFunc>
inline GroupBy<Iterator, KeyFunc> GroupByFunc(Iterator begin, Iterator end, const KeyFunc& key_func = KeyFunc())
{
    return GroupBy<Iterator, KeyFunc>(begin, end, key_func);
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
std::vector<std::string> GetKeyList(T map)
{
    return Transform(map, [](auto&& p) { return p.first; });
}

inline std::string BoolToString(const bool b)
{
    std::ostringstream ss;
    ss << std::boolalpha << b;
    return ss.str();
}

}  // namespace lightinfer