#pragma once

#include <fstream>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <fmt/format.h>
#include <hip/hip_runtime.h>

namespace lightinfer {

template<typename... Args>
inline void Fprintf(std::string& s, const char* format, Args... args)
{
    fmt::vformat_to(std::back_inserter(s), format, fmt::make_format_args(args...));
}

inline std::string Sprintf()
{
    return "";
}

template<typename... Args>
inline std::string Sprintf(const Args&... args)
{
    std::string s;
    Fprintf(s, "{}", args...);
    return s;
}

template<typename... Args>
inline std::string Sprintf(const char* format, const Args&... args)
{
    std::string s;
    Fprintf(s, format, args...);
    return s;
}

// printf 1d vector
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "{";
    for (auto it = v.begin(); it != v.end(); ++it) {
        os << *it << (it == v.end() - 1 ? "}" : ", ");
    }

    return os;
}

// print 2d vector
// template<typename T>
// inline std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& lod)
// {
//     os << "{";
//     for (auto& v : lod) {
//         os << "{";
//         bool is_first = true;
//         for (auto& i : v) {
//             if (is_first) {
//                 os << i;
//                 is_first = false;
//             }
//             else {
//                 os << ", " << i;
//             }
//         }
//         os << "}";
//     }
//     os << "}";

//     return os;
// }

// printf tuple
template<typename Type, unsigned N, unsigned Last>
struct tuple_printer {

    static void print(std::ostream& out, const Type& value)
    {
        out << std::get<N>(value) << ", ";
        tuple_printer<Type, N + 1, Last>::print(out, value);
    }
};

template<typename Type, unsigned N>
struct tuple_printer<Type, N, N> {

    static void print(std::ostream& out, const Type& value)
    {
        out << std::get<N>(value);
    }
};

template<typename... Types>
std::ostream& operator<<(std::ostream& out, const std::tuple<Types...>& value)
{
    out << "(";
    tuple_printer<std::tuple<Types...>, 0, sizeof...(Types) - 1>::print(out, value);
    out << ")";
    return out;
}

// Output memopry size in human readable format
std::string HumanReadableSize(double f_size);

/*------------------------------------------debug------------------------------------------------*/
template<typename T>
void PrintToFile(const T*           result,
                 const int          size,
                 const char*        file,
                 hipStream_t        stream,
                 std::ios::openmode open_mode = std::ios::out);

template<typename T>
void PrintAbsMean(const T* buf, uint size, hipStream_t stream, std::string name);

template<typename T>
void PrintToScreen(const T* result, const int size, const std::string& name = "");

template<typename T>
void PrintMatrix(T* ptr, int m, int k, int stride, bool is_device_ptr);

template<typename T>
void CheckMaxVal(const T* result, const int size);

template<typename T>
void CheckAbsMeanVal(const T* result, const int size);

}  // namespace lightinfer