#pragma once

#include <algorithm>
#include <exception>
#include <regex>
#include <string>
#include <tuple>
#include <vector>

#include "flashck/core/utils/log.h"

#include "flashck/core/utils/named_tuple_utils.h"
#include "flashck/core/utils/string_utils.h"

namespace flashck {

// Constants for profiler configuration
static constexpr int g_profiler_run_max_attempts        = 3;
static constexpr int g_profiler_run_retry_delay_seconds = 10000;

// Regular expressions for parsing kernel and time values
static const std::regex kernel_pattern(R"(KERNEL:([a-zA-Z0-9_]+))");  // Captures kernel name
static const std::regex time_pattern(R"(TIME:([\d\.]+))");            // Captures time value

/**
 * @brief CRTP base class for named tuples (compile-time polymorphism)
 * @tparam N Number of elements in the tuple
 * @tparam Derived The actual derived tuple type
 */
template<size_t N, typename Derived>
struct NamedTupleBase {
    /// @brief Compile-time size information
    static constexpr size_t size()
    {
        return N;
    }

    /**
     * @brief Compile-time name access
     * @tparam I Index of the element (0-based)
     * @return C-string with the element name
     */
    template<size_t I>
    static constexpr const char* name()
    {
        static_assert(I < N, "Index out of bounds");
        return Derived::names()[I];
    }
};

/**
 * @brief Type trait to check if a type is a named tuple
 * @tparam T Type to check
 */
template<typename T>
struct is_namedtuple: std::false_type {};

/// @cond
template<typename T>
struct is_namedtuple<T, std::void_t<decltype(T::namedtuple_tag), decltype(T::names())>>: std::true_type {};
/// @endcond

/// @brief Helper variable template for named tuple check
template<typename T>
inline constexpr bool is_namedtuple_v = is_namedtuple<T>::value;

/**
 * @def NAMEDTUPLE(...)
 * @brief Creates a named tuple structure with specified members
 * @param ... Comma-separated list of member declarations (e.g. "int x, float y")
 */
#define NAMEDTUPLE(...)                                                                                                \
    struct: ::flashck::NamedTupleBase<sizeof((const char*[]){#__VA_ARGS__}) / sizeof(const char*), void> {             \
        using Base = ::flashck::NamedTupleBase<sizeof((const char*[]){#__VA_ARGS__}) / sizeof(const char*), void>;     \
                                                                                                                       \
        /// @brief Tag for named tuple detection                               \
    enum { namedtuple_tag };                                                \
                                                                            \
    /// @brief Internal tuple storage for named members                    \
    std::tuple<__VA_ARGS__> data;                                           \
                                                                            \
    /**                                                                     \
     * @brief Element access by index (non-const version)                   \
     * @tparam I Index of the element to access                             \
     */                                                                     \
    template <size_t I>                                                    \
    auto& get() { return std::get<I>(data); }                              \
                                                                            \
    /**                                                                     \
     * @brief Element access by index (const version)                       \
     * @tparam I Index of the element to access                             \
     */                                                                     \
    template <size_t I>                                                    \
    const auto& get() const { return std::get<I>(data); }                  \
                                                                            \
    /**                                                                     \
     * @brief Compile-time names array accessor                             \
     * @return Array of C-style strings with member names                   \
     */                                                                     \
    static constexpr auto names() {                                        \
      return std::array<const char*, sizeof((const char*[]){#__VA_ARGS__})>\
          {#__VA_ARGS__};                                                  \
    }                                                                       \
  };

/**
 * @struct ProfileResult
 * @brief Structure to store profiling results with named members
 *
 * This structure uses the NAMEDTUPLE macro to define named members,
 * providing compile-time introspection and type safety.
 */
struct ProfileResult {
    NAMEDTUPLE(std::string kernel_config, float duration)
};

// Compile-time validation of the structure
static_assert(ProfileResult::size() == 2);
static_assert(ProfileResult::name<0>() == std::string_view("kernel_config"));
static_assert(ProfileResult::name<1>() == std::string_view("duration"));

// struct Point {
//     NAMEDTUPLE(int x, float y, double z);
// };

// // 编译期验证
// static_assert(Point::size() == 3);
// static_assert(Point::name<0>() == std::string_view("x"));
// static_assert(flashck::is_namedtuple_v<Point>);

// // 运行时使用
// Point p;
// p.get<0>()        = 10;              // 访问x
// p.get<1>()        = 3.14f;           // 访问y
// const auto& names = Point::names();  // 获取所有成员名

/**
 * @brief Extracts all matches of a regular expression from a string.
 * @param regex The regular expression to match.
 * @param input The input string to search.
 * @return A vector of matched strings.
 */
inline std::vector<std::string_view> ExtractMatches(const std::regex& regex, std::string_view input)
{
    std::vector<std::string_view> matches;
    std::cregex_iterator          iter(input.begin(), input.end(), regex);
    std::cregex_iterator          end;

    for (; iter != end; ++iter) {
        if (iter->size() > 1) {  // Ensure there is at least one capture group
            matches.emplace_back(iter->str(1).data(), iter->str(1).length());  // Capture the first group
        }
    }

    return matches;
}

/**
 * @brief Extracts profiling results from a given output string.
 * @param output The output string containing kernel and time information.
 * @param return_kernels A set of kernel names to filter results (default is all kernels).
 * @return A tuple containing:
 *         - A vector of ProfileResult objects.
 *         - A boolean indicating if the extraction failed.
 */
inline std::tuple<std::vector<ProfileResult>, bool>
ExtractProfileResult(std::string_view output, const std::set<std::string>& return_kernels = {})
{
    std::vector<ProfileResult> results;
    bool                       failed = false;

    try {
        // Extract kernel configurations and times
        auto kernel_configs = ExtractMatches(kernel_pattern, output);
        auto time_values    = ExtractMatches(time_pattern, output);

        // Ensure we have matching pairs of kernel configs and times
        if (kernel_configs.size() != time_values.size()) {
            throw std::runtime_error("Mismatch between kernel configs and time values");
        }

        // Preallocate memory for results
        results.reserve(kernel_configs.size());

        // Populate results
        for (size_t i = 0; i < kernel_configs.size(); ++i) {
            // Filter kernels if return_kernels is specified
            if (return_kernels.empty() || return_kernels.count(std::string(kernel_configs[i])) > 0) {
                float duration = std::stof(std::string(time_values[i]));  // Convert string to float
                results.emplace_back(std::string(kernel_configs[i]), duration);
            }
        }
    }
    catch (const std::regex_error& e) {
        // Handle regex errors
        failed = true;
    }
    catch (const std::invalid_argument& e) {
        // Handle invalid time values
        failed = true;
    }
    catch (const std::exception& e) {
        // Handle other exceptions
        failed = true;
    }

    return std::make_tuple(results, failed);
}

}  // namespace flashck