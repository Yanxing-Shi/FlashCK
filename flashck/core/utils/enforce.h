#pragma once

#include <array>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unistd.h>

#include <fmt/format.h>

#include "flashck/core/utils/errors.h"

namespace flashck {

// ==============================================================================
// Type Traits for Template Metaprogramming
// ==============================================================================

template<typename T>
inline constexpr bool IsArithmetic()
{
    return std::is_arithmetic_v<T>;
}

template<typename T1, typename T2, bool kIsArithmetic = true>
struct TypeConverterImpl {
    using Type1 = typename std::common_type_t<T1, T2>;
    using Type2 = Type1;
};

template<typename T1, typename T2>
struct TypeConverterImpl<T1, T2, false> {
    using Type1 = T1;
    using Type2 = T2;
};

template<typename T1, typename T2>
struct TypeConverter {
    static constexpr bool kIsArithmetic = IsArithmetic<T1>() && IsArithmetic<T2>();
    using Type1                         = typename TypeConverterImpl<T1, T2, kIsArithmetic>::Type1;
    using Type2                         = typename TypeConverterImpl<T1, T2, kIsArithmetic>::Type2;
};

template<typename T1, typename T2>
using CommonType1 =
    typename std::add_lvalue_reference_t<typename std::add_const_t<typename TypeConverter<T1, T2>::Type1>>;

template<typename T1, typename T2>
using CommonType2 =
    typename std::add_lvalue_reference_t<typename std::add_const_t<typename TypeConverter<T1, T2>::Type2>>;

// ==============================================================================
// String Conversion Utilities
// ==============================================================================

// SFINAE check for types that can be converted to string via std::cout
template<typename T>
struct CanToString {
private:
    using YesType = uint8_t;
    using NoType  = uint16_t;

    template<typename U>
    static YesType Check(decltype(std::cout << std::declval<U>())*)
    {
        return 0;
    }

    template<typename U>
    static NoType Check(...)
    {
        return 0;
    }

public:
    static constexpr bool kValue = std::is_same_v<YesType, decltype(Check<T>(nullptr))>;
};

template<bool kCanToString = true>
struct BinaryCompareMessageConverter {
    template<typename T>
    static std::string Convert(const char* expression, const T& value)
    {
        return std::string(expression) + ":" + ToString(value);
    }
};

template<>
struct BinaryCompareMessageConverter<false> {
    template<typename T>
    static const char* Convert(const char* expression, const T& /* value */)
    {
        return expression;
    }
};

// ==============================================================================
// Stack Trace and Error Formatting Functions
// ==============================================================================

int         GetCallStackLevel();
std::string GetCurrentTraceBackString(bool for_signal = false);
std::string SimplifyErrorTypeFormat(const std::string& str);
void        InternalThrowWarning(const std::string& message);

// ==============================================================================
// Error Message Formatting Templates
// ==============================================================================

template<typename StrType>
static std::string GetErrorSummaryString(StrType&& what, const char* file, int line)
{
    std::ostringstream sout;
    if (GetCallStackLevel() > 1) {
        sout << "\n----------------------\n"
             << "Error Message Summary:\n"
             << "----------------------\n";
    }

    sout << fmt::format("{} ({}:{})\n", std::forward<StrType>(what), file, line);
    return sout.str();
}

template<typename StrType>
std::string GetCompleteTraceBackString(StrType&& what, const char* file, int line)
{
    std::ostringstream sout;
    sout << "\n----------------------\n"
         << "Error Message Summary:\n"
         << "----------------------\n";

    sout << fmt::format("{} ({}:{})\n", std::forward<StrType>(what), file, line);
    return GetCurrentTraceBackString() + sout.str();
}

template<typename StrType>
static std::string GetTraceBackString(StrType&& what, const char* file, int line)
{
    if (GetCallStackLevel() > 1) {
        // Show full C++ stack trace when level > 1
        return GetCurrentTraceBackString() + GetErrorSummaryString(what, file, line);
    }
    else {
        // Show only error summary when level <= 1
        return GetErrorSummaryString(what, file, line);
    }
}

// ==============================================================================
// Utility Functions
// ==============================================================================

inline bool is_error(bool stat)
{
    return !stat;
}

// ==============================================================================
// Exception Classes
// ==============================================================================

// Internal macro for throwing errors
#define __THROW_ERROR_INTERNAL__(error_summary)                                                                        \
    do {                                                                                                               \
        throw EnforceNotMet(error_summary, __FILE__, __LINE__);                                                        \
    } while (0)

/**
 * @brief Main exception class for FlashCK enforcement failures
 *
 * This class provides comprehensive error handling with stack traces,
 * error categorization, and different verbosity levels based on the
 * FC_CALL_STACK_LEVEL flag.
 */
class EnforceNotMet: public std::exception {
public:
    // Constructor from std::exception_ptr
    EnforceNotMet(std::exception_ptr e, const char* file, int line)
    {
        try {
            std::rethrow_exception(e);
        }
        catch (EnforceNotMet& e) {
            flag_           = e.GetErrorFlag();
            err_str_        = GetTraceBackString(e.what(), file, line);
            simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
        }
        catch (std::exception& e) {
            flag_           = ErrorFlag::LEGACY;
            err_str_        = GetTraceBackString(e.what(), file, line);
            simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
        }
    }

    // Constructor from string message
    EnforceNotMet(const std::string& str, const char* file, int line): err_str_(GetTraceBackString(str, file, line))
    {
        simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
    }

    // Constructor from ErrorSummary
    EnforceNotMet(const ErrorSummary& error, const char* file, int line):
        flag_(error.GetErrorFlag()), err_str_(GetTraceBackString(error.ToString(), file, line))
    {
        simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
    }

    // Override what() to return appropriate error message based on call stack level
    const char* what() const noexcept override
    {
        if (GetCallStackLevel() > 1) {
            return err_str_.c_str();
        }
        else {
            return simple_err_str_.c_str();
        }
    }

    // Accessors
    ErrorFlag GetErrorFlag() const
    {
        return flag_;
    }
    const std::string& GetErrorStr() const
    {
        return err_str_;
    }
    const std::string& SimpleErrorStr() const
    {
        return simple_err_str_;
    }

    // Set error string based on call stack level
    void SetErrorStr(std::string str)
    {
        if (GetCallStackLevel() > 1) {
            err_str_ = std::move(str);
        }
        else {
            simple_err_str_ = std::move(str);
        }
    }

    ~EnforceNotMet() override = default;

private:
    ErrorFlag   flag_ = ErrorFlag::LEGACY;  // Error categorization
    std::string err_str_;                   // Full error message with stack trace
    std::string simple_err_str_;            // Simplified error message
};

// ==============================================================================
// Enforcement Macros
// ==============================================================================

/**
 * @brief Throw an exception with error summary
 * Usage: FC_THROW(ErrorFlag::INVALID_ARGUMENT, "Invalid parameter: {}", param_name);
 */
#define FC_THROW(...)                                                                                                  \
    do {                                                                                                               \
        throw EnforceNotMet(ErrorSummary(__VA_ARGS__), __FILE__, __LINE__);                                            \
    } while (0)

/**
 * @brief Enforce that a pointer is not null
 * Usage: FC_ENFORCE_NOT_NULL(ptr, "Pointer cannot be null");
 */
#define FC_ENFORCE_NOT_NULL(val, ...)                                                                                  \
    do {                                                                                                               \
        if (nullptr == (val)) {                                                                                        \
            auto error_summary = ErrorSummary(__VA_ARGS__);                                                            \
            auto message = fmt::format("{}\n  [Hint: " #val " should not be null.]", error_summary.GetErrorMessage()); \
            __THROW_ERROR_INTERNAL__(ErrorSummary(error_summary.GetErrorFlag(), std::move(message)));                  \
        }                                                                                                              \
    } while (0)

/**
 * @brief Warn if a pointer is null (non-fatal)
 * Usage: FC_WARN_NOT_NULL(ptr, "Warning: pointer should not be null");
 */
#define FC_WARN_NOT_NULL(val, ...)                                                                                     \
    do {                                                                                                               \
        if (nullptr == (val)) {                                                                                        \
            auto error_summary = ErrorSummary(__VA_ARGS__);                                                            \
            auto message = fmt::format("{}\n  [Hint: " #val " should not be null.]", error_summary.GetErrorMessage()); \
            InternalThrowWarning(std::move(message));                                                                  \
        }                                                                                                              \
    } while (0)

/**
 * @brief Internal binary comparison macro for type-safe comparisons
 * This macro handles arithmetic type conversions and provides detailed error messages
 */
#define __FC_BINARY_COMPARE(val1, val2, op, inv_op, ...)                                                                   \
    do {                                                                                                                   \
        auto __val1            = (val1);                                                                                   \
        auto __val2            = (val2);                                                                                   \
        using __TYPE1__        = decltype(__val1);                                                                         \
        using __TYPE2__        = decltype(__val2);                                                                         \
        using __COMMON_TYPE1__ = CommonType1<__TYPE1__, __TYPE2__>;                                                        \
        using __COMMON_TYPE2__ = CommonType2<__TYPE1__, __TYPE2__>;                                                        \
        bool __is_not_error    = (static_cast<__COMMON_TYPE1__>(__val1))op(static_cast<__COMMON_TYPE2__>(__val2));         \
        if (!__is_not_error) {                                                                                             \
            auto           error_summary    = ErrorSummary(__VA_ARGS__);                                                   \
            constexpr bool __kCanToString__ = CanToString<__TYPE1__>::kValue && CanToString<__TYPE2__>::kValue;            \
            auto           message = fmt::format("{}\n  [Hint: Expected {} " #op " {}, but received {} " #inv_op " {}]",   \
                                       error_summary.GetErrorMessage(),                                          \
                                       #val1,                                                                    \
                                       #val2,                                                                    \
                                       BinaryCompareMessageConverter<__kCanToString__>::Convert(#val1, __val1),  \
                                       BinaryCompareMessageConverter<__kCanToString__>::Convert(#val2, __val2)); \
            __THROW_ERROR_INTERNAL__(ErrorSummary(error_summary.GetErrorFlag(), std::move(message)));                      \
        }                                                                                                                  \
    } while (0)

// Binary comparison macros
#define FC_ENFORCE_EQ(val1, val2, ...) __FC_BINARY_COMPARE(val1, val2, ==, !=, __VA_ARGS__)
#define FC_ENFORCE_NE(val1, val2, ...) __FC_BINARY_COMPARE(val1, val2, !=, ==, __VA_ARGS__)
#define FC_ENFORCE_GT(val1, val2, ...) __FC_BINARY_COMPARE(val1, val2, >, <=, __VA_ARGS__)
#define FC_ENFORCE_GE(val1, val2, ...) __FC_BINARY_COMPARE(val1, val2, >=, <, __VA_ARGS__)
#define FC_ENFORCE_LT(val1, val2, ...) __FC_BINARY_COMPARE(val1, val2, <, >=, __VA_ARGS__)
#define FC_ENFORCE_LE(val1, val2, ...) __FC_BINARY_COMPARE(val1, val2, <=, >, __VA_ARGS__)

}  // namespace flashck