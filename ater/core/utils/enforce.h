#pragma once

#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unistd.h>  // sleep

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <rccl/rccl.h>

#include "ater/core/utils/errors.h"
#include "ater/core/utils/printf.h"
#include "ater/core/utils/string_utils.h"

namespace ater {

template<typename T>
inline constexpr bool IsArithmetic()
{
    return std::is_arithmetic<T>::value;
}

template<typename T1, typename T2, bool kIsArithmetic /* = true */>
struct TypeConverterImpl {
    using Type1 = typename std::common_type<T1, T2>::type;
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
    typename std::add_lvalue_reference<typename std::add_const<typename TypeConverter<T1, T2>::Type1>::type>::type;

template<typename T1, typename T2>
using CommonType2 =
    typename std::add_lvalue_reference<typename std::add_const<typename TypeConverter<T1, T2>::Type2>::type>::type;

// Here, we use SFINAE to check whether T can be converted to std::string
template<typename T>
struct CanToString {
private:
    using YesType = uint8_t;
    using NoType  = uint16_t;

    template<typename U>
    static YesType Check(decltype(std::cout << std::declval<U>()))
    {
        return 0;
    }

    template<typename U>
    static NoType Check(...)
    {
        return 0;
    }

public:
    static constexpr bool kValue = std::is_same<YesType, decltype(Check<T>(std::cout))>::value;
};

template<bool kCanToString /* = true */>
struct BinaryCompareMessageConverter {
    template<typename T>
    static std::string Convert(const char* expression, const T& value)
    {
        return expression + std::string(":") + ToString(value);
    }
};

template<>
struct BinaryCompareMessageConverter<false> {
    template<typename T>
    static const char* Convert(const char* expression, const T& value)
    {
        return expression;
    }
};

int         GetCallStackLevel();
std::string GetCurrentTraceBackString(bool for_signal = false);
std::string SimplifyErrorTypeFormat(const std::string& str);

template<typename StrType>
static std::string GetErrorSumaryString(StrType&& what, const char* file, int line)
{
    std::ostringstream sout;
    if (GetCallStackLevel() > 1) {
        sout << "\n----------------------\nError Message "
                "Summary:\n----------------------\n";
    }
    sout << Sprintf("{} at{} : {}", std::forward<StrType>(what), file, line) << std::endl;
    return sout.str();
}

template<typename StrType>
std::string GetCompleteTraceBackString(StrType&& what, const char* file, int line)
{
    std::ostringstream sout;
    sout << "\n----------------------\nError Message "
            "Summary:\n----------------------\n";

    sout << Sprintf("{} at{} : {}", std::forward<StrType>(what), file, line) << std::endl;

    return GetCurrentTraceBackString() + sout.str();
}

template<typename StrType>
static std::string GetTraceBackString(StrType&& what, const char* file, int line)
{
    if (GetCallStackLevel() > 1) {
        // FLAGS_call_stack_level>1 means showing c++ call stack
        return GetCurrentTraceBackString() + GetErrorSumaryString(what, file, line);
    }
    else {
        return GetErrorSumaryString(what, file, line);
    }
}

inline bool is_error(bool stat)
{
    return !stat;
}

// Note: This Macro can only be used within enforce.h
#define __THROW_ERROR_INTERNAL__(__ERROR_SUMMARY)                                                                      \
    do {                                                                                                               \
        throw EnforceNotMet(__ERROR_SUMMARY, __FILE__, __LINE__);                                                      \
    } while (0)

void InternalThrowWarning(const std::string& message);

/** ENFORCE EXCEPTION AND MACROS **/
class EnforceNotMet: public std::exception {
public:
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
            err_str_        = GetTraceBackString(e.what(), file, line);
            simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
        }
    }

    EnforceNotMet(const std::string& str, const char* file, int line): err_str_(GetTraceBackString(str, file, line))
    {
        simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
    }

    EnforceNotMet(const ErrorSummary& error, const char* file, int line):
        flag_(error.GetErrorFlag()), err_str_(GetTraceBackString(error.ToString(), file, line))
    {
        simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
    }

    const char* what() const noexcept override
    {
        if (GetCallStackLevel() > 1) {
            return err_str_.c_str();
        }
        else {
            return simple_err_str_.c_str();
        }
    }

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

    void SetErrorStr(std::string str)
    {
        if (GetCallStackLevel() > 1) {
            err_str_ = str;
        }
        else {
            simple_err_str_ = str;
        }
    }

    ~EnforceNotMet() override = default;

private:
    // Used to determine the final type of exception thrown
    ErrorFlag flag_ = ErrorFlag::LEGACY;
    // Complete error message
    // e.g. InvalidArgumentError: ***
    std::string err_str_;
    // Simple error message used when no C++ stack and python compile stack
    // e.g. (InvalidArgument) ***
    std::string simple_err_str_;
};

#define ATER_THROW(...)                                                                                                \
    do {                                                                                                               \
        throw EnforceNotMet(ErrorSummary(__VA_ARGS__), __FILE__, __LINE__);                                            \
    } while (0)

#define ATER_ENFORCE(_IS_NOT_ERROR, __FORMAT, ...)                                                                     \
    do {                                                                                                               \
        if (!(_IS_NOT_ERROR)) {                                                                                        \
            printf("Error: %s:%d Assertion `%s` failed. " __FORMAT "\n",                                               \
                   __FILE__,                                                                                           \
                   __LINE__,                                                                                           \
                   #_IS_NOT_ERROR,                                                                                     \
                   ##__VA_ARGS__);                                                                                     \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#define ATER_ENFORCE_NOT_NULL(__VAL, ...)                                                                              \
    do {                                                                                                               \
        if (nullptr == (__VAL)) {                                                                                      \
            auto __summary__ = ErrorSummary(__VA_ARGS__);                                                              \
            auto __message__ = Sprintf("{}\n  [Hint: " #__VAL " should not be null.]", __summary__.GetErrorMessage()); \
            __THROW_ERROR_INTERNAL__(ErrorSummary(__summary__.GetErrorFlag(), std::move(__message__)));                \
        }                                                                                                              \
    } while (0)

#define ATER_WARN_NOT_NULL(__VAL, ...)                                                                                 \
    do {                                                                                                               \
        if (nullptr == (__VAL)) {                                                                                      \
            auto __summary__ = ErrorSummary(__VA_ARGS__);                                                              \
            auto __message__ = Sprintf("{}\n  [Hint: " #__VAL " should not be null.]", __summary__.GetErrorMessage()); \
            InternalThrowWarning(std::move(__message__));                                                              \
        }                                                                                                              \
    } while (0)

#define __ATER_BINARY_COMPARE(__VAL1, __VAL2, __CMP, __INV_CMP, ...)                                                   \
    do {                                                                                                               \
        auto __val1            = (__VAL1);                                                                             \
        auto __val2            = (__VAL2);                                                                             \
        using __TYPE1__        = decltype(__val1);                                                                     \
        using __TYPE2__        = decltype(__val2);                                                                     \
        using __COMMON_TYPE1__ = CommonType1<__TYPE1__, __TYPE2__>;                                                    \
        using __COMMON_TYPE2__ = CommonType2<__TYPE1__, __TYPE2__>;                                                    \
        bool __is_not_error    = (static_cast<__COMMON_TYPE1__>(__val1))__CMP(static_cast<__COMMON_TYPE2__>(__val2));  \
        if (!__is_not_error) {                                                                                         \
            auto           __summary__      = ErrorSummary(__VA_ARGS__);                                               \
            constexpr bool __kCanToString__ = CanToString<__TYPE1__>::kValue && CanToString<__TYPE2__>::kValue;        \
            auto __message__ = Sprintf("{}\n  [Hint: Expected {}" #__CMP "{}], but received {}" #__INV_CMP "{}]",      \
                                       __summary__.GetErrorMessage(),                                                  \
                                       __val1,                                                                         \
                                       __val2,                                                                         \
                                       BinaryCompareMessageConverter<__kCanToString__>::Convert(#__VAL1, __val1),      \
                                       BinaryCompareMessageConverter<__kCanToString__>::Convert(#__VAL2, __val2));     \
            __THROW_ERROR_INTERNAL__(ErrorSummary(__summary__.GetErrorFlag(), std::move(__message__)));                \
        }                                                                                                              \
    } while (0)

#define ATER_ENFORCE_EQ(__VAL0, __VAL1, ...) __ATER_BINARY_COMPARE(__VAL0, __VAL1, ==, !=, __VA_ARGS__)
#define ATER_ENFORCE_NE(__VAL0, __VAL1, ...) __ATER_BINARY_COMPARE(__VAL0, __VAL1, !=, ==, __VA_ARGS__)
#define ATER_ENFORCE_GT(__VAL0, __VAL1, ...) __ATER_BINARY_COMPARE(__VAL0, __VAL1, >, <=, __VA_ARGS__)
#define ATER_ENFORCE_GE(__VAL0, __VAL1, ...) __ATER_BINARY_COMPARE(__VAL0, __VAL1, >=, <, __VA_ARGS__)
#define ATER_ENFORCE_LT(__VAL0, __VAL1, ...) __ATER_BINARY_COMPARE(__VAL0, __VAL1, <, >=, __VA_ARGS__)
#define ATER_ENFORCE_LE(__VAL0, __VAL1, ...) __ATER_BINARY_COMPARE(__VAL0, __VAL1, <=, >, __VA_ARGS__)

/*-----------HIP error-----------*/
inline bool is_error(hipError_t e)
{
    return e != hipSuccess;
}

inline std::string build_rocm_error_msg(hipError_t e)
{
    std::ostringstream sout;
    sout << " Hip error(" << e << "), " << hipGetErrorString(e) << ".";
    return sout.str();
}

/*-----------HIPRAND error-----------*/
inline bool is_error(hiprandStatus_t stat)
{
    return stat != HIPRAND_STATUS_SUCCESS;
}

inline const char* hiprandGetErrorString(hiprandStatus_t stat)
{
    switch (stat) {
        case HIPRAND_STATUS_SUCCESS:
            return "HIPRAND_STATUS_SUCCESS";
        case HIPRAND_STATUS_VERSION_MISMATCH:
            return "HIPRAND_STATUS_VERSION_MISMATCH";
        case HIPRAND_STATUS_NOT_INITIALIZED:
            return "HIPRAND_STATUS_NOT_INITIALIZED";
        case HIPRAND_STATUS_ALLOCATION_FAILED:
            return "HIPRAND_STATUS_ALLOCATION_FAILED";
        case HIPRAND_STATUS_TYPE_ERROR:
            return "HIPRAND_STATUS_TYPE_ERROR";
        case HIPRAND_STATUS_OUT_OF_RANGE:
            return "HIPRAND_STATUS_OUT_OF_RANGE";
        case HIPRAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "HIPRAND_STATUS_LENGTH_NOT_MULTIPLE";
        case HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED";
        case HIPRAND_STATUS_LAUNCH_FAILURE:
            return "HIPRAND_STATUS_LAUNCH_FAILURE";
        case HIPRAND_STATUS_PREEXISTING_FAILURE:
            return "HIPRAND_STATUS_PREEXISTING_FAILURE";
        case HIPRAND_STATUS_INITIALIZATION_FAILED:
            return "HIPRAND_STATUS_INITIALIZATION_FAILED";
        case HIPRAND_STATUS_ARCH_MISMATCH:
            return "HIPRAND_STATUS_ARCH_MISMATCH";
        case HIPRAND_STATUS_INTERNAL_ERROR:
            return "HIPRAND_STATUS_INTERNAL_ERROR";
        case HIPRAND_STATUS_NOT_IMPLEMENTED:
            return "HIPRAND_STATUS_NOT_IMPLEMENTED";
        default:
            return "Unknown hiprand status";
    }
}

inline std::string build_rocm_error_msg(hiprandStatus_t stat)
{
    std::string msg(" Hiprand error, ");
    return msg + hiprandGetErrorString(stat) + " ";
}

/*------------RCCL error---------*/
inline bool is_error(ncclResult_t nccl_result)
{
    return nccl_result != ncclSuccess;
}

inline std::string build_rocm_error_msg(ncclResult_t nccl_result)
{
    std::string msg(" Rccl error, ");
    return msg + ncclGetErrorString(nccl_result) + " ";
}

template<typename T>
struct ROCmLibErrorType {
    using type = T;
};

#define DEFINE_ERROR_TYPE(type, success_value)                                                                         \
    template<>                                                                                                         \
    struct ROCmLibErrorType<type> {                                                                                    \
        using Type                     = type;                                                                         \
        static constexpr Type kSuccess = success_value;                                                                \
    }

// Macor
DEFINE_ERROR_TYPE(hipError_t, hipSuccess);
DEFINE_ERROR_TYPE(hiprandStatus_t, HIPRAND_STATUS_SUCCESS);
DEFINE_ERROR_TYPE(ncclResult_t, ncclSuccess);

#define ATER_WARN_HIP_SUCCESS(COND)                                                                                    \
    do {                                                                                                               \
        auto __cond__                   = (COND);                                                                      \
        using __HIP_STATUS_TYPE__       = decltype(__cond__);                                                          \
        constexpr auto __success_type__ = ROCmLibErrorType<__HIP_STATUS_TYPE__>::kSuccess;                             \
        if (__cond__ != __success_type__) {                                                                            \
            auto __summary__ = InternalThrowWarning(build_rocm_error_msg(__cond__));                                   \
            __THROW_ERROR_INTERNAL__(__summary__);                                                                     \
        }                                                                                                              \
    } while (0)

#define ATER_ENFORCE_HIP_SUCCESS(COND)                                                                                 \
    do {                                                                                                               \
        auto __cond__                   = (COND);                                                                      \
        using __HIP_STATUS_TYPE__       = decltype(__cond__);                                                          \
        constexpr auto __success_type__ = ROCmLibErrorType<__HIP_STATUS_TYPE__>::kSuccess;                             \
        if (__cond__ != __success_type__) {                                                                            \
            auto __summary__ = External("{}", build_rocm_error_msg(__cond__));                                         \
            __THROW_ERROR_INTERNAL__(__summary__);                                                                     \
        }                                                                                                              \
    } while (0)

#define ATER_RETRY_HIP_SUCCESS(COND)                                                                                   \
    do {                                                                                                               \
        auto __cond__                   = (COND);                                                                      \
        int  retry_count                = 1;                                                                           \
        using __HIP_STATUS_TYPE__       = decltype(__cond__);                                                          \
        constexpr auto __success_type__ = ROCmLibErrorType<__HIP_STATUS_TYPE__>::kSuccess;                             \
        while (__cond__ != __success_type__ && retry_count < 5) {                                                      \
            sleep(10000);                                                                                              \
            __cond__ = (COND);                                                                                         \
            ++retry_count;                                                                                             \
        }                                                                                                              \
        if (__cond__ != __success_type__) {                                                                            \
            auto __summary__ = External("{}", build_rocm_error_msg(__cond__));                                         \
            __THROW_ERROR_INTERNAL__(__summary__);                                                                     \
        }                                                                                                              \
    } while (0)

}  // namespace ater