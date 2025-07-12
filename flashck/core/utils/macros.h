#pragma once

#include <string>

#include <glog/logging.h>

#include <hip/hip_runtime.h>
#include <jinja2cpp/template.h>

// Forward declarations to reduce header dependencies
namespace glog {
class LogMessage;
}

namespace flashck {

/**
 * @brief Template loading and rendering utility
 * @param source The template source string
 * @param params Template parameters
 * @return Rendered template string
 * @throws std::runtime_error if template loading or rendering fails
 */
std::string TemplateLoadAndRender(const std::string& source, const jinja2::ValuesMap& params);

/**
 * @brief Get formatted HIP error message
 * @param err HIP error code
 * @param call The function call that failed
 * @return Formatted error message
 */
std::string GetHipErrorMessage(hipError_t err, const char* call) noexcept;

/**
 * @brief Check HIP error and throw exception on failure
 * @param err HIP error code
 * @param call The function call that failed
 * @param file Source file name
 * @param line Source line number
 */
[[noreturn]] void ThrowHipError(hipError_t err, const char* call, const char* file, int line);

/**
 * @brief Log HIP warning
 * @param err HIP error code
 * @param call The function call that failed
 * @param file Source file name
 * @param line Source line number
 */
void LogHipWarning(hipError_t err, const char* call, const char* file, int line) noexcept;

/**
 * @brief Throw Jinja2 template error
 * @param error Jinja2 error message
 * @param expr The expression that failed
 * @param context Additional context information
 * @param file Source file name
 * @param line Source line number
 */
[[noreturn]] void
ThrowJinjaError(const std::string& error, const char* expr, const char* context, const char* file, int line);

/**
 * @brief Optimized template loading and rendering with caching
 * @param source The template source string
 * @param params Template parameters
 * @return Rendered template string
 * @note Uses thread-local cache for better performance on repeated templates
 */
std::string TemplateLoadAndRenderCached(const std::string& source, const jinja2::ValuesMap& params);

/**
 * @brief Clear the template cache
 * @note Useful for memory management in long-running applications
 */
void ClearTemplateCache() noexcept;

/**
 * @brief Optimized HIP error checking macro with detailed error reporting
 * This macro provides comprehensive error checking with file/line information
 * and uses likely/unlikely hints for better branch prediction
 */
#define HIP_ERROR_CHECK(call)                                                                                          \
    do {                                                                                                               \
        const hipError_t _hip_err_ = (call);                                                                           \
        if (_hip_err_ != hipSuccess) [[unlikely]] {                                                                    \
            ::flashck::ThrowHipError(_hip_err_, #call, __FILE__, __LINE__);                                            \
        }                                                                                                              \
    } while (0)

/**
 * @brief HIP warning checking macro for non-critical errors
 * This macro logs warnings without throwing exceptions
 */
#define HIP_WARN_CHECK(call)                                                                                           \
    do {                                                                                                               \
        const hipError_t _hip_stat_ = (call);                                                                          \
        if (_hip_stat_ != hipSuccess) [[unlikely]] {                                                                   \
            ::flashck::LogHipWarning(_hip_stat_, #call, __FILE__, __LINE__);                                           \
        }                                                                                                              \
    } while (0)

/**
 * @brief Conditional HIP error checking macro
 * Only performs checking when condition is true
 */
#define HIP_ERROR_CHECK_IF(condition, call)                                                                            \
    do {                                                                                                               \
        if (condition) {                                                                                               \
            HIP_ERROR_CHECK(call);                                                                                     \
        }                                                                                                              \
    } while (0)

/**
 * @brief HIP error checking macro that returns error code instead of throwing
 * Useful for functions that need to handle errors gracefully
 */
#define HIP_ERROR_RETURN(call)                                                                                         \
    [&]() -> hipError_t {                                                                                              \
        const hipError_t _hip_err_ = (call);                                                                           \
        if (_hip_err_ != hipSuccess) [[unlikely]] {                                                                    \
            ::flashck::LogHipWarning(_hip_err_, #call, __FILE__, __LINE__);                                            \
        }                                                                                                              \
        return _hip_err_;                                                                                              \
    }()

/**
 * @brief Template validation macro for Jinja2 operations
 * Provides better error reporting for template operations
 */
#define JINJA_CHECK(expr, context)                                                                                     \
    do {                                                                                                               \
        auto _result_ = (expr);                                                                                        \
        if (!_result_) [[unlikely]] {                                                                                  \
            ::flashck::ThrowJinjaError(_result_.error(), #expr, context, __FILE__, __LINE__);                          \
        }                                                                                                              \
    } while (0)

/**
 * @brief Debug mode HIP error checking (only active in debug builds)
 * This macro is optimized away in release builds for better performance
 */
#ifdef NDEBUG
#define HIP_DEBUG_CHECK(call) (call)
#else
#define HIP_DEBUG_CHECK(call) HIP_ERROR_CHECK(call)
#endif

}  // namespace flashck
