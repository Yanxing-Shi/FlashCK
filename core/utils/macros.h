#pragma once

#include <stdexcept>
#include <string>

#include "jinja2cpp/user_callable.h"
#include <glog/logging.h>
#include <hip/hip_runtime.h>
#include <jinja2cpp/reflected_value.h>
#include <jinja2cpp/template.h>
#include <sqlite3.h>

namespace flashck {

// ==============================================================================
// Template Processing Functions
// ==============================================================================

/*!
 * @brief Load and render Jinja2 template with error handling
 * @param source The template source string
 * @param params Template parameters
 * @return Rendered template string
 * @throws std::runtime_error if template loading or rendering fails
 */
std::string TemplateLoadAndRender(const std::string& source, const jinja2::ValuesMap& params);

/*!
 * @brief Validate and load Jinja2 template
 * @param source The template source string
 * @return Loaded template object
 * @throws std::runtime_error if template loading fails
 */
jinja2::Template TemplateLoad(const std::string& source);

/*!
 * @brief Render template with parameters
 * @param template_obj The loaded template object
 * @param params Template parameters
 * @return Rendered template string
 * @throws std::runtime_error if rendering fails
 */
std::string TemplateRender(jinja2::Template& template_obj, const jinja2::ValuesMap& params);

// ==============================================================================
// HIP Error Handling Functions
// ==============================================================================

/*!
 * @brief Get formatted HIP error message
 * @param err HIP error code
 * @param call The function call that failed
 * @return Formatted error message
 * @note noexcept guarantee for safe error reporting
 */
std::string GetHipErrorMessage(hipError_t err, const char* call) noexcept;

/*!
 * @brief Throw HIP error with detailed context
 * @param err HIP error code
 * @param call The function call that failed
 * @param file Source file name
 * @param line Source line number
 * @note [[noreturn]] attribute for compiler optimization
 */
[[noreturn]] void ThrowHipError(hipError_t err, const char* call, const char* file, int line);

/*!
 * @brief Log HIP warning without throwing
 * @param err HIP error code
 * @param call The function call that failed
 * @param file Source file name
 * @param line Source line number
 * @note noexcept guarantee for safe logging
 */
void LogHipWarning(hipError_t err, const char* call, const char* file, int line) noexcept;

// ==============================================================================
// HIP Error Checking Macros
// ==============================================================================

/*!
 * @brief Primary HIP error checking macro with detailed error reporting
 * @param call HIP function call to check
 * @note Uses [[likely]]/[[unlikely]] hints for optimal branch prediction
 * @note Provides comprehensive error context with file/line information
 */
#define HIP_ERROR_CHECK(call)                                                                                          \
    do {                                                                                                               \
        const hipError_t _hip_err_ = (call);                                                                           \
        if (_hip_err_ != hipSuccess) [[unlikely]] {                                                                    \
            ::flashck::ThrowHipError(_hip_err_, #call, __FILE__, __LINE__);                                            \
        }                                                                                                              \
    } while (0)

/*!
 * @brief HIP warning checking macro for non-critical errors
 * @param call HIP function call to check
 * @note Logs warnings without throwing exceptions
 */
#define HIP_WARN_CHECK(call)                                                                                           \
    do {                                                                                                               \
        const hipError_t _hip_stat_ = (call);                                                                          \
        if (_hip_stat_ != hipSuccess) [[unlikely]] {                                                                   \
            ::flashck::LogHipWarning(_hip_stat_, #call, __FILE__, __LINE__);                                           \
        }                                                                                                              \
    } while (0)

/*!
 * @brief Conditional HIP error checking macro
 * @param condition Boolean condition for checking
 * @param call HIP function call to check
 * @note Only performs checking when condition is true
 */
#define HIP_ERROR_CHECK_IF(condition, call)                                                                            \
    do {                                                                                                               \
        if (condition) {                                                                                               \
            HIP_ERROR_CHECK(call);                                                                                     \
        }                                                                                                              \
    } while (0)

/*!
 * @brief HIP error checking macro that returns error code
 * @param call HIP function call to check
 * @return hipError_t error code
 * @note Useful for functions that need to handle errors gracefully
 */
#define HIP_ERROR_RETURN(call)                                                                                         \
    [&]() -> hipError_t {                                                                                              \
        const hipError_t _hip_err_ = (call);                                                                           \
        if (_hip_err_ != hipSuccess) [[unlikely]] {                                                                    \
            ::flashck::LogHipWarning(_hip_err_, #call, __FILE__, __LINE__);                                            \
        }                                                                                                              \
        return _hip_err_;                                                                                              \
    }()

/*!
 * @brief Debug mode HIP error checking (only active in debug builds)
 * @param call HIP function call to check
 * @note Optimized away in release builds for better performance
 */
#ifdef NDEBUG
#define HIP_DEBUG_CHECK(call) (call)
#else
#define HIP_DEBUG_CHECK(call) HIP_ERROR_CHECK(call)
#endif

// ==============================================================================
// Template Validation Macros
// ==============================================================================

/*!
 * @brief Template validation macro specifically for TemplateLoadAndRender
 * @param source The template source string
 * @param params The template parameters (jinja2::ValuesMap)
 * @param context Additional context string for error reporting
 * @return The rendered template string
 * @note Catches exceptions from TemplateLoadAndRender and provides detailed error context
 */
#define TEMPLATE_CHECK(source, params, context)                                                                        \
    [&]() -> std::string {                                                                                             \
        try {                                                                                                          \
            return ::flashck::TemplateLoadAndRender(source, params);                                                   \
        }                                                                                                              \
        catch (const std::exception& e) {                                                                              \
            const std::string error_msg = std::string("Template error in ") + context + " at " + __FILE__ + ":"        \
                                          + std::to_string(__LINE__) + " - " + e.what();                               \
            LOG(ERROR) << error_msg;                                                                                   \
            throw std::runtime_error(error_msg);                                                                       \
        }                                                                                                              \
    }()

/*!
 * @brief Template validation macro with warning (non-throwing version)
 * @param source The template source string
 * @param params The template parameters (jinja2::ValuesMap)
 * @param context Additional context string for error reporting
 * @return The rendered template string on success, empty string on failure
 * @note Logs warnings without throwing exceptions
 */
#define TEMPLATE_WARN_CHECK(source, params, context)                                                                   \
    [&]() -> std::string {                                                                                             \
        try {                                                                                                          \
            return ::flashck::TemplateLoadAndRender(source, params);                                                   \
        }                                                                                                              \
        catch (const std::exception& e) {                                                                              \
            const std::string error_msg = std::string("Template warning in ") + context + " at " + __FILE__ + ":"      \
                                          + std::to_string(__LINE__) + " - " + e.what();                               \
            LOG(WARNING) << error_msg;                                                                                 \
            return "";                                                                                                 \
        }                                                                                                              \
    }()

/*!
 * @brief Conditional template validation macro
 * @param condition Boolean condition for checking
 * @param source The template source string
 * @param params The template parameters (jinja2::ValuesMap)
 * @param context Additional context string for error reporting
 * @return The rendered template string on success, empty string when condition is false
 * @note Only performs template processing when condition is true
 */
#define TEMPLATE_CHECK_IF(condition, source, params, context)                                                          \
    [&]() -> std::string {                                                                                             \
        if (condition) {                                                                                               \
            return TEMPLATE_CHECK(source, params, context);                                                            \
        }                                                                                                              \
        return "";                                                                                                     \
    }()

/*!
 * @brief Debug mode template checking (only active in debug builds)
 * @param source The template source string
 * @param params The template parameters (jinja2::ValuesMap)
 * @param context Additional context string for error reporting
 * @return The rendered template string in debug mode, calls TemplateLoadAndRender directly in release mode
 * @note Optimized away in release builds for better performance
 */
#ifdef NDEBUG
#define TEMPLATE_DEBUG_CHECK(source, params, context) ::flashck::TemplateLoadAndRender(source, params)
#else
#define TEMPLATE_DEBUG_CHECK(source, params, context) TEMPLATE_CHECK(source, params, context)
#endif

// ==============================================================================
// SQLite Error Checking Macros
// ==============================================================================

/*!
 * @brief SQLite error checking macro
 * @param expr SQLite expression to check
 * @param db SQLite database handle
 * @note Throws std::runtime_error on failure
 */
#define CHECK_SQLITE3(expr, db)                                                                                        \
    do {                                                                                                               \
        int result_code = (expr);                                                                                      \
        if (result_code != SQLITE_OK) {                                                                                \
            const char* err = sqlite3_errmsg(db);                                                                      \
            throw std::runtime_error("SQLite error[" + std::to_string(result_code)                                     \
                                     + "]: " + (err ? err : "unknown error") + " at " + std::string(__FILE__) + ":"    \
                                     + std::to_string(__LINE__));                                                      \
        }                                                                                                              \
    } while (0)

/*!
 * @brief SQLite error checking macro with return code capture
 * @param expr SQLite expression to check
 * @param db SQLite database handle
 * @param rc Variable to store return code
 * @note Allows checking for SQLITE_ROW and SQLITE_DONE as success
 */
#define CHECK_SQLITE3_RC(expr, db, rc)                                                                                 \
    do {                                                                                                               \
        rc = (expr);                                                                                                   \
        if (rc != SQLITE_OK && rc != SQLITE_ROW && rc != SQLITE_DONE) {                                                \
            const char* err = sqlite3_errmsg(db);                                                                      \
            throw std::runtime_error("SQLite error[" + std::to_string(rc) + "]: " + (err ? err : "unknown error")      \
                                     + " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));               \
        }                                                                                                              \
    } while (0)

}  // namespace flashck
