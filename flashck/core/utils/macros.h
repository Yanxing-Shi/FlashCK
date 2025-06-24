#pragma once

#include <cstdio>
#include <stdexcept>

#include <glog/logging.h>
#include <hip/hip_runtime.h>
#include <jinja2cpp/template.h>
#include <sqlite3.h>

namespace flashck {

/**
 * @brief Validates SQLite operation results and triggers fatal error on failure
 * @param[in] result SQLite result code to check
 * @param[in] db_conn SQLite database connection pointer
 * @param[in] operation Name of the operation being checked
 *
 * @throws Does not throw, but terminates program on error through glog
 *
 * @par Usage:
 * @code
 * CHECK_SQLITE(sqlite3_step(stmt), db, "sqlite3_step");
 * @endcode
 *
 * @note Checks against SQLITE_OK, SQLITE_DONE and SQLITE_ROW return codes.
 *       All other codes trigger fatal error.
 */
#define CHECK_SQLITE(result, db_conn, operation)                                                                       \
    do {                                                                                                               \
        const int _rc = (result);                                                                                      \
        if (_rc != SQLITE_OK && _rc != SQLITE_DONE && _rc != SQLITE_ROW) {                                             \
            LOG(ERROR) << "[SQLITE][" << operation << "] FAILED! "                                                     \
                       << "Code: " << _rc << " | Error: " << sqlite3_errmsg(db_conn) << " | File: " << __FILE__ << ":" \
                       << __LINE__;                                                                                    \
            LOG(FATAL) << "Terminating due to SQLite error";                                                           \
        }                                                                                                              \
    } while (0)

/**
 * @brief Loads and renders Jinja2 template with parameters
 * @param[in] source Template source content
 * @param[in] params Template parameters as key-value pairs
 * @return Rendered template string
 *
 * @throws Does not throw, but may terminate program through CHECK_JINJA
 *
 * @par Example:
 * @code
 * jinja2::ValuesMap params{{"name", "John"}};
 * auto html = TemplateLoadAndRender("Hello {{name}}!", params);
 * @endcode
 */
std::string TemplateLoadAndRender(const std::string& source, const jinja2::ValuesMap& params);

/**
 * @brief Generates formatted HIP error message
 * @param[in] err HIP error code
 * @param[in] call API call expression
 * @return Formatted error string without location info
 *
 * @note File/line information will be automatically added by glog
 */
[[nodiscard]] std::string GetHipErrorMessage(hipError_t err, const char* call);

/**
 * @brief Validates HIP API call and throws on error
 * @param[in] call HIP runtime API function call
 * @throws std::runtime_error containing error context
 *
 * @par Usage example:
 * @code{.cpp}
 * HIP_ERROR_CHECK(hipMalloc(&d_ptr, size));
 * @endcode
 *
 * @note Automatically captures call context via glog
 */
#define HIP_ERROR_CHECK(call)                                                                                          \
    do {                                                                                                               \
        const hipError_t _hip_err_ = (call);                                                                           \
        if (_hip_err_ != hipSuccess) [[unlikely]] {                                                                    \
            const std::string _msg_ = GetHipErrorMessage(_hip_err_, #call);                                            \
            LOG(ERROR) << _msg_;                                                                                       \
            throw std::runtime_error(_msg_);                                                                           \
        }                                                                                                              \
    } while (0)

/**
 * @brief Non-fatal HIP API check with warning logging
 * @param[in] call HIP API function call
 *
 * @par Usage example:
 * @code{.cpp}
 * HIP_WARN_CHECK(hipStreamSynchronize(stream));
 * @endcode
 */
#define HIP_WARN_CHECK(call)                                                                                           \
    do {                                                                                                               \
        const hipError_t _hip_stat_ = (call);                                                                          \
        if (_hip_stat_ != hipSuccess) [[unlikely]] {                                                                   \
            LOG(WARNING) << GetHipErrorMessage(_hip_stat_, #call);                                                     \
        }                                                                                                              \
    } while (0)

}  // namespace flashck
