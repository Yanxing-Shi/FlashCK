#pragma once

#include <string>

#include "flashck/core/utils/printf.h"

namespace flashck {

/**
 * @brief Enumeration of error types with categorized error handling.
 *
 * Each entry represents a specific error category with corresponding error
 * type string and description.
 */
enum class ErrorFlag {
    /// Legacy error type (string: "LegacyError")
    LEGACY = 0,

    /// Client specified an invalid argument (string: "InvalidArgumentError")
    INVALID_ARGUMENT = 1,

    /// Requested entity not found (string: "NotFoundError")
    NOT_FOUND = 2,

    /// Operation past valid input range (string: "OutOfRangeError")
    OUT_OF_RANGE = 3,

    /// Entity already exists (string: "AlreadyExistsError")
    ALREADY_EXISTS = 4,

    /// Resource exhausted (string: "ResourceExhaustedError")
    RESOURCE_EXHAUSTED = 5,

    /// System state precondition not met (string: "PreconditionNotMetError")
    PRECONDITION_NOT_MET = 6,

    /// Permission denied (string: "PermissionDeniedError")
    PERMISSION_DENIED = 7,

    /// Execution deadline expired (string: "ExecutionTimeout")
    EXECUTION_TIMEOUT = 8,

    /// Unimplemented operation (string: "UnimplementedError")
    UNIMPLEMENTED = 9,

    /// Service unavailable (string: "UnavailableError")
    UNAVAILABLE = 10,

    /// Fatal system invariant violation (string: "FatalError")
    FATAL = 11,

    /// Third-party library error (string: "ExternalError")
    EXTERNAL = 12,
};

static std::string GetErrorName(ErrorFlag flag)
{
    switch (flag) {
        case ErrorFlag::LEGACY:
            return "LegacyError";
            break;
        case ErrorFlag::INVALID_ARGUMENT:
            return "InvalidArgumentError";
            break;
        case ErrorFlag::NOT_FOUND:
            return "NotFoundError";
            break;
        case ErrorFlag::OUT_OF_RANGE:
            return "OutOfRangeError";
            break;
        case ErrorFlag::PERMISSION_DENIED:
            return "PermissionDeniedError";
            break;
        case ErrorFlag::UNIMPLEMENTED:
            return "UnimplementedError";
            break;
        case ErrorFlag::UNAVAILABLE:
            return "UnavailableError";
            break;
        case ErrorFlag::FATAL:
            return "FatalError";
            break;
        default:
            throw std::invalid_argument("The error type is undefined.");
            break;
    }
}

/**
 * @brief Encapsulates error information with type and message.
 *
 * Provides structured error handling with categorized error types and
 * formatted error messages.
 */
class ErrorSummary {
public:
    /**
     * @brief Constructs an ErrorSummary with specified error type and message.
     *
     * @param flag Error category from ErrorFlag enumeration
     * @param msg Detailed error description
     */
    explicit ErrorSummary(ErrorFlag flag, std::string msg): flag_(flag), msg_(std::move(msg)) {}

    /**
     * @brief Gets the error category flag.
     * @return ErrorFlag enumeration value
     */
    ErrorFlag GetErrorFlag() const
    {
        return flag_;
    }

    /**
     * @brief Gets the detailed error message.
     * @return Const reference to the message string
     */
    const std::string GetErrorMessage() const
    {
        return msg_;
    }

    /**
     * @brief Generates formatted string representation of the error.
     * @return String combining error type and message
     */
    std::string ToString() const
    {
        const std::string name = GetErrorName(GetErrorFlag());
        const std::string msg  = GetErrorMessage();

        return Sprintf("{}: {}", name, msg);
    }

private:
    ErrorFlag   flag_;
    std::string msg_;
};

#define REGISTER_ERROR(FUNC, CONST)                                                                                    \
    template<typename... Args>                                                                                         \
    inline ErrorSummary FUNC(fmt::format_string<Args...> format, Args&&... args)                                       \
    {                                                                                                                  \
        return ErrorSummary(CONST, fmt::vformat(format, fmt::make_format_args(args...)));                              \
    }

// Error type generator instantiations
REGISTER_ERROR(InvalidArgument, ErrorFlag::INVALID_ARGUMENT)
REGISTER_ERROR(NotFound, ErrorFlag::NOT_FOUND)
REGISTER_ERROR(OutOfRange, ErrorFlag::OUT_OF_RANGE)
REGISTER_ERROR(AlreadyExists, ErrorFlag::ALREADY_EXISTS)
REGISTER_ERROR(ResourceExhausted, ErrorFlag::RESOURCE_EXHAUSTED)
REGISTER_ERROR(PreconditionNotMet, ErrorFlag::PRECONDITION_NOT_MET)
REGISTER_ERROR(PermissionDenied, ErrorFlag::PERMISSION_DENIED)
REGISTER_ERROR(ExecutionTimeout, ErrorFlag::EXECUTION_TIMEOUT)
REGISTER_ERROR(Unimplemented, ErrorFlag::UNIMPLEMENTED)
REGISTER_ERROR(Unavailable, ErrorFlag::UNAVAILABLE)
REGISTER_ERROR(Fatal, ErrorFlag::FATAL)
REGISTER_ERROR(External, ErrorFlag::EXTERNAL)

#undef REGISTER_ERROR

}  // namespace flashck
