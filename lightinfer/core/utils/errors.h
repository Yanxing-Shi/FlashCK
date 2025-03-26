#pragma once

#include <string>

#include "lightinfer/core/utils/printf.h"

namespace lightinfer {

enum class ErrorFlag {
    // Legacy error.
    // Error type string: "LegacyError"
    LEGACY = 0,

    // Client specified an invalid argument.
    // Error type string: "InvalidArgumentError"
    INVALID_ARGUMENT = 1,

    // Some requested entity (e.g., file or directory) was not found.
    // Error type string: "NotFoundError"
    NOT_FOUND = 2,

    // Operation tried to iterate past the valid input range.  E.g., seeking or
    // reading past end of file.
    // Error type string: "OutOfRangeError"
    OUT_OF_RANGE = 3,

    // Some entity that we attempted to create (e.g., file or directory)
    // already exists.
    // Error type string: "AlreadyExistsError"
    ALREADY_EXISTS = 4,

    // Some resource has been exhausted, perhaps a per-user quota, or
    // perhaps the entire file system is out of space.
    // Error type string: "ResourceExhaustedError"
    RESOURCE_EXHAUSTED = 5,

    // Operation was rejected because the system is not in a state
    // required for the operation's execution.
    // Error type string: "PreconditionNotMetError"
    PRECONDITION_NOT_MET = 6,

    // The caller does not have permission to execute the specified
    // operation.
    // Error type string: "PermissionDeniedError"
    PERMISSION_DENIED = 7,

    // Deadline expired before operation could complete.
    // Error type string: "ExecutionTimeout"
    EXECUTION_TIMEOUT = 8,

    // Operation is not implemented or not supported/enabled in this service.
    // Error type string: "UnimplementedError"
    UNIMPLEMENTED = 9,

    // The service is currently unavailable.  This is a most likely a
    // transient condition and may be corrected by retrying with
    // a backoff.
    // Error type string: "UnavailableError"
    UNAVAILABLE = 10,

    // Fatal errors.  Means some invariant expected by the underlying
    // system has been broken.  If you see one of these errors,
    // something is very broken.
    // Error type string: "FatalError"
    FATAL = 11,

    // Third-party library error.
    // Error type string: "ExternalError"
    EXTERNAL = 12,
};

class ErrorSummary {
public:
    // // compatible with current existing untyped ATER_ENFORCE
    // template<typename... Args>
    // explicit ErrorSummary(Args... args)
    // {
    //     flag_ = ErrorFlag::LEGACY;
    //     msg_  = Sprintf(args...);
    // }

    explicit ErrorSummary(ErrorFlag flag, std::string msg): flag_(flag), msg_(msg) {}

    ErrorFlag GetErrorFlag() const
    {
        return flag_;
    }

    const std::string GetErrorMessage() const
    {
        return msg_;
    }

    std::string ToString() const;

private:
    ErrorFlag   flag_;
    std::string msg_;
};

#define REGISTER_ERROR(FUNC, CONST, ...)                                                                               \
    template<typename... Args>                                                                                         \
    ErrorSummary FUNC(const char* format, Args... args)                                                                \
    {                                                                                                                  \
        return ErrorSummary(CONST, Sprintf(format, args...));                                                          \
    }

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

}  // namespace lightinfer