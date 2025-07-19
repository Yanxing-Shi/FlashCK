#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include <fmt/format.h>

namespace flashck {

// ==============================================================================
// Error Type Enumeration
// ==============================================================================

/**
 * @brief Enumeration of error types with categorized error handling.
 *
 * Each entry represents a specific error category with corresponding error
 * type string and description. Values are assigned explicitly to ensure
 * stability across different compiler versions.
 */
enum class ErrorFlag : uint8_t {
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

// ==============================================================================
// Error Name Mapping
// ==============================================================================

namespace detail {

// Compile-time error name mapping for better performance
constexpr std::array<std::string_view, 13> kErrorNames = {
    "LegacyError",              // LEGACY = 0
    "InvalidArgumentError",     // INVALID_ARGUMENT = 1
    "NotFoundError",            // NOT_FOUND = 2
    "OutOfRangeError",          // OUT_OF_RANGE = 3
    "AlreadyExistsError",       // ALREADY_EXISTS = 4
    "ResourceExhaustedError",   // RESOURCE_EXHAUSTED = 5
    "PreconditionNotMetError",  // PRECONDITION_NOT_MET = 6
    "PermissionDeniedError",    // PERMISSION_DENIED = 7
    "ExecutionTimeout",         // EXECUTION_TIMEOUT = 8
    "UnimplementedError",       // UNIMPLEMENTED = 9
    "UnavailableError",         // UNAVAILABLE = 10
    "FatalError",               // FATAL = 11
    "ExternalError"             // EXTERNAL = 12
};

}  // namespace detail

/**
 * @brief Get the string name for an error flag.
 * @param flag The error flag to get the name for
 * @return String view of the error name
 * @throws std::invalid_argument if the error flag is undefined
 */
constexpr std::string_view GetErrorName(ErrorFlag flag)
{
    const auto index = static_cast<size_t>(flag);
    if (index >= detail::kErrorNames.size()) {
        throw std::invalid_argument("The error type is undefined.");
    }
    return detail::kErrorNames[index];
}

/**
 * @brief Check if an error flag is valid.
 * @param flag The error flag to validate
 * @return True if the flag is valid, false otherwise
 */
constexpr bool IsValidErrorFlag(ErrorFlag flag) noexcept
{
    const auto index = static_cast<size_t>(flag);
    return index < detail::kErrorNames.size();
}

// ==============================================================================
// Error Summary Class
// ==============================================================================

/**
 * @brief Encapsulates error information with type and message.
 *
 * Provides structured error handling with categorized error types and
 * formatted error messages. Optimized for performance with move semantics
 * and efficient string handling.
 */
class ErrorSummary {
public:
    /**
     * @brief Constructs an ErrorSummary with specified error type and message.
     * @param flag Error category from ErrorFlag enumeration
     * @param msg Detailed error description
     */
    explicit ErrorSummary(ErrorFlag flag, std::string msg): flag_(flag), msg_(std::move(msg))
    {
        // Validate the error flag at construction time
        if (!IsValidErrorFlag(flag)) {
            flag_ = ErrorFlag::LEGACY;
        }
    }

    /**
     * @brief Constructs an ErrorSummary with LEGACY flag and message.
     * @param msg Error message
     */
    explicit ErrorSummary(std::string msg): flag_(ErrorFlag::LEGACY), msg_(std::move(msg)) {}

    /**
     * @brief Gets the error category flag.
     * @return ErrorFlag enumeration value
     */
    ErrorFlag GetErrorFlag() const noexcept
    {
        return flag_;
    }

    /**
     * @brief Gets the detailed error message.
     * @return Const reference to the message string
     */
    const std::string& GetErrorMessage() const noexcept
    {
        return msg_;
    }

    /**
     * @brief Generates formatted string representation of the error.
     * @return String combining error type and message
     */
    std::string ToString() const
    {
        const std::string_view name = GetErrorName(flag_);
        if (msg_.empty()) {
            return std::string(name);
        }
        return fmt::format("{}: {}", name, msg_);
    }

    /**
     * @brief Check if the error summary is empty.
     * @return True if the message is empty, false otherwise
     */
    bool IsEmpty() const noexcept
    {
        return msg_.empty();
    }

    /**
     * @brief Get the size of the error message.
     * @return Size of the error message string
     */
    size_t Size() const noexcept
    {
        return msg_.size();
    }

private:
    ErrorFlag   flag_;
    std::string msg_;
};

// ==============================================================================
// Error Factory Functions
// ==============================================================================

/**
 * @brief Macro to register error factory functions.
 * Creates a template function that constructs ErrorSummary objects with
 * type-safe formatting using fmt library.
 */
#define REGISTER_ERROR(FUNC, CONST)                                                                                    \
    template<typename... Args>                                                                                         \
    inline ErrorSummary FUNC(fmt::format_string<Args...> format, Args&&... args)                                       \
    {                                                                                                                  \
        return ErrorSummary(CONST, fmt::vformat(format, fmt::make_format_args(args...)));                              \
    }                                                                                                                  \
    inline ErrorSummary FUNC(const std::string& message)                                                               \
    {                                                                                                                  \
        return ErrorSummary(CONST, message);                                                                           \
    }                                                                                                                  \
    inline ErrorSummary FUNC(const char* message)                                                                      \
    {                                                                                                                  \
        return ErrorSummary(CONST, std::string(message));                                                              \
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

// ==============================================================================
// Utility Functions
// ==============================================================================

/**
 * @brief Create a legacy error (for backward compatibility).
 * @param message Error message
 * @return ErrorSummary with LEGACY flag
 */
inline ErrorSummary LegacyError(const std::string& message)
{
    return ErrorSummary(ErrorFlag::LEGACY, message);
}

/**
 * @brief Create a legacy error with formatting.
 * @param format Format string
 * @param args Format arguments
 * @return ErrorSummary with LEGACY flag
 */
template<typename... Args>
inline ErrorSummary LegacyError(fmt::format_string<Args...> format, Args&&... args)
{
    return ErrorSummary(ErrorFlag::LEGACY, fmt::vformat(format, fmt::make_format_args(args...)));
}

}  // namespace flashck
