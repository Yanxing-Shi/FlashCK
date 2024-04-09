#include "ater/core/utils/errors.h"

#include <stdexcept>

namespace ater {

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

std::string ErrorSummary::ToString() const
{
    std::string result(GetErrorName(GetErrorFlag()));
    result += ": ";
    result += GetErrorMessage();
    return result;
}

}  // namespace ater