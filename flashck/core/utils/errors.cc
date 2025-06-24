#include "flashck/core/utils/errors.h"

#include <stdexcept>

namespace flashck {

static const std::string_view GetErrorName(ErrorFlag flag)
{
    switch (flag) {
        case ErrorFlag::LEGACY:
            return ErrorName<ErrorFlag::LEGACY>;
        case ErrorFlag::INVALID_ARGUMENT:
            return ErrorName<ErrorFlag::INVALID_ARGUMENT>;
        case ErrorFlag::NOT_FOUND:
            return ErrorName<ErrorFlag::NOT_FOUND>;
        case ErrorFlag::OUT_OF_RANGE:
            return ErrorName<ErrorFlag::OUT_OF_RANGE>;
        case ErrorFlag::ALREADY_EXISTS:
            return ErrorName<ErrorFlag::ALREADY_EXISTS>;
        case ErrorFlag::RESOURCE_EXHAUSTED:
            return ErrorName<ErrorFlag::RESOURCE_EXHAUSTED>;
        case ErrorFlag::PRECONDITION_NOT_MET:
            return ErrorName<ErrorFlag::PRECONDITION_NOT_MET>;
        case ErrorFlag::PERMISSION_DENIED:
            return ErrorName<ErrorFlag::PERMISSION_DENIED>;
        case ErrorFlag::EXECUTION_TIMEOUT:
            return ErrorName<ErrorFlag::EXECUTION_TIMEOUT>;
        case ErrorFlag::UNIMPLEMENTED:
            return ErrorName<ErrorFlag::UNIMPLEMENTED>;
        case ErrorFlag::UNAVAILABLE:
            return ErrorName<ErrorFlag::UNAVAILABLE>;
        case ErrorFlag::FATAL:
            return ErrorName<ErrorFlag::FATAL>;
        case ErrorFlag::EXTERNAL:
            return ErrorName<ErrorFlag::EXTERNAL>;
        default:
            throw std::invalid_argument("The error type is undefined.");
            break;
    }
}

}  // namespace flashck