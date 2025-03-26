#pragma once

#include <iostream>

namespace lightinfer {
// kernel source type
enum class SourceType {
    CK             = 0,  // old ck, only for gemm
    CK_TILE        = 1,  // ck tile
    HIP            = 2,
    UNKNOWN_SOURCE = 3,
};

inline std::ostream& operator<<(std::ostream& os, const SourceType& source_type)
{
    switch (source_type) {
        case SourceType::CK:
            os << "CK";
            break;
        case SourceType::CK_TILE:
            os << "CK_TILE";
            break;
        case SourceType::HIP:
            os << "HIP";
            break;
        default:
            // LOG(ERROR) << "SourceType " << static_cast<int>(source_type) << " is not supported.";
            os << "UNDEFINED";
    }
    return os;
}
}  // namespace lightinfer