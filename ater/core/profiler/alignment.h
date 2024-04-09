#pragma once

#include "ater/core/utils/dtype.h"

namespace ater {

// Return all of the valid alignment values for the dtype.
inline std::vector<int> GetAlignments(const DataType& data_type)
{
    if (data_type == DataType::FLOAT16 || data_type == DataType::BFLOAT16) {
        return {8, 4, 2, 1};
    }
    else if (data_type == DataType::FLOAT32) {
        return {4, 2, 1};
    }
    else {
        ATER_THROW(Unimplemented("unsupported {} for valid alignment", DataTypeToString(data_type)));
    }
}

// Return the first alignment value that meets the alignment requirement
//  for accessing the `number` of elements. This is dtype dependent.
inline int FindMaxAlignment(const int number, const DataType& dtype)
{
    const auto alignments = GetAlignments(dtype);
    for (const auto& alignment : alignments) {
        if (number % alignment == 0) {
            return alignment;
        }
    }
    return 1;
}

// Return the max alignment value that is valid for all the numbers.
inline int FindMaxAlignMentFrom(const std::vector<int> numbers, const DataType& dtype)
{
    const auto alignments = GetAlignments(dtype);
    for (const auto& alignment : alignments) {
        bool valid = true;
        for (const auto& number : numbers) {
            if (number % alignment != 0) {
                valid = false;
                break;
            }
        }
        if (valid) {
            return alignment;
        }
    }
    return 1;
}

// Return True if the given align value is legitimate for the dtype.
inline bool IsValidAlignment(const int alignment, const DataType& dtype)
{
    // 2-elem-alignment is required by fp16, because async.copy needs at least 32
    // bits. For fp32 dtype values, 1-elem-alignment is valid.
    if (dtype == DataType::FLOAT16 || dtype == DataType::BFLOAT16) {
        return alignment % 2 == 0;
    }
    else if (dtype == DataType::FLOAT32) {
        return true;
    }
    else {
        ATER_THROW(Unimplemented("unsupported {} for valid alignment", DataTypeToString(dtype)));
    }
}

inline int DefaultAlignAB(const int a, const int b, const DataType& dtype)
{
    auto ab = std::__gcd(a, b);
    return FindMaxAlignment(ab, dtype);
}
}  // namespace ater