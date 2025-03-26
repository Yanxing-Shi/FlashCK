#pragma once

#include <iostream>
#include <vector>

/*
Util functions to handle shapes.
*/

namespace lightinfer {

template<typename T>
std::vector<T> SliceVec(const std::vector<T>& vec, int start, int end = -1)
{
    if (end >= 0) {
        end = std::min(end, static_cast<int>(vec.size()));
    }
    else {
        end = vec.size() + end;
    }

    if (start > end) {
        std::cerr << "SliceVec: start > end" << std::endl;
        exit(1);
    }

    if (start == end) {
        return {vec[start]};
    }
    else {
        return std::vector<T>(vec.begin() + start, vec.begin() + end);
    }
}

template<typename T>
void ConcatVec(std::vector<T>& a, const std::vector<T>& b)
{
    a.reserve(a.size() + b.size());
    a.insert(a.end(), b.begin(), b.end());
}

}  // namespace lightinfer