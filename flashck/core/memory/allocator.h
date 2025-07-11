#pragma once

#include <unordered_map>

#include "flashck/core/utils/common.h"

namespace flashck {
class Allocator {
public:
    Allocator(int device_id = 0);

    ~Allocator();

    char* Malloc(const size_t size, int init_type = 0, bool is_device = true);

    void Free(char* tensor_ptr, bool is_device = true);

private:
    std::unordered_map<char*, bool> ptr_info_;   ///< Map of [pointer -> is_device]
    const int                       device_id_;  ///< Target HIP device ID
    hipStream_t                     stream_;     ///< HIP stream for async operations
};
}  // namespace flashck