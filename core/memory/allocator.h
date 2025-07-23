#pragma once

#include <unordered_map>

#include "core/utils/common.h"

namespace flashck {

/**
 * @class Allocator
 * @brief Thread-safe memory allocator for GPU and CPU operations
 *
 * Provides unified interface for allocating and freeing both device (GPU)
 * and host (CPU) memory. Tracks all allocations for automatic cleanup
 * and uses HIP streams for efficient async operations.
 */
class Allocator {
public:
    /**
     * @brief Construct allocator for specified device
     * @param device_id Target HIP device ID (default: 0)
     */
    Allocator(int device_id = 0);

    /**
     * @brief Destructor - automatically frees all tracked allocations
     */
    ~Allocator();

    /**
     * @brief Allocate memory on device or host
     * @param size Number of bytes to allocate
     * @param init_type Initialization value (0 = no init, >0 = memset value)
     * @param is_device True for GPU memory, false for pinned host memory
     * @return Pointer to allocated memory, nullptr if size is 0
     */
    char* Malloc(const size_t size, int init_type = 0, bool is_device = true);

    /**
     * @brief Free previously allocated memory
     * @param tensor_ptr Pointer to memory allocated by this allocator
     * @param is_device Must match the allocation type (device/host)
     */
    void Free(char* tensor_ptr, bool is_device = true);

private:
    std::unordered_map<char*, bool> ptr_info_;   ///< Tracks [pointer -> is_device] for cleanup
    const int                       device_id_;  ///< Target HIP device ID
};

}  // namespace flashck