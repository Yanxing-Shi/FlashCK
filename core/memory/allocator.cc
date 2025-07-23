#include "core/memory/allocator.h"

#include "core/utils/common.h"

namespace flashck {

Allocator::Allocator(int device_id): device_id_(device_id)
{
    // Create HIP stream for async memory operations
    VLOG(2) << "Allocator initialized for device " << device_id_;
}

Allocator::~Allocator()
{
    // Clean up all tracked allocations before destruction
    while (!ptr_info_.empty()) {
        auto  it        = ptr_info_.begin();
        char* ptr       = it->first;
        bool  is_device = it->second;

        try {
            Free(ptr, is_device);
        }
        catch (const std::exception& e) {
            // Log error but don't throw in destructor to avoid termination
            LOG(ERROR) << "Error freeing pointer " << ptr << " in destructor: " << e.what();
            // Remove from map to prevent infinite loop
            ptr_info_.erase(it);
        }
    }

    VLOG(2) << "Allocator destroyed for device " << device_id_;
}

char* Allocator::Malloc(const size_t size, int init_type, bool is_device)
{
    // Early return for zero-size allocation
    if (size == 0) {
        return nullptr;
    }

    char* tensor_ptr = nullptr;

    // Ensure we're operating on the correct device
    HIP_ERROR_CHECK(hipSetDevice(device_id_));

    if (is_device) {
        // Allocate GPU device memory
        HIP_ERROR_CHECK(hipMalloc((void**)&tensor_ptr, size));

        // Initialize memory with specified pattern if requested
        if (init_type != 0) {
            HIP_ERROR_CHECK(hipMemset(tensor_ptr, init_type, size));
        }
    }
    else {
        // Allocate pinned host memory for efficient GPU transfers
        HIP_ERROR_CHECK(hipHostMalloc((void**)&tensor_ptr, size, hipHostMallocDefault));

        // Initialize host memory if requested
        if (init_type != 0) {
            memset(tensor_ptr, init_type, size);
        }
    }

    VLOG(3) << "Allocated " << size << " bytes at " << (void*)tensor_ptr << " on " << (is_device ? "device" : "host")
            << " " << device_id_;

    // Check for duplicate pointer (should not happen with proper allocation)
    if (ptr_info_.find(tensor_ptr) != ptr_info_.end()) {
        LOG(WARNING) << "Duplicate pointer detected at " << (void*)tensor_ptr << " on device " << device_id_;
    }

    // Track allocation for automatic cleanup
    ptr_info_[tensor_ptr] = is_device;

    return tensor_ptr;
}

void Allocator::Free(char* tensor_ptr, bool is_device)
{
    // Validate pointer and check if it's tracked by this allocator
    if (tensor_ptr == nullptr) {
        LOG(WARNING) << "Attempted to free null pointer";
        return;
    }

    auto it = ptr_info_.find(tensor_ptr);
    if (it == ptr_info_.end()) {
        LOG(WARNING) << "Attempted to free untracked pointer " << (void*)tensor_ptr;
        return;
    }

    // Verify allocation type matches expected type
    if (it->second != is_device) {
        LOG(WARNING) << "Memory type mismatch for pointer " << (void*)tensor_ptr << " - tracked as "
                     << (it->second ? "device" : "host") << ", freeing as " << (is_device ? "device" : "host");
    }

    // Ensure we're operating on the correct device
    HIP_ERROR_CHECK(hipSetDevice(device_id_));

    if (is_device) {
        // Free GPU memory asynchronously for better performance
        HIP_ERROR_CHECK(hipFree(tensor_ptr));
    }
    else {
        // Free pinned host memory
        HIP_ERROR_CHECK(hipHostFree(tensor_ptr));
    }

    VLOG(3) << "Freed pointer " << (void*)tensor_ptr << " from " << (is_device ? "device" : "host") << " "
            << device_id_;

    // Remove from tracking map
    ptr_info_.erase(it);
}

}  // namespace flashck