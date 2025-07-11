#include "flashck/core/memory/allocator.h"

#include "flashck/core/utils/common.h"

namespace flashck {

Allocator::Allocator(int device_id): device_id_(device_id), stream_(nullptr)
{
    HIP_ERROR_CHECK(hipStreamCreate(&stream_));
}

Allocator::~Allocator()
{
    // Free all allocated pointers
    while (!ptr_info_.empty()) {
        auto  it        = ptr_info_.begin();
        char* ptr       = it->first;
        bool  is_device = it->second;

        try {
            Free(ptr, is_device);
        }
        catch (const std::exception& e) {
            // Log error but don't throw in destructor
            LOG(ERROR) << "Error freeing pointer in destructor: " << e.what();
            // Remove from map to avoid infinite loop
            ptr_info_.erase(it);
        }
    }
}

char* Allocator::Malloc(const size_t size, int init_type, bool is_device)
{
    if (size == 0) {
        return nullptr;
    }

    char* tensor_ptr = nullptr;

    HIP_ERROR_CHECK(hipSetDevice(device_id_));

    if (is_device) {
        // Allocate device memory
        HIP_ERROR_CHECK(hipMalloc((void**)&tensor_ptr, size));

        // Initialize memory if requested
        if (init_type != 0) {
            HIP_ERROR_CHECK(hipMemset(tensor_ptr, init_type, size));
        }
    }
    else {
        // Allocate pinned host memory
        HIP_ERROR_CHECK(hipHostMalloc((void**)&tensor_ptr, size, hipHostMallocDefault));
    }

    VLOG(1) << "Allocator Malloc " << size << " bytes on device: " << device_id_;

    if (ptr_info_.find(tensor_ptr) != ptr_info_.end()) {
        LOG(WARNING) << "ptr_info_ already has info of ptr at " << tensor_ptr << " on device " << device_id_;
    }

    ptr_info_[tensor_ptr] = is_device;

    return tensor_ptr;
}

void Allocator::Free(char* tensor_ptr, bool is_device)
{
    if (tensor_ptr == nullptr || ptr_info_.find(tensor_ptr) == ptr_info_.end()) {
        LOG(WARNING) << "ptr_info_ does not have info of tensor ptr";
        return;
    }

    HIP_ERROR_CHECK(hipSetDevice(device_id_));

    if (is_device) {
        HIP_ERROR_CHECK(hipFreeAsync(tensor_ptr, stream_));
        HIP_ERROR_CHECK(hipStreamSynchronize(stream_));
    }
    else {
        HIP_ERROR_CHECK(hipHostFree(tensor_ptr));
    }

    ptr_info_.erase(tensor_ptr);
}

}  // namespace flashck