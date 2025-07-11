#include "flashck/core/memory/allocator.h"

#include "flashck/core/utils/common.h"

namespace flashck {

Allocator::Allocator(int device_id): device_id_(device_id), stream_(nullptr)
{
    ScopedContext guard(device_id_);
    HIP_ERROR_CHECK(hipStreamCreate(&stream_));
}

Allocator::~Allocator()
{
    for (auto& [ptr, is_device] : ptr_info_) {
        if (is_device) {
            // device memory
            HIP_ERROR_CHECK(hipFreeAsync(ptr, stream_));
        }
        else {
            // host memory
            HIP_ERROR_CHECK(hipHostFree(ptr));
        }
    }

    if (stream_ != nullptr) {
        HIP_ERROR_CHECK(hipStreamSynchronize(stream_));
        HIP_ERROR_CHECK(hipStreamDestroy(stream_));
    }

    ptr_info_.clear();
}

void Allocator::DeviceMalloc(char*& tensor_ptr, size_t size, int init_type)
{
    if (init_type == 0) {
        // Standard allocation
        HIP_ERROR_CHECK(hipMalloc((void**)&tensor_ptr, size));
    }
    else {
        // Zero-initialized allocation
        HIP_ERROR_CHECK(hipMalloc((void**)&tensor_ptr, size));
        HIP_ERROR_CHECK(hipMemset(tensor_ptr, 0, size));
    }
}

void Allocator::GetSetDevice(int device_id)
{
    HIP_ERROR_CHECK(hipSetDevice(device_id));
}

void Allocator::GpuStreamSync(hipStream_t stream)
{
    HIP_ERROR_CHECK(hipStreamSynchronize(stream));
}

char* Allocator::Malloc(const size_t size, int init_type, bool is_device)
{
    if (size == 0) {
        return nullptr;
    }

    char* tensor_ptr = nullptr;

    ScopedContext guard(device_id_);

    if (is_device) {
        // Allocate device memory
        DeviceMalloc(tensor_ptr, size, init_type);
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

    ScopedContext guard(device_id_);

    if (is_device) {
        HIP_ERROR_CHECK(hipFreeAsync(tensor_ptr, stream_));
        GpuStreamSync(stream_);
    }
    else {
        HIP_ERROR_CHECK(hipHostFree(tensor_ptr));
    }

    ptr_info_.erase(tensor_ptr);
}

}  // namespace flashck