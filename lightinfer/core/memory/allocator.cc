#include "lightinfer/core/memory/allocator.h"

#include "lightinfer/core/utils/enforce.h"
#include "lightinfer/core/utils/log.h"
#include "lightinfer/core/utils/rocm_info.h"

namespace lightinfer {

Allocator::Allocator(int device_id): device_id_(device_id)
{
    int device_count = GetGPUDeviceCount();

    hipMemPool_t hip_mem_pool;
    LI_ENFORCE_HIP_SUCCESS(hipDeviceGetDefaultMemPool(&hip_mem_pool, device_id));
    hipMemAccessDesc desc                  = {};
    int              peer_access_available = 0;
    for (int i = 0; i < device_count; i++) {
        if (i == device_id) {
            continue;
        }
        LI_ENFORCE_HIP_SUCCESS(hipDeviceCanAccessPeer(&peer_access_available, device_id, i));
        if (peer_access_available) {
            LI_ENFORCE_HIP_SUCCESS(hipDeviceGetMemPool(&hip_mem_pool, i));
            desc.location.type = hipMemLocationTypeDevice;
            desc.location.id   = i;
            desc.flags         = hipMemAccessFlagsProtReadWrite;
            LI_ENFORCE_HIP_SUCCESS(hipMemPoolSetAccess(hip_mem_pool, &desc, 1));
        }
        else {
            LOG(WARNING) << "Device " << device_id << " cannot access peer device " << i;
            continue;
        }
    }

    // set memory pool threshold to avoid shrinking the pool
    uint64_t set_val = UINT64_MAX;
    LI_ENFORCE_HIP_SUCCESS(hipMemPoolSetAttribute(hip_mem_pool, hipMemPoolAttrReleaseThreshold, &set_val));
}

Allocator::~Allocator()
{
    while (!tensor_ptr_set_.empty()) {
        Free(*(tensor_ptr_set_.begin()));
    }
    tensor_ptr_set_.clear();
}

char* Allocator::Malloc(const size_t size, bool is_zero, bool is_device)
{
    if (size == 0) {
        return nullptr;
    }

    char* tensor_ptr    = nullptr;
    int   set_device_id = 0;

    GetSetDevice(device_id_, &set_device_id);

    if (is_device) {
        LI_ENFORCE_HIP_SUCCESS(hipMalloc((void**)&tensor_ptr, size * sizeof(char)));
        // LI_ENFORCE_HIP_SUCCESS(hipMallocAsync((void**)&tensor_ptr, size, stream_));
    }
    else {
        LI_ENFORCE_HIP_SUCCESS(hipHostMalloc((void**)&tensor_ptr, size * sizeof(char), hipHostMallocDefault));
    }

    if (is_zero) {
        LI_ENFORCE_HIP_SUCCESS(hipMemsetAsync((void*)tensor_ptr, 0, size * sizeof(char), stream_));
    }

    GetSetDevice(set_device_id);
    VLOG(1) << "Malloc " << size * sizeof(char) << " bytes on device " << device_id_;

    if (tensor_ptr_set_.find(tensor_ptr) != tensor_ptr_set_.end()) {
        LOG(WARNING) << "tensor_ptr_set already has info of ptr at " << tensor_ptr << " on device " << device_id_;
    }

    tensor_ptr_set_.insert(tensor_ptr);

    return tensor_ptr;
}

void Allocator::Free(char* tensor_ptr, bool is_device)
{
    if (tensor_ptr == nullptr || tensor_ptr_set_.find(tensor_ptr) == tensor_ptr_set_.end()) {
        LOG(WARNING) << "tensor_ptr_set dones not have info of tensor ptr";
        return;
    }

    int set_device_id = 0;
    GetSetDevice(device_id_, &set_device_id);
    if (is_device) {
        // LI_ENFORCE_HIP_SUCCESS(hipFree((void**)&tensor_ptr));
        LI_ENFORCE_HIP_SUCCESS(hipFreeAsync((void**)&tensor_ptr, stream_));
        GpuStreamSync(stream_);
    }
    else {
        LI_ENFORCE_HIP_SUCCESS(hipHostFree((void**)&tensor_ptr));
    }

    GetSetDevice(set_device_id);
    tensor_ptr_set_.erase(tensor_ptr);

    tensor_ptr = nullptr;
}

void Allocator::SetStream(hipStream_t stream)
{
    stream_ = stream;
}

hipStream_t Allocator::GetStream() const
{
    return stream_;
}

void Allocator::MemSet(char* tensor_ptr, const int val, const size_t size)
{
    LI_ENFORCE_HIP_SUCCESS(hipMemsetAsync((void*)tensor_ptr, val, size, stream_));
}

}  // namespace lightinfer