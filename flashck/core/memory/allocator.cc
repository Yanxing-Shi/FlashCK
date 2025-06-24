#include "flashck/core/memory/allocator.h"

#include "flashck/core/utils/enforce.h"
#include "flashck/core/utils/log.h"
#include "flashck/core/utils/rocm_info.h"

namespace flashck {

Allocator::Allocator(int device_id): device_id_(device_id)
{
    constexpr uint64_t kMaxReleaseThreshold = UINT64_MAX;
    const int          device_count         = GetGPUDeviceCount();

    if (device_id < 0 || device_id >= device_count) {
        LI_THROW(InvalidArgument("Invalid device id: {}, total devices: {}", device_id, device_count));
    }

    hipMemPool_t mem_pool = nullptr;
    HIP_ERROR_CHECK(hipDeviceGetDefaultMemPool(&mem_pool, device_id_));

    ConfigurePeerAccess(mem_pool, device_count);
    SetMemoryPoolThreshold(mem_pool, kMaxReleaseThreshold);
}

void Allocator::ConfigurePeerAccess(hipMemPool_t mem_pool, int device_count) const
{
    for (int peer_id = 0; peer_id < device_count; ++peer_id) {
        if (peer_id == device_id_)
            continue;

        int can_access = 0;
        HIP_ERROR_CHECK(hipDeviceCanAccessPeer(&can_access, device_id_, peer_id));

        if (!can_access) {
            LOG_FIRST_N(WARNING, 3) << "P2P access disabled from device " << device_id_ << " to device " << peer_id;
            continue;
        }

        SetMemPoolAccess(mem_pool, peer_id);
    }
}

void Allocator::SetMemPoolAccess(hipMemPool_t mem_pool, int peer_id) const
{
    hipMemAccessDesc desc{};
    desc.location.type = hipMemLocationTypeDevice;
    desc.location.id   = peer_id;
    desc.flags         = hipMemAccessFlagsProtReadWrite;

    hipMemPool_t peer_mem_pool = nullptr;
    HIP_ERROR_CHECK(hipDeviceGetMemPool(&peer_mem_pool, peer_id));

    HIP_ERROR_CHECK(hipMemPoolSetAccess(peer_mem_pool, &desc, 1));
}

void Allocator::SetMemoryPoolThreshold(hipMemPool_t mem_pool, uint64_t threshold) const
{
    HIP_ERROR_CHECK(hipMemPoolSetAttribute(mem_pool, hipMemPoolAttrReleaseThreshold, &threshold));
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
        HIP_ERROR_CHECK(hipMalloc((void**)&tensor_ptr, size * sizeof(char)));
        // HIP_ERROR_CHECK(hipMallocAsync((void**)&tensor_ptr, size, stream_));
    }
    else {
        HIP_ERROR_CHECK(hipHostMalloc((void**)&tensor_ptr, size * sizeof(char), hipHostMallocDefault));
    }

    if (is_zero) {
        HIP_ERROR_CHECK(hipMemsetAsync((void*)tensor_ptr, 0, size * sizeof(char), stream_));
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
        // HIP_ERROR_CHECK(hipFree((void**)&tensor_ptr));
        HIP_ERROR_CHECK(hipFreeAsync((void**)&tensor_ptr, stream_));
        GpuStreamSync(stream_);
    }
    else {
        HIP_ERROR_CHECK(hipHostFree((void**)&tensor_ptr));
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
    HIP_ERROR_CHECK(hipMemsetAsync((void*)tensor_ptr, val, size, stream_));
}

}  // namespace flashck