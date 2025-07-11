#pragma once

#include <unordered_map>

#include <hip/hip_runtime.h>

namespace flashck {
class Allocator {
public:
    Allocator(int device_id = 0);

    ~Allocator();

    char* Malloc(const size_t size, int init_type = 0, bool is_device = true);

    void Free(char* tensor_ptr, bool is_device = true);

private:
    struct ScopedContext {
        int original_;

        explicit ScopedContext(int dev)
        {
            hipGetDevice(&original_);
            hipSetDevice(dev);
        }

        ~ScopedContext()
        {
            hipSetDevice(original_);
        }
    };

    void DeviceMalloc(char*& tensor_ptr, size_t size, int init_type);
    void GetSetDevice(int device_id);
    void GpuStreamSync(hipStream_t stream);

    std::unordered_map<char*, bool> ptr_info_;   ///< Map of [pointer -> is_device]
    hipStream_t                     stream_;     ///< HIP stream for async operations
    const int                       device_id_;  ///< Target HIP device ID
};
}  // namespace flashck