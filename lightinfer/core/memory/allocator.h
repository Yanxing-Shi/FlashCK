#pragma once

#include <hip/hip_runtime.h>
#include <unordered_set>

namespace lightinfer {
class Allocator {
public:
    Allocator(int device_id = 0);
    ~Allocator();

    char* Malloc(const size_t size, bool is_zero = true, bool is_device = true);
    void  Free(char* tensor_ptr, bool is_device = true);

    void        SetStream(hipStream_t stream);
    hipStream_t GetStream() const;

    void MemSet(char* tensor_ptr, const int val, const size_t size);

private:
    int                       device_id_;
    std::unordered_set<char*> tensor_ptr_set_;

    hipStream_t stream_ = nullptr;  // default stream
};
}  // namespace lightinfer