#pragma once

#include <unordered_set>

#include <hip/hip_runtime.h>

namespace flashck {
/**
 * @class Allocator
 * @brief HIP memory manager with asynchronous operations and type tracking
 *
 * Manages both device and host pinned memory allocations with RAII semantics.
 * Tracks allocated pointers and their memory types for automatic cleanup.
 * @warning All interface methods are NOT thread-safe
 */
class Allocator {
public:
    /**
     * @brief Constructs allocator for specified GPU device
     * @param device_id Physical GPU device ID (0 <= id < GetGPUDeviceCount())
     * @throw InvalidArgument for invalid device ID
     */
    Allocator(int device_id = 0);

    /**
     * @brief Destroys allocator and releases all managed memory
     * @note Synchronizes bound stream before releasing device memory
     * @warning Any pending asynchronous operations must complete before destruction
     */
    ~Allocator()
    {
        if (stream_)
            GpuStreamSync(stream_);
        for (auto& [ptr, _] : ptr_info_)
            FreeImpl(ptr);
        ptr_info_.clear();
    }

    /**
     * @brief Allocates memory with optional initialization
     * @param size Requested memory size in bytes (0 returns nullptr)
     * @param is_zero Whether to initialize memory with zeros asynchronously
     * @param is_device Allocate device memory (true) or host pinned memory (false)
     * @return Pointer to allocated memory, nullptr if size==0
     * @throws hipError_t Throws if HIP allocation fails
     * @note Zero initialization uses the bound HIP stream
     */
    char* Malloc(size_t size, bool is_zero = false, bool is_device = true)
    {
        if (!size)
            return nullptr;
        LI_ENFORCE(size <= kMaxAllocation, "Exceeds max allocation size");

        ScopedContext guard(device_id_);
        char*         ptr = AllocMemory(size, is_device);
        ptr_info_[ptr]    = is_device;

        if (is_zero) {
            HIP_ERROR_CHECK(hipMemsetAsync(ptr, 0, size, stream_));
        }
        return ptr;
    }

    /**
     * @brief Releases allocated memory
     * @param ptr Memory pointer to free (no-op if nullptr)
     * @note For device memory: queues async free and syncs stream
     * @note For host memory: immediate synchronous free
     */
    void Free(char* ptr)
    {
        if (!ptr || !ptr_info_.count(ptr))
            return;

        ScopedContext guard(device_id_);
        FreeImpl(ptr);
        ptr_info_.erase(ptr);
    }

    /**
     * @brief Sets HIP stream for asynchronous operations
     * @param stream Valid HIP stream handle (can be nullptr)
     * @warning Stream must remain valid during allocator's lifetime
     */
    void SetStream(hipStream_t stream)
    {
        stream_ = stream;
    }

    /**
     * @brief Gets current HIP stream
     * @return Bound HIP stream (may be nullptr)
     */
    hipStream_t GetStream() const
    {
        return stream_;
    }

private:
    /**
     * @brief Configures peer access permissions across devices
     * @param mem_pool Target memory pool to configure
     * @param device_count Total available GPU devices
     */
    void ConfigurePeerAccess(hipMemPool_t mem_pool, int device_count) const;

    /**
     * @brief Enables memory pool access for specific peer device
     * @param mem_pool Memory pool to modify
     * @param peer_id Peer device ID to grant access
     */
    void SetMemPoolAccess(hipMemPool_t mem_pool, int peer_id) const;

    /**
     * @brief Sets memory pool retention threshold
     * @param mem_pool Memory pool to configure
     * @param threshold Release threshold value
     */
    void SetMemoryPoolThreshold(hipMemPool_t mem_pool, uint64_t threshold) const;

    /**
     * @brief RAII context manager for HIP device switching
     *
     * Switches to target device on construction,
     * restores original device on destruction.
     */
    struct ScopedContext {
        int original_;  ///< Original device ID

        /**
         * @brief Construct context guard switching to target device
         * @param dev Target HIP device ID
         */
        explicit ScopedContext(int dev)
        {
            hipGetDevice(&original_);
            hipSetDevice(dev);
        }

        ~ScopedContext()
        {
            hipSetDevice(original_);
        }  ///< Restore original device
    };

    /**
     * @brief Internal memory allocator implementation
     * @param size Allocation size in bytes
     * @param is_device Memory type flag
     * @return Raw allocated pointer
     * @throws hipError_t Propagates HIP errors
     */
    char* AllocMemory(size_t size, bool is_device)
    {
        char* p;
        auto  err = is_device ? hipMalloc(&p, size) : hipHostMalloc(&p, size, hipHostMallocDefault);
        HIP_ERROR_CHECK(err);
        return p;
    }

    /**
     * @brief Internal memory deallocator
     * @param ptr Pointer to free
     * @note For device memory: uses async free with stream synchronization
     */
    void FreeImpl(char* ptr)
    {
        if (ptr_info_[ptr]) {
            HIP_ERROR_CHECK(hipFreeAsync(ptr, stream_));
            if (stream_)
                GpuStreamSync(stream_);
        }
        else {
            HIP_ERROR_CHECK(hipHostFree(ptr));
        }
    }

    std::unordered_map<char*, bool> ptr_info_;   ///< Map of [pointer -> is_device]
    hipStream_t                     stream_;     ///< Bound HIP stream for async ops (nullable)
    const int                       device_id_;  ///< Target HIP device ID
};
}  // namespace flashck