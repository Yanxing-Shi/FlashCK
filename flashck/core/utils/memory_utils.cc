#include "flashck/core/utils/memory_utils.h"

namespace flashck {

/**
 * @brief Parallel random initialization kernel for device memory
 * @tparam T Data type of elements (arithmetic type)
 */
template<typename T>
__global__ void UniformRandomInitKernel(T* data, size_t n, T min_val, T max_val)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Xorshift128+ algorithm
        unsigned seed = tid ^ clock64();
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;

        if constexpr (std::is_floating_point_v<T>) {
            constexpr T scale = 1.0 / UINT_MAX;
            data[tid]         = min_val + (max_val - min_val) * (seed * scale);
        }
        else {
            data[tid] = min_val + (seed % (max_val - min_val + 1));
        }
    }
}

/**
 * @brief Launches random initialization kernel with optimal configuration
 */
template<typename T>
void LaunchRandomInitKernel(T* ptr, size_t count, T min_val, T max_val)
{
    constexpr int kBlockSize = 256;
    const int     grid_size  = (count + kBlockSize - 1) / kBlockSize;

    hipLaunchKernelGGL(
        UniformRandomInitKernel<T>, dim3(grid_size), dim3(kBlockSize), 0, 0, ptr, count, min_val, max_val);

    HIP_ERROR_CHECK(hipGetLastError());
    HIP_ERROR_CHECK(hipStreamSynchronize(0));
}

/**
 * @brief Optimized device memory allocator with initialization capabilities
 */
template<typename T>
void DeviceMalloc(T**    ptr,
                  size_t count,
                  int    init_type,
                  typename std::conditional_t<IsSupportedType<T>::value && std::is_floating_point_v<T>, T, int> min_val,
                  typename std::conditional_t<IsSupportedType<T>::value && std::is_floating_point_v<T>, T, int> max_val)
{
    static_assert(IsSupportedType<T>::value, "T must be arithmetic type");

    // Clean output pointer first
    if (ptr)
        *ptr = nullptr;

    // Handle zero-size allocation
    if (count == 0)
        return;

    // Validate arguments
    if (!ptr)
        throw std::invalid_argument("Output pointer cannot be null");
    if (min_val > max_val)
        throw std::invalid_argument("min_val must be <= max_val");

    // HIP memory allocation
    T* raw_ptr = nullptr;
    HIP_ERROR_CHECK(hipMalloc(&raw_ptr, count * sizeof(T)));
    HipUniquePtr<T> device_ptr(raw_ptr);

    // Handle initialization types
    switch (init_type) {
        case 0:  // No initialization
            break;

        case 1:  // Device-side parallel random init
            LaunchRandomInitKernel(device_ptr.get(), count, min_val, max_val);
            break;

        case 2:  // HIP-accelerated zero initialization
            HIP_ERROR_CHECK(hipMemset(device_ptr.get(), 0, count * sizeof(T)));
            break;

        default:
            throw std::invalid_argument("Invalid init_type (0-2 supported)");
    }

    // Transfer ownership to caller
    *ptr = device_ptr.release();
}

template void DeviceMalloc(
    float**                                                                                                   ptr,
    size_t                                                                                                    count,
    int                                                                                                       init_type,
    typename std::conditional_t<IsSupportedType<float>::value && std::is_floating_point_v<float>, float, int> min_val,
    typename std::conditional_t<IsSupportedType<float>::value && std::is_floating_point_v<float>, float, int> max_val);

/**
 * @brief Safely deallocates HIP device memory and nullifies the pointer.
 *
 * @tparam T Type of the elements in the allocated memory. Must match the type used in `hipMalloc`.
 * @param[in,out] ptr Reference to a pointer pointing to HIP device memory.
 *                   After deallocation, the pointer is set to `nullptr`.
 *
 * @note
 * - Idempotent: Safe to call multiple times on the same pointer.
 * - Memory Safety: Ensures no double-free if `ptr` is already `nullptr`.
 * - HIP Integration: Requires the pointer to be allocated via HIP APIs (e.g., `hipMalloc`).
 *
 * @par Example:
 * @code
 * float* d_data = nullptr;
 * HIP_ERROR_CHECK(hipMalloc(&d_data, 1024 * sizeof(float)));  // Allocate
 * DeviceFree(d_data);  // Deallocate; d_data becomes nullptr
 * @endcode
 */
template<typename T>
void DeviceFree(T*& ptr)
{
    if (ptr != nullptr) {
        HIP_ERROR_CHECK(hipFree(ptr));  // Error handling via macro
        ptr = nullptr;
    }
}

template void DeviceFree(float*& ptr);
template void DeviceFree(_Float16*& ptr);
template void DeviceFree(ushort*& ptr);

/**
 * @brief Fills HIP device memory with a specified value asynchronously.
 *
 * @tparam T Data type of the device memory and the fill value.
 * @param[out] devptr Pointer to the device memory to be filled.
 * @param size Number of elements to fill.
 * @param value The value to fill into the device memory.
 * @param stream HIP stream for asynchronous operation (defaults to the null stream).
 *
 * @note
 * - Uses RAII to manage temporary host memory automatically.
 * - The `hipMemcpyAsync` operation requires the host memory (temporary buffer) to remain valid
 *   until the copy completes. If stream synchronization is not guaranteed, consider using
 *   HIP pinned memory instead of the default `std::vector`.
 * - For large datasets, prefer HIP APIs like `hipMemsetAsync` or kernel-based filling for better performance.
 *
 * @par Example:
 * @code
 * float* d_data;
 * HIP_ERROR_CHECK(hipMalloc(&d_data, 1024 * sizeof(float)));
 * DeviceFill(d_data, 1024, 3.14f);  // Fill with 3.14f asynchronously
 * HIP_ERROR_CHECK(hipStreamSynchronize(0));  // Optional synchronization
 * @endcode
 */
template<typename T>
void DeviceFill(T* devptr, size_t size, T value, hipStream_t stream)
{
    // RAII-managed host buffer initialized with the fill value
    std::vector<T> host_buffer(size, value);

    // Asynchronously copy data to the device
    HIP_ERROR_CHECK(hipMemcpyAsync(devptr, host_buffer.data(), sizeof(T) * size, hipMemcpyHostToDevice, stream));

    // Note: If stream synchronization is not enforced, the host buffer
    //       must outlive the asynchronous operation. For safety, uncomment:
    // HIP_ERROR_CHECK(hipStreamSynchronize(stream));
}

template void DeviceFill(float* devptr, size_t size, float value, hipStream_t stream);
template void DeviceFill(_Float16* devptr, size_t size, _Float16 value, hipStream_t stream);
template void DeviceFill(ushort* devptr, size_t size, ushort value, hipStream_t stream);

/**
 * @brief Asynchronously copies data from HIP device memory to host memory.
 *
 * @tparam T Data type of the source and destination pointers.
 * @param[out] tgt Pointer to host memory where data will be copied.
 * @param[in] src Pointer to device memory from which data will be copied.
 * @param size Number of elements to copy.
 * @param stream HIP stream for asynchronous operation (default: nullptr).
 *
 * @note
 * - The target host memory (`tgt`) should be pinned (page-locked) for optimal
 *   asynchronous performance. Use `hipHostMalloc` for allocation.
 * - The source device memory (`src`) must be valid and accessible.
 * - Ensure stream synchronization (`hipStreamSynchronize`) or use completion
 *   callbacks to guarantee data is ready before accessing `tgt`.
 */
template<typename T>
void HipD2HCpyAsync(T* tgt, const T* src, size_t size, hipStream_t stream)
{
    // Asynchronous device-to-host copy
    HIP_ERROR_CHECK(hipMemcpyAsync(tgt,                    // Host destination
                                   src,                    // Device source
                                   sizeof(T) * size,       // Byte size
                                   hipMemcpyDeviceToHost,  // Explicit direction
                                   stream                  // HIP stream
                                   ));
}

template void HipD2HCpyAsync(float* tgt, const float* src, size_t size, hipStream_t stream);
template void HipD2HCpyAsync(_Float16* tgt, const _Float16* src, size_t size, hipStream_t stream);
template void HipD2HCpyAsync(ushort* tgt, const ushort* src, size_t size, hipStream_t stream);

/**
 * @brief Asynchronously copies data from host memory to HIP device memory.
 *
 * @tparam T Data type of the source and destination pointers.
 * @param[out] tgt Pointer to device memory where data will be copied.
 * @param[in] src Pointer to host memory from which data will be copied.
 * @param size Number of elements to copy.
 * @param stream HIP stream for asynchronous operation (default: nullptr).
 *
 * @note
 * - The source host memory (`src`) should be pinned (page-locked) for optimal
 *   asynchronous performance. Use `hipHostMalloc` for allocation.
 * - The target device memory (`tgt`) must be properly allocated.
 * - For overlapping compute and data transfer, use non-default streams and ensure
 *   proper synchronization via `hipStreamSynchronize` or event-based callbacks.
 */
template<typename T>
void HipH2DCpyAsync(T* tgt, const T* src, size_t size, hipStream_t stream)
{
    // Asynchronous host-to-device copy
    HIP_ERROR_CHECK(hipMemcpyAsync(tgt,                    // Device destination
                                   src,                    // Host source
                                   sizeof(T) * size,       // Byte size
                                   hipMemcpyHostToDevice,  // Explicit direction
                                   stream                  // HIP stream
                                   ));
}

template void HipH2DCpyAsync(float* tgt, const float* src, size_t size, hipStream_t stream);
template void HipH2DCpyAsync(_Float16* tgt, const _Float16* src, size_t size, hipStream_t stream);
template void HipH2DCpyAsync(ushort* tgt, const ushort* src, size_t size, hipStream_t stream);

/**
 * @brief Asynchronously copies data between HIP device memory buffers.
 *
 * @tparam T Data type of the source and destination pointers.
 * @param[out] tgt Pointer to device memory destination.
 * @param[in] src Pointer to device memory source.
 * @param size Number of elements to copy.
 * @param stream HIP stream for asynchronous operation (default: nullptr).
 *
 * @note
 * - Both source (`src`) and target (`tgt`) must reside on the same device context.
 * - For cross-device copies (peer-to-peer), ensure peer access is enabled via `hipDeviceEnablePeerAccess`.
 * - Zero-copy operations have minimal overhead but still require stream synchronization.
 */
template<typename T>
void HipD2DCpyAsync(T* tgt, const T* src, size_t size, hipStream_t stream)
{
    // Asynchronous device-to-device copy
    HIP_ERROR_CHECK(hipMemcpyAsync(tgt,                      // Device destination
                                   src,                      // Device source
                                   sizeof(T) * size,         // Byte size
                                   hipMemcpyDeviceToDevice,  // Explicit direction
                                   stream                    // HIP stream
                                   ));
}

template void HipD2DCpyAsync(float* tgt, const float* src, size_t size, hipStream_t stream);
template void HipD2DCpyAsync(_Float16* tgt, const _Float16* src, size_t size, hipStream_t stream);
template void HipD2DCpyAsync(ushort* tgt, const ushort* src, size_t size, hipStream_t stream);

}  // namespace flashck