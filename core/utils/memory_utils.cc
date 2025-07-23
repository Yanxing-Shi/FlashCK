#include "core/utils/memory_utils.h"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace flashck {

// ==============================================================================
// GPU Kernel Implementations
// ==============================================================================

/*!
 * @brief High-performance GPU kernel for uniform random initialization
 * @tparam T Element type
 * @param data Output array
 * @param n Number of elements
 * @param min_val Minimum value
 * @param max_val Maximum value
 * @note Uses optimized Xorshift128+ algorithm for random generation
 */
template<typename T>
__global__ void UniformRandomInitKernel(T* data, size_t n, T min_val, T max_val)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // High-quality Xorshift128+ algorithm for random number generation
        uint64_t seed = tid ^ clock64();
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;

        if constexpr (std::is_floating_point_v<T>) {
            // For floating point: normalize to [0,1] then scale to [min_val, max_val]
            constexpr T scale = T(1.0) / T(UINT64_MAX);
            data[tid]         = min_val + (max_val - min_val) * T(seed * scale);
        }
        else {
            // For integers: use modulo operation
            data[tid] = min_val + T(seed % uint64_t(max_val - min_val + 1));
        }
    }
}

/*!
 * @brief Launch random initialization kernel with optimal grid configuration
 * @tparam T Element type
 * @param ptr Device pointer to initialize
 * @param count Number of elements
 * @param min_val Minimum value
 * @param max_val Maximum value
 * @note Automatically determines optimal block and grid sizes
 */
template<typename T>
void LaunchRandomInitKernel(T* ptr, size_t count, T min_val, T max_val)
{
    if (count == 0)
        return;

    // Optimize block size based on element type and GPU architecture
    constexpr int kBlockSize = 256;  // Optimal for most modern GPUs
    const int     grid_size  = (count + kBlockSize - 1) / kBlockSize;

    // Launch kernel with error checking
    hipLaunchKernelGGL(UniformRandomInitKernel<T>,
                       dim3(grid_size),
                       dim3(kBlockSize),
                       0,
                       nullptr,  // Use default stream
                       ptr,
                       count,
                       min_val,
                       max_val);

    // Check for kernel launch errors
    HIP_ERROR_CHECK(hipGetLastError());

    // Synchronize to ensure completion
    HIP_ERROR_CHECK(hipDeviceSynchronize());
}

// ==============================================================================
// Memory Allocation Implementation
// ==============================================================================

template<typename T>
void DeviceMalloc(T**                                                                                        ptr,
                  size_t                                                                                     count,
                  InitType                                                                                   init_type,
                  typename std::conditional_t<is_supported_type_v<T> && std::is_floating_point_v<T>, T, int> min_val,
                  typename std::conditional_t<is_supported_type_v<T> && std::is_floating_point_v<T>, T, int> max_val)
{
    // Compile-time type checking
    static_assert(is_supported_type_v<T>, "T must be a supported type (float, _Float16, ushort)");

    // Validate input parameters
    if (!ptr) {
        throw std::invalid_argument("DeviceMalloc: Output pointer cannot be null");
    }

    // Clean output pointer first
    *ptr = nullptr;

    // Handle zero-size allocation
    if (count == 0) {
        return;
    }

    // Validate value range
    if (min_val > max_val) {
        throw std::invalid_argument("DeviceMalloc: min_val must be <= max_val");
    }

    // Calculate allocation size with overflow check
    const size_t byte_size = count * sizeof(T);
    if (byte_size / sizeof(T) != count) {
        throw std::overflow_error("DeviceMalloc: Allocation size overflow");
    }

    // Allocate device memory
    T* raw_ptr = nullptr;
    HIP_ERROR_CHECK(hipMalloc(&raw_ptr, byte_size));

    // Use RAII for exception safety
    HipUniquePtr<T> device_ptr(raw_ptr);

    // Handle different initialization types
    switch (init_type) {
        case InitType::None:
            // No initialization - fastest option
            break;

        case InitType::Random:
            // Device-side parallel random initialization
            LaunchRandomInitKernel(device_ptr.get(), count, static_cast<T>(min_val), static_cast<T>(max_val));
            break;

        case InitType::Zero:
            // HIP-accelerated zero initialization
            HIP_ERROR_CHECK(hipMemset(device_ptr.get(), 0, byte_size));
            break;

        default:
            throw std::invalid_argument("DeviceMalloc: Invalid initialization type");
    }

    // Transfer ownership to caller
    *ptr = device_ptr.release();
}

// ==============================================================================
// Memory Deallocation Implementation
// ==============================================================================

template<typename T>
void DeviceFree(T*& ptr)
{
    if (ptr != nullptr) {
        // Free device memory with error checking
        HIP_ERROR_CHECK(hipFree(ptr));
        ptr = nullptr;  // Set to nullptr for safety
    }
}

// ==============================================================================
// Memory Fill Implementation
// ==============================================================================

/*!
 * @brief Optimized GPU kernel for device memory fill
 * @tparam T Element type
 * @param devptr Device pointer to fill
 * @param size Number of elements
 * @param value Fill value
 */
template<typename T>
__global__ void DeviceFillKernel(T* devptr, size_t size, T value)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        devptr[tid] = value;
    }
}

template<typename T>
void DeviceFill(T* devptr, size_t size, T value, hipStream_t stream)
{
    if (!devptr || size == 0) {
        return;  // Nothing to fill
    }

    // For small sizes, use host-to-device copy
    if (size <= 1024) {
        // Use host buffer for small fills (more efficient)
        std::vector<T> host_buffer(size, value);
        HIP_ERROR_CHECK(hipMemcpyAsync(devptr, host_buffer.data(), sizeof(T) * size, hipMemcpyHostToDevice, stream));
    }
    else {
        // Use GPU kernel for large fills (better parallelism)
        constexpr int kBlockSize = 256;
        const int     grid_size  = (size + kBlockSize - 1) / kBlockSize;

        hipLaunchKernelGGL(DeviceFillKernel<T>, dim3(grid_size), dim3(kBlockSize), 0, stream, devptr, size, value);

        HIP_ERROR_CHECK(hipGetLastError());
    }
}

// ==============================================================================
// Asynchronous Memory Copy Implementations
// ==============================================================================

template<typename T>
void HipD2HCpyAsync(T* tgt, const T* src, size_t size, hipStream_t stream)
{
    if (!tgt || !src || size == 0) {
        return;  // Nothing to copy
    }

    // Asynchronous device-to-host copy with explicit direction
    HIP_ERROR_CHECK(hipMemcpyAsync(tgt,                    // Host destination
                                   src,                    // Device source
                                   sizeof(T) * size,       // Byte size
                                   hipMemcpyDeviceToHost,  // Explicit direction
                                   stream                  // HIP stream
                                   ));
}

template<typename T>
void HipH2DCpyAsync(T* tgt, const T* src, size_t size, hipStream_t stream)
{
    if (!tgt || !src || size == 0) {
        return;  // Nothing to copy
    }

    // Asynchronous host-to-device copy with explicit direction
    HIP_ERROR_CHECK(hipMemcpyAsync(tgt,                    // Device destination
                                   src,                    // Host source
                                   sizeof(T) * size,       // Byte size
                                   hipMemcpyHostToDevice,  // Explicit direction
                                   stream                  // HIP stream
                                   ));
}

template<typename T>
void HipD2DCpyAsync(T* tgt, const T* src, size_t size, hipStream_t stream)
{
    if (!tgt || !src || size == 0) {
        return;  // Nothing to copy
    }

    // Asynchronous device-to-device copy with explicit direction
    HIP_ERROR_CHECK(hipMemcpyAsync(tgt,                      // Device destination
                                   src,                      // Device source
                                   sizeof(T) * size,         // Byte size
                                   hipMemcpyDeviceToDevice,  // Explicit direction
                                   stream                    // HIP stream
                                   ));
}

// ==============================================================================
// Explicit Template Instantiations
// ==============================================================================

// DeviceMalloc instantiations
template void DeviceMalloc<float>(float** ptr, size_t count, InitType init_type, float min_val, float max_val);

template void DeviceMalloc<_Float16>(_Float16** ptr, size_t count, InitType init_type, int min_val, int max_val);

template void DeviceMalloc<ushort>(ushort** ptr, size_t count, InitType init_type, int min_val, int max_val);

// DeviceFree instantiations
template void DeviceFree<float>(float*& ptr);
template void DeviceFree<_Float16>(_Float16*& ptr);
template void DeviceFree<ushort>(ushort*& ptr);

// DeviceFill instantiations
template void DeviceFill<float>(float* devptr, size_t size, float value, hipStream_t stream);
template void DeviceFill<_Float16>(_Float16* devptr, size_t size, _Float16 value, hipStream_t stream);
template void DeviceFill<ushort>(ushort* devptr, size_t size, ushort value, hipStream_t stream);

// HipD2HCpyAsync instantiations
template void HipD2HCpyAsync<float>(float* tgt, const float* src, size_t size, hipStream_t stream);
template void HipD2HCpyAsync<_Float16>(_Float16* tgt, const _Float16* src, size_t size, hipStream_t stream);
template void HipD2HCpyAsync<ushort>(ushort* tgt, const ushort* src, size_t size, hipStream_t stream);

// HipH2DCpyAsync instantiations
template void HipH2DCpyAsync<float>(float* tgt, const float* src, size_t size, hipStream_t stream);
template void HipH2DCpyAsync<_Float16>(_Float16* tgt, const _Float16* src, size_t size, hipStream_t stream);
template void HipH2DCpyAsync<ushort>(ushort* tgt, const ushort* src, size_t size, hipStream_t stream);

// HipD2DCpyAsync instantiations
template void HipD2DCpyAsync<float>(float* tgt, const float* src, size_t size, hipStream_t stream);
template void HipD2DCpyAsync<_Float16>(_Float16* tgt, const _Float16* src, size_t size, hipStream_t stream);
template void HipD2DCpyAsync<ushort>(ushort* tgt, const ushort* src, size_t size, hipStream_t stream);

}  // namespace flashck