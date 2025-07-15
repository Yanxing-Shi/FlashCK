#include "flashck/core/utils/debug_utils.h"

#include <algorithm>
#include <memory>
#include <sstream>

#include "flashck/core/utils/enforce.h"
#include "flashck/core/utils/macros.h"

namespace flashck {

constexpr int kBlockSize         = 256;
constexpr int kElementsPerThread = 4;

template<typename T>
__device__ __inline__ void
check_element(T val, const char* name, int idx, int* nan_cnt, int* pos_inf_cnt, int* neg_inf_cnt)
{
    const float fval = static_cast<float>(val);

    if (__isnanf(fval)) {
        atomicAdd(nan_cnt, 1);
        printf("[NaN] %s[%d] = %f\n", name, idx, fval);
    }
    else if (__isinff(fval)) {
        if (fval > 0) {
            atomicAdd(pos_inf_cnt, 1);
            printf("[+Inf] %s[%d]\n", name, idx);
        }
        else {
            atomicAdd(neg_inf_cnt, 1);
            printf("[-Inf] %s[%d]\n", name, idx);
        }
    }
}

template<typename T>
__global__ void tensor_validator_kernel(const T* __restrict__ tensor,
                                        int64_t     elem_cnt,
                                        const char* tensor_name,
                                        int* __restrict__ nan_cnt,
                                        int* __restrict__ pos_inf_cnt,
                                        int* __restrict__ neg_inf_cnt)
{
    const int tid = blockIdx.x * blockDim.x * kElementsPerThread + threadIdx.x;

    for (int i = 0; i < kElementsPerThread; ++i) {
        const int idx = tid + i * blockDim.x;
        if (idx >= elem_cnt)
            return;

        check_element(tensor[idx], tensor_name, idx, nan_cnt, pos_inf_cnt, neg_inf_cnt);
    }
}

template<typename T>
void ResultChecker(const T* tensor, int64_t elem_cnt, const std::string& tensor_name, hipStream_t stream)
{
    // Device memory allocation
    int* d_nan     = nullptr;
    int* d_pos_inf = nullptr;
    int* d_neg_inf = nullptr;

    HIP_ERROR_CHECK(hipMalloc(&d_nan, sizeof(int)));
    HIP_ERROR_CHECK(hipMalloc(&d_pos_inf, sizeof(int)));
    HIP_ERROR_CHECK(hipMalloc(&d_neg_inf, sizeof(int)));

    // Initialize counters
    HIP_ERROR_CHECK(hipMemsetAsync(d_nan, 0, sizeof(int), stream));
    HIP_ERROR_CHECK(hipMemsetAsync(d_pos_inf, 0, sizeof(int), stream));
    HIP_ERROR_CHECK(hipMemsetAsync(d_neg_inf, 0, sizeof(int), stream));

    // Kernel configuration
    const int block_dim = kBlockSize;
    const int grid_dim  = (elem_cnt + block_dim * kElementsPerThread - 1) / (block_dim * kElementsPerThread);

    // Kernel launch
    hipLaunchKernelGGL(tensor_validator_kernel<T>,
                       dim3(grid_dim),
                       dim3(block_dim),
                       0,
                       stream,
                       tensor,
                       elem_cnt,
                       tensor_name.c_str(),
                       d_nan,
                       d_pos_inf,
                       d_neg_inf);

    // Check for kernel launch errors
    HIP_ERROR_CHECK(hipGetLastError());

    // Copy results back
    int nan_count = 0, pos_inf_count = 0, neg_inf_count = 0;
    HIP_ERROR_CHECK(hipMemcpyAsync(&nan_count, d_nan, sizeof(int), hipMemcpyDeviceToHost, stream));
    HIP_ERROR_CHECK(hipMemcpyAsync(&pos_inf_count, d_pos_inf, sizeof(int), hipMemcpyDeviceToHost, stream));
    HIP_ERROR_CHECK(hipMemcpyAsync(&neg_inf_count, d_neg_inf, sizeof(int), hipMemcpyDeviceToHost, stream));
    HIP_ERROR_CHECK(hipStreamSynchronize(stream));

    // Cleanup
    HIP_ERROR_CHECK(hipFree(d_nan));
    HIP_ERROR_CHECK(hipFree(d_pos_inf));
    HIP_ERROR_CHECK(hipFree(d_neg_inf));

    // Error handling
    if (nan_count > 0 || pos_inf_count > 0 || neg_inf_count > 0) {
        throw std::runtime_error("Tensor validation failed: " + tensor_name + " NaN: " + std::to_string(nan_count)
                                 + " +Inf: " + std::to_string(pos_inf_count)
                                 + " -Inf: " + std::to_string(neg_inf_count));
    }
}

template<typename T>
void PrintToFile(const T* result, const int size, const char* file, hipStream_t stream, std::ios::openmode open_mode)
{
    // Parameter validation
    if (size < 0) {
        FC_THROW(InvalidArgument("Invalid size: {}", size));
    }
    if (!result && size != 0) {
        FC_THROW(InvalidArgument("Null pointer with non-zero size"));
    }

    LOG(INFO) << "[SAFE] Writing " << size << " elements to " << file;

    // RAII file management
    std::ofstream out_file(file, open_mode);
    if (!out_file.is_open()) {
        FC_THROW(Unavailable("Failed to open: {}", file));
    }

    // Smart pointer for host buffer
    auto host_buffer = std::make_unique<T[]>(size);

    // Asynchronous copy with stream synchronization
    HIP_ERROR_CHECK(hipMemcpyAsync(host_buffer.get(), result, sizeof(T) * size, hipMemcpyDeviceToHost, stream));
    HIP_ERROR_CHECK(hipStreamSynchronize(stream));

    // Buffered output generation
    std::ostringstream file_buffer;

    for (int i = 0; i < size; ++i) {
        file_buffer << static_cast<float>(host_buffer[i]) << '\n';
    }

    // Bulk write operation
    out_file << file_buffer.str();

    // Write verification
    if (out_file.fail()) {
        FC_THROW(Unavailable("Write failure to: {}", file));
    }
}

template<typename T>
void CheckMaxVal(const T* result, const int size, hipStream_t stream)
{
    // Parameter validation
    if (size <= 0) {
        FC_THROW(InvalidArgument("Invalid size: {}", size));
    }
    if (!result) {
        FC_THROW(InvalidArgument("Null device pointer"));
    }

    // RAII-managed host buffer
    auto host_buffer = std::make_unique<T[]>(size);

    // Asynchronous memory copy with stream synchronization
    HIP_ERROR_CHECK(hipMemcpyAsync(host_buffer.get(), result, sizeof(T) * size, hipMemcpyDeviceToHost, stream));
    HIP_ERROR_CHECK(hipStreamSynchronize(stream));

    // Find maximum value with proper type conversion
    float max_val = static_cast<float>(host_buffer[0]);
    for (int i = 1; i < size; ++i) {
        float current_val = static_cast<float>(host_buffer[i]);
        if (current_val > max_val) {
            max_val = current_val;
        }
    }

    // Diagnostic output
    LOG(INFO) << "[HIP] addr " << result << " Max: " << max_val;
}

template<typename T>
void PrintToScreen(const T* result, const int size, const std::string& name)
{
    LOG(INFO) << "------------------------------------------";

    if (result == nullptr) {
        LOG(WARNING) << "name: " << name << ", " << "value: " << "It is an nullptr, skip!";
        return;
    }

    T* tmp = reinterpret_cast<T*>(malloc(sizeof(T) * size));
    hipMemcpy(tmp, result, sizeof(T) * size, hipMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        LOG(INFO) << "name: " << name << ", " << "index: " << i << ", " << "value: " << static_cast<float>(tmp[i]);
    }

    free(tmp);
}

template<>
void PrintToScreen(const ushort* result, const int size, const std::string& name)
{
    LOG(INFO) << "------------------------------------------";

    if (result == nullptr) {
        LOG(WARNING) << "name: " << name << ", " << "value: " << "It is an nullptr, skip!";
        return;
    }

    ushort* tmp = reinterpret_cast<ushort*>(malloc(sizeof(ushort) * size));
    hipMemcpy(tmp, result, sizeof(ushort) * size, hipMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        LOG(INFO) << "name: " << name << ", " << "index: " << i << ", "
                  << "value: " << bf16_to_float_raw(bit_cast<uint16_t>(tmp[i]));
    }

    free(tmp);
}

template void PrintToScreen(const float* result, const int size, const std::string& name);
template void PrintToScreen(const _Float16* result, const int size, const std::string& name);
template void PrintToScreen(const int32_t* result, const int size, const std::string& name);
template void PrintToScreen(const int64_t* result, const int size, const std::string& name);

}  // namespace flashck