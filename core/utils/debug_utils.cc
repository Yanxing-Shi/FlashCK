#include "core/utils/debug_utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <tuple>

#include "core/utils/dtype.h"
#include "core/utils/enforce.h"
#include "core/utils/macros.h"

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
    // Input validation
    if (!tensor) {
        FC_THROW(InvalidArgument("Tensor pointer is null: {}", tensor_name));
    }
    if (elem_cnt <= 0) {
        FC_THROW(InvalidArgument("Invalid element count: {} for tensor: {}", elem_cnt, tensor_name));
    }
    if (!IsValidDevicePointer(tensor)) {
        FC_THROW(InvalidArgument("Invalid device pointer for tensor: {}", tensor_name));
    }

    // Device memory allocation with RAII
    int* d_nan     = nullptr;
    int* d_pos_inf = nullptr;
    int* d_neg_inf = nullptr;

    HIP_ERROR_CHECK(hipMalloc(&d_nan, sizeof(int)));
    HIP_ERROR_CHECK(hipMalloc(&d_pos_inf, sizeof(int)));
    HIP_ERROR_CHECK(hipMalloc(&d_neg_inf, sizeof(int)));

    // RAII cleanup helper
    auto cleanup = [&]() {
        if (d_nan)
            HIP_WARN_CHECK(hipFree(d_nan));
        if (d_pos_inf)
            HIP_WARN_CHECK(hipFree(d_pos_inf));
        if (d_neg_inf)
            HIP_WARN_CHECK(hipFree(d_neg_inf));
    };

    try {
        // Initialize counters
        HIP_ERROR_CHECK(hipMemsetAsync(d_nan, 0, sizeof(int), stream));
        HIP_ERROR_CHECK(hipMemsetAsync(d_pos_inf, 0, sizeof(int), stream));
        HIP_ERROR_CHECK(hipMemsetAsync(d_neg_inf, 0, sizeof(int), stream));

        // Optimized kernel configuration
        const int block_dim = kBlockSize;
        const int grid_dim  = (elem_cnt + block_dim * kElementsPerThread - 1) / (block_dim * kElementsPerThread);

        // Kernel launch with error checking
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

        HIP_ERROR_CHECK(hipGetLastError());

        // Copy results back with stream synchronization
        int nan_count = 0, pos_inf_count = 0, neg_inf_count = 0;
        HIP_ERROR_CHECK(hipMemcpyAsync(&nan_count, d_nan, sizeof(int), hipMemcpyDeviceToHost, stream));
        HIP_ERROR_CHECK(hipMemcpyAsync(&pos_inf_count, d_pos_inf, sizeof(int), hipMemcpyDeviceToHost, stream));
        HIP_ERROR_CHECK(hipMemcpyAsync(&neg_inf_count, d_neg_inf, sizeof(int), hipMemcpyDeviceToHost, stream));
        HIP_ERROR_CHECK(hipStreamSynchronize(stream));

        // Error reporting with detailed statistics
        if (nan_count > 0 || pos_inf_count > 0 || neg_inf_count > 0) {
            std::ostringstream error_msg;
            error_msg << "Tensor validation failed for '" << tensor_name << "' (type: " << GetTypeName<T>() << ")\n";
            error_msg << "  Elements: " << elem_cnt << "\n";
            error_msg << "  NaN count: " << nan_count << " (" << (100.0f * nan_count / elem_cnt) << "%)\n";
            error_msg << "  +Inf count: " << pos_inf_count << " (" << (100.0f * pos_inf_count / elem_cnt) << "%)\n";
            error_msg << "  -Inf count: " << neg_inf_count << " (" << (100.0f * neg_inf_count / elem_cnt) << "%)";

            LOG(ERROR) << error_msg.str();
            cleanup();
            throw std::runtime_error(error_msg.str());
        }

        VLOG(1) << "Tensor validation passed for '" << tensor_name << "' (" << elem_cnt << " elements)";
    }
    catch (...) {
        cleanup();
        throw;
    }

    cleanup();
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
    if (!file) {
        FC_THROW(InvalidArgument("Null file path"));
    }
    if (!IsValidDevicePointer(result)) {
        FC_THROW(InvalidArgument("Invalid device pointer"));
    }

    VLOG(1) << "[FILE] Writing " << size << " elements to " << file;

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

    // Buffered output generation with proper formatting
    std::ostringstream file_buffer;
    file_buffer << std::fixed << std::setprecision(6);

    for (int i = 0; i < size; ++i) {
        file_buffer << FormatValue(host_buffer[i]) << '\n';
    }

    // Bulk write operation
    out_file << file_buffer.str();

    // Write verification
    if (out_file.fail()) {
        FC_THROW(Unavailable("Write failure to: {}", file));
    }

    VLOG(1) << "[FILE] Successfully wrote " << size << " elements to " << file;
}

template<typename T>
void PrintToFile(
    const T* result, const int size, const std::string& filepath, hipStream_t stream, std::ios::openmode open_mode)
{
    PrintToFile(result, size, filepath.c_str(), stream, open_mode);
}

template<typename T>
void PrintToScreen(const T* result, const int size, const std::string& name, int max_elements)
{
    const std::string display_name = name.empty() ? "tensor" : name;

    VLOG(1) << "=== PrintToScreen: " << display_name << " ===";
    VLOG(1) << "Type: " << GetTypeName<T>();
    VLOG(1) << "Address: " << result;
    VLOG(1) << "Size: " << size << " elements";

    if (result == nullptr) {
        LOG(WARNING) << "Null pointer, skipping output";
        return;
    }

    if (size <= 0) {
        LOG(WARNING) << "Invalid size: " << size;
        return;
    }

    if (!IsValidDevicePointer(result)) {
        LOG(WARNING) << "Invalid device pointer";
        return;
    }

    // Limit output size for readability
    const int output_size = std::min(size, max_elements);

    // RAII-managed buffer
    auto host_buffer = std::make_unique<T[]>(output_size);

    // Use proper error checking
    HIP_ERROR_CHECK(hipMemcpy(host_buffer.get(), result, sizeof(T) * output_size, hipMemcpyDeviceToHost));

    // Formatted output with proper alignment
    VLOG(1) << "Contents (showing " << output_size << " of " << size << " elements):";

    for (int i = 0; i < output_size; ++i) {
        VLOG(1) << "  [" << std::setw(6) << i << "] = " << std::setw(12) << FormatValue(host_buffer[i]);
    }

    if (output_size < size) {
        VLOG(1) << "  ... (" << (size - output_size) << " more elements)";
    }

    // Show basic statistics if we have enough elements
    if (size > 1) {
        auto [min_val, max_val, mean, std_dev] = GetTensorStats(result, size, nullptr);
        VLOG(1) << "Statistics - Min: " << min_val << " Max: " << max_val << " Mean: " << mean << " Std: " << std_dev;
    }
}

// Specialized version for ushort (bfloat16)
template<>
void PrintToScreen<ushort>(const ushort* result, const int size, const std::string& name, int max_elements)
{
    const std::string display_name = name.empty() ? "tensor" : name;

    VLOG(1) << "=== PrintToScreen: " << display_name << " (bf16) ===";
    VLOG(1) << "Type: ushort (bfloat16)";
    VLOG(1) << "Address: " << result;
    VLOG(1) << "Size: " << size << " elements";

    if (result == nullptr) {
        LOG(WARNING) << "Null pointer, skipping output";
        return;
    }

    if (size <= 0) {
        LOG(WARNING) << "Invalid size: " << size;
        return;
    }

    if (!IsValidDevicePointer(result)) {
        LOG(WARNING) << "Invalid device pointer";
        return;
    }

    // Limit output size for readability
    const int output_size = std::min(size, max_elements);

    // RAII-managed buffer
    auto host_buffer = std::make_unique<ushort[]>(output_size);

    // Use proper error checking
    HIP_ERROR_CHECK(hipMemcpy(host_buffer.get(), result, sizeof(ushort) * output_size, hipMemcpyDeviceToHost));

    // Formatted output with bfloat16 conversion
    VLOG(1) << "Contents (showing " << output_size << " of " << size << " elements):";

    for (int i = 0; i < output_size; ++i) {
        float float_val = bhalf2float(host_buffer[i]);
        VLOG(1) << "  [" << std::setw(6) << i << "] = " << std::setw(12) << float_val << " (raw: 0x" << std::hex
                << host_buffer[i] << std::dec << ")";
    }

    if (output_size < size) {
        VLOG(1) << "  ... (" << (size - output_size) << " more elements)";
    }

    // Show basic statistics
    if (size > 1) {
        auto [min_val, max_val, mean, std_dev] = GetTensorStats(result, size, nullptr);
        VLOG(1) << "Statistics - Min: " << min_val << " Max: " << max_val << " Mean: " << mean << " Std: " << std_dev;
    }
}

// ==============================================================================
// Utility Functions Implementation
// ==============================================================================

template<typename T>
constexpr std::string_view GetTypeName()
{
    if constexpr (std::is_same_v<T, float>) {
        return "float";
    }
    else if constexpr (std::is_same_v<T, _Float16>) {
        return "_Float16";
    }
    else if constexpr (std::is_same_v<T, ushort>) {
        return "ushort(bf16)";
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        return "int32_t";
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        return "int64_t";
    }
    else {
        return "unknown";
    }
}

template<typename T>
std::string FormatValue(const T& value)
{
    if constexpr (std::is_same_v<T, ushort>) {
        return std::to_string(bhalf2float(value));
    }
    else if constexpr (std::is_same_v<T, _Float16>) {
        return std::to_string(half2float(static_cast<uint16_t>(value)));
    }
    else {
        return std::to_string(static_cast<float>(value));
    }
}

bool IsValidDevicePointer(const void* ptr)
{
    if (!ptr)
        return false;

    hipPointerAttribute_t attributes;
    hipError_t            err = hipPointerGetAttributes(&attributes, ptr);
    if (err != hipSuccess) {
        return false;
    }

    return attributes.type == hipMemoryTypeDevice;
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
    if (!IsValidDevicePointer(result)) {
        FC_THROW(InvalidArgument("Invalid device pointer"));
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
    VLOG(1) << "[HIP] addr " << result << " Max: " << max_val;
}

template<typename T>
void CheckMinVal(const T* result, const int size, hipStream_t stream)
{
    // Parameter validation
    if (size <= 0) {
        FC_THROW(InvalidArgument("Invalid size: {}", size));
    }
    if (!result) {
        FC_THROW(InvalidArgument("Null device pointer"));
    }
    if (!IsValidDevicePointer(result)) {
        FC_THROW(InvalidArgument("Invalid device pointer"));
    }

    // RAII-managed host buffer
    auto host_buffer = std::make_unique<T[]>(size);

    // Asynchronous memory copy with stream synchronization
    HIP_ERROR_CHECK(hipMemcpyAsync(host_buffer.get(), result, sizeof(T) * size, hipMemcpyDeviceToHost, stream));
    HIP_ERROR_CHECK(hipStreamSynchronize(stream));

    // Find minimum value with proper type conversion
    float min_val = static_cast<float>(host_buffer[0]);
    for (int i = 1; i < size; ++i) {
        float current_val = static_cast<float>(host_buffer[i]);
        if (current_val < min_val) {
            min_val = current_val;
        }
    }

    // Diagnostic output
    VLOG(1) << "[HIP] addr " << result << " Min: " << min_val;
}

template<typename T>
std::tuple<float, float, float, float> GetTensorStats(const T* result, const int size, hipStream_t stream)
{
    // Parameter validation
    if (size <= 0) {
        FC_THROW(InvalidArgument("Invalid size: {}", size));
    }
    if (!result) {
        FC_THROW(InvalidArgument("Null device pointer"));
    }
    if (!IsValidDevicePointer(result)) {
        FC_THROW(InvalidArgument("Invalid device pointer"));
    }

    // RAII-managed host buffer
    auto host_buffer = std::make_unique<T[]>(size);

    // Asynchronous memory copy with stream synchronization
    HIP_ERROR_CHECK(hipMemcpyAsync(host_buffer.get(), result, sizeof(T) * size, hipMemcpyDeviceToHost, stream));
    HIP_ERROR_CHECK(hipStreamSynchronize(stream));

    // Calculate statistics
    float  min_val = static_cast<float>(host_buffer[0]);
    float  max_val = static_cast<float>(host_buffer[0]);
    double sum     = 0.0;
    double sum_sq  = 0.0;

    for (int i = 0; i < size; ++i) {
        float val = static_cast<float>(host_buffer[i]);
        min_val   = std::min(min_val, val);
        max_val   = std::max(max_val, val);
        sum += val;
        sum_sq += val * val;
    }

    float mean     = static_cast<float>(sum / size);
    float variance = static_cast<float>(sum_sq / size - mean * mean);
    float std_dev  = std::sqrt(std::max(0.0f, variance));

    VLOG(1) << "[STATS] addr " << result << " Min: " << min_val << " Max: " << max_val << " Mean: " << mean
            << " Std: " << std_dev;

    return std::make_tuple(min_val, max_val, mean, std_dev);
}

template<typename T>
void InspectDeviceMemory(const T* ptr, int size, const std::string& name, hipStream_t stream)
{
    VLOG(1) << "=== Memory Inspection: " << (name.empty() ? "unnamed" : name) << " ===";

    if (!ptr) {
        LOG(WARNING) << "Null pointer";
        return;
    }

    if (!IsValidDevicePointer(ptr)) {
        LOG(WARNING) << "Invalid device pointer";
        return;
    }

    VLOG(1) << "Type: " << GetTypeName<T>();
    VLOG(1) << "Address: " << ptr;
    VLOG(1) << "Size: " << size << " elements (" << size * sizeof(T) << " bytes)";

    // Show first few elements
    const int preview_size = std::min(size, 10);
    auto      host_buffer  = std::make_unique<T[]>(preview_size);

    HIP_ERROR_CHECK(hipMemcpyAsync(host_buffer.get(), ptr, sizeof(T) * preview_size, hipMemcpyDeviceToHost, stream));
    HIP_ERROR_CHECK(hipStreamSynchronize(stream));

    VLOG(1) << "First " << preview_size << " elements:";
    for (int i = 0; i < preview_size; ++i) {
        VLOG(1) << "  [" << i << "] = " << FormatValue(host_buffer[i]);
    }

    // Get basic statistics
    if (size > 0) {
        auto [min_val, max_val, mean, std_dev] = GetTensorStats(ptr, size, stream);
        VLOG(1) << "Statistics - Min: " << min_val << " Max: " << max_val << " Mean: " << mean << " Std: " << std_dev;
    }
}

template<typename T>
int CompareTensors(
    const T* tensor1, const T* tensor2, int size, float tolerance, const std::string& name, hipStream_t stream)
{
    // Parameter validation
    if (size <= 0) {
        FC_THROW(InvalidArgument("Invalid size: {}", size));
    }
    if (!tensor1 || !tensor2) {
        FC_THROW(InvalidArgument("Null tensor pointer"));
    }
    if (!IsValidDevicePointer(tensor1) || !IsValidDevicePointer(tensor2)) {
        FC_THROW(InvalidArgument("Invalid device pointer"));
    }

    // RAII-managed host buffers
    auto buffer1 = std::make_unique<T[]>(size);
    auto buffer2 = std::make_unique<T[]>(size);

    // Asynchronous memory copies
    HIP_ERROR_CHECK(hipMemcpyAsync(buffer1.get(), tensor1, sizeof(T) * size, hipMemcpyDeviceToHost, stream));
    HIP_ERROR_CHECK(hipMemcpyAsync(buffer2.get(), tensor2, sizeof(T) * size, hipMemcpyDeviceToHost, stream));
    HIP_ERROR_CHECK(hipStreamSynchronize(stream));

    // Compare elements
    int   mismatch_count = 0;
    float max_diff       = 0.0f;
    int   max_diff_idx   = -1;

    for (int i = 0; i < size; ++i) {
        float val1 = static_cast<float>(buffer1[i]);
        float val2 = static_cast<float>(buffer2[i]);
        float diff = std::abs(val1 - val2);

        if (diff > tolerance) {
            mismatch_count++;
            if (diff > max_diff) {
                max_diff     = diff;
                max_diff_idx = i;
            }
        }
    }

    VLOG(1) << "=== Tensor Comparison: " << (name.empty() ? "unnamed" : name) << " ===";
    VLOG(1) << "Elements: " << size;
    VLOG(1) << "Tolerance: " << tolerance;
    VLOG(1) << "Mismatches: " << mismatch_count << " (" << (100.0f * mismatch_count / size) << "%)";

    if (mismatch_count > 0) {
        VLOG(1) << "Max difference: " << max_diff << " at index " << max_diff_idx;
        VLOG(1) << "  Tensor1[" << max_diff_idx << "] = " << FormatValue(buffer1[max_diff_idx]);
        VLOG(1) << "  Tensor2[" << max_diff_idx << "] = " << FormatValue(buffer2[max_diff_idx]);
    }

    return mismatch_count;
}

// ==============================================================================
// Explicit Template Instantiations
// ==============================================================================

// ResultChecker instantiations
template void
ResultChecker<float>(const float* tensor, int64_t elem_cnt, const std::string& tensor_name, hipStream_t stream);
template void
ResultChecker<_Float16>(const _Float16* tensor, int64_t elem_cnt, const std::string& tensor_name, hipStream_t stream);
template void
ResultChecker<ushort>(const ushort* tensor, int64_t elem_cnt, const std::string& tensor_name, hipStream_t stream);
template void
ResultChecker<int32_t>(const int32_t* tensor, int64_t elem_cnt, const std::string& tensor_name, hipStream_t stream);
template void
ResultChecker<int64_t>(const int64_t* tensor, int64_t elem_cnt, const std::string& tensor_name, hipStream_t stream);

// CheckMaxVal instantiations
template void CheckMaxVal<float>(const float* result, const int size, hipStream_t stream);
template void CheckMaxVal<_Float16>(const _Float16* result, const int size, hipStream_t stream);
template void CheckMaxVal<ushort>(const ushort* result, const int size, hipStream_t stream);
template void CheckMaxVal<int32_t>(const int32_t* result, const int size, hipStream_t stream);
template void CheckMaxVal<int64_t>(const int64_t* result, const int size, hipStream_t stream);

// CheckMinVal instantiations
template void CheckMinVal<float>(const float* result, const int size, hipStream_t stream);
template void CheckMinVal<_Float16>(const _Float16* result, const int size, hipStream_t stream);
template void CheckMinVal<ushort>(const ushort* result, const int size, hipStream_t stream);
template void CheckMinVal<int32_t>(const int32_t* result, const int size, hipStream_t stream);
template void CheckMinVal<int64_t>(const int64_t* result, const int size, hipStream_t stream);

// GetTensorStats instantiations
template std::tuple<float, float, float, float>
GetTensorStats<float>(const float* result, const int size, hipStream_t stream);
template std::tuple<float, float, float, float>
GetTensorStats<_Float16>(const _Float16* result, const int size, hipStream_t stream);
template std::tuple<float, float, float, float>
GetTensorStats<ushort>(const ushort* result, const int size, hipStream_t stream);
template std::tuple<float, float, float, float>
GetTensorStats<int32_t>(const int32_t* result, const int size, hipStream_t stream);
template std::tuple<float, float, float, float>
GetTensorStats<int64_t>(const int64_t* result, const int size, hipStream_t stream);

// InspectDeviceMemory instantiations
template void InspectDeviceMemory<float>(const float* ptr, int size, const std::string& name, hipStream_t stream);
template void InspectDeviceMemory<_Float16>(const _Float16* ptr, int size, const std::string& name, hipStream_t stream);
template void InspectDeviceMemory<ushort>(const ushort* ptr, int size, const std::string& name, hipStream_t stream);
template void InspectDeviceMemory<int32_t>(const int32_t* ptr, int size, const std::string& name, hipStream_t stream);
template void InspectDeviceMemory<int64_t>(const int64_t* ptr, int size, const std::string& name, hipStream_t stream);

// CompareTensors instantiations
template int CompareTensors<float>(
    const float* tensor1, const float* tensor2, int size, float tolerance, const std::string& name, hipStream_t stream);
template int CompareTensors<_Float16>(const _Float16*    tensor1,
                                      const _Float16*    tensor2,
                                      int                size,
                                      float              tolerance,
                                      const std::string& name,
                                      hipStream_t        stream);
template int CompareTensors<ushort>(const ushort*      tensor1,
                                    const ushort*      tensor2,
                                    int                size,
                                    float              tolerance,
                                    const std::string& name,
                                    hipStream_t        stream);
template int CompareTensors<int32_t>(const int32_t*     tensor1,
                                     const int32_t*     tensor2,
                                     int                size,
                                     float              tolerance,
                                     const std::string& name,
                                     hipStream_t        stream);
template int CompareTensors<int64_t>(const int64_t*     tensor1,
                                     const int64_t*     tensor2,
                                     int                size,
                                     float              tolerance,
                                     const std::string& name,
                                     hipStream_t        stream);

// PrintToScreen instantiations
template void PrintToScreen<float>(const float* result, const int size, const std::string& name, int max_elements);
template void
PrintToScreen<_Float16>(const _Float16* result, const int size, const std::string& name, int max_elements);
template void PrintToScreen<int32_t>(const int32_t* result, const int size, const std::string& name, int max_elements);
template void PrintToScreen<int64_t>(const int64_t* result, const int size, const std::string& name, int max_elements);
// Note: ushort specialization is already explicitly defined above

// PrintToFile (const char*) instantiations
template void PrintToFile<float>(
    const float* result, const int size, const char* file, hipStream_t stream, std::ios::openmode open_mode);
template void PrintToFile<_Float16>(
    const _Float16* result, const int size, const char* file, hipStream_t stream, std::ios::openmode open_mode);
template void PrintToFile<ushort>(
    const ushort* result, const int size, const char* file, hipStream_t stream, std::ios::openmode open_mode);
template void PrintToFile<int32_t>(
    const int32_t* result, const int size, const char* file, hipStream_t stream, std::ios::openmode open_mode);
template void PrintToFile<int64_t>(
    const int64_t* result, const int size, const char* file, hipStream_t stream, std::ios::openmode open_mode);

// PrintToFile (std::string) instantiations
template void PrintToFile<float>(
    const float* result, const int size, const std::string& filepath, hipStream_t stream, std::ios::openmode open_mode);
template void PrintToFile<_Float16>(const _Float16*    result,
                                    const int          size,
                                    const std::string& filepath,
                                    hipStream_t        stream,
                                    std::ios::openmode open_mode);
template void PrintToFile<ushort>(const ushort*      result,
                                  const int          size,
                                  const std::string& filepath,
                                  hipStream_t        stream,
                                  std::ios::openmode open_mode);
template void PrintToFile<int32_t>(const int32_t*     result,
                                   const int          size,
                                   const std::string& filepath,
                                   hipStream_t        stream,
                                   std::ios::openmode open_mode);
template void PrintToFile<int64_t>(const int64_t*     result,
                                   const int          size,
                                   const std::string& filepath,
                                   hipStream_t        stream,
                                   std::ios::openmode open_mode);

// Utility function instantiations
template std::string_view GetTypeName<float>();
template std::string_view GetTypeName<_Float16>();
template std::string_view GetTypeName<ushort>();
template std::string_view GetTypeName<int32_t>();
template std::string_view GetTypeName<int64_t>();

template std::string FormatValue<float>(const float& value);
template std::string FormatValue<_Float16>(const _Float16& value);
template std::string FormatValue<ushort>(const ushort& value);
template std::string FormatValue<int32_t>(const int32_t& value);
template std::string FormatValue<int64_t>(const int64_t& value);

}  // namespace flashck