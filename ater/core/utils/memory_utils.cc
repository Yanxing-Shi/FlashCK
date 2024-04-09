#include "ater/core/utils/memory_utils.h"

#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>
#include <random>

#include "ater/core/utils/enforce.h"
#include "ater/core/utils/log.h"

namespace ater {

/*-----------------------------device memory function ----------------------------*/
template<typename T>
void DeviceMalloc(T** ptr, size_t size, bool is_random_initialize)
{
    ATER_ENFORCE_EQ(size >= ((size_t)0), true, Unavailable("Ask deviceMalloc size {}< 0 is invalid.", size));
    ATER_ENFORCE_HIP_SUCCESS(hipMalloc((void**)(ptr), sizeof(T) * size));
    if (is_random_initialize) {
        HipRandomUniform(*ptr, size);
    }
}

template void DeviceMalloc(float** ptr, size_t size, bool is_random_initialize);
template void DeviceMalloc(_Float16** ptr, size_t size, bool is_random_initialize);
template void DeviceMalloc(int** ptr, size_t size, bool is_random_initialize);

template<typename T>
void DeviceMemSetZero(T* ptr, size_t size)
{
    ATER_ENFORCE_HIP_SUCCESS(hipMemset(static_cast<void*>(ptr), 0, sizeof(T) * size));
}

template void DeviceMemSetZero(float* ptr, size_t size);
template void DeviceMemSetZero(_Float16* ptr, size_t size);

template<typename T>
void DeviceFree(T*& ptr)
{
    if (ptr != NULL) {
        ATER_ENFORCE_HIP_SUCCESS(hipFree(ptr));
        ptr = NULL;
    }
}

template void DeviceFree(float*& ptr);
template void DeviceFree(_Float16*& ptr);
template void DeviceFree(int*& ptr);

template<typename T>
void DeviceFill(T* devptr, size_t size, T value, hipStream_t stream)
{
    T* arr = new T[size];
    std::fill(arr, arr + size, value);
    ATER_ENFORCE_HIP_SUCCESS(hipMemcpyAsync(devptr, arr, sizeof(T) * size, hipMemcpyHostToDevice, stream));
    delete[] arr;
}

template void DeviceFill(float* devptr, size_t size, float value, hipStream_t stream);
template void DeviceFill(_Float16* devptr, size_t size, _Float16 value, hipStream_t stream);
template void DeviceFill(int* devptr, size_t size, int value, hipStream_t stream);

template<typename T>
void HipD2HCpy(T* tgt, const T* src, const size_t size)
{
    ATER_ENFORCE_HIP_SUCCESS(hipMemcpy(tgt, src, sizeof(T) * size, hipMemcpyDeviceToHost));
}
template void HipD2HCpy(float* tgt, const float* src, size_t size);
template void HipD2HCpy(_Float16* tgt, const _Float16* src, size_t size);
template void HipD2HCpy(int* tgt, const int* src, size_t size);

template<typename T>
void HipH2DCpy(T* tgt, const T* src, const size_t size)
{
    ATER_ENFORCE_HIP_SUCCESS(hipMemcpy(tgt, src, sizeof(T) * size, hipMemcpyHostToDevice));
}
template void HipH2DCpy(float* tgt, const float* src, size_t size);
template void HipH2DCpy(_Float16* tgt, const _Float16* src, size_t size);
template void HipH2DCpy(int* tgt, const int* src, size_t size);

template<typename T>
void HipD2DCpy(T* tgt, const T* src, const size_t size)
{
    ATER_ENFORCE_HIP_SUCCESS(hipMemcpy(tgt, src, sizeof(T) * size, hipMemcpyDeviceToDevice));
}
template void HipD2DCpy(float* tgt, const float* src, size_t size);
template void HipD2DCpy(_Float16* tgt, const _Float16* src, size_t size);

template<typename T_OUT, typename T_IN>
__global__ void HipCast(T_OUT* dst, T_IN* src, const size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        dst[tid] = (T_OUT)((float)(src[tid]));
    }
}

template<typename T_OUT, typename T_IN>
void InvokeHipCast(T_OUT* dst, T_IN const* const src, const size_t size, hipStream_t stream)
{
    hipLaunchKernelGGL(HipCast, 256, 256, 0, stream, dst, src, size);
}
template void InvokeHipCast(float* dst, _Float16 const* const src, const size_t size, hipStream_t stream);

template<typename T>
void HipAutoCpy(T* tgt, const T* src, const size_t size, hipStream_t stream)
{
    if (stream != NULL) {
        ATER_ENFORCE_HIP_SUCCESS(hipMemcpyAsync(tgt, src, sizeof(T) * size, hipMemcpyDefault, stream));
    }
    else {
        ATER_ENFORCE_HIP_SUCCESS(hipMemcpy(tgt, src, sizeof(T) * size, hipMemcpyDefault));
    }
}
template void HipAutoCpy(float* tgt, const float* src, size_t size, hipStream_t stream);
template void HipAutoCpy(_Float16* tgt, const _Float16* src, size_t size, hipStream_t stream);

template<typename T_IN, typename T_OUT>
__global__ void HipD2DCpyConvert(T_OUT* dst, const T_IN* src, const size_t size)
{
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
        dst[tid] = (T_OUT)src[tid];
    }
}

template<typename T_IN, typename T_OUT>
void InvokeHipD2DCpyConvert(T_OUT* tgt, const T_IN* src, const size_t size, hipStream_t stream)
{
    hipLaunchKernelGGL(HipD2DCpyConvert, 256, 256, 0, stream, tgt, src, size);
}

template void InvokeHipD2DCpyConvert(float* tgt, const _Float16* src, const size_t size, hipStream_t stream);
template void InvokeHipD2DCpyConvert(float* tgt, const float* src, const size_t size, hipStream_t stream);
template void InvokeHipD2DCpyConvert(_Float16* tgt, const float* src, const size_t size, hipStream_t stream);
template void InvokeHipD2DCpyConvert(_Float16* tgt, const _Float16* src, const size_t size, hipStream_t stream);

template<typename T>
__global__ void hip_random_uniform_kernel(T* buffer, const size_t size, const int seq_offset)
{
    const int      idx = blockIdx.x * blockDim.x + threadIdx.x;
    hiprandState_t local_state;
    hiprand_init((unsigned long long int)1337, idx + seq_offset, 0, &local_state);
    for (size_t index = idx; index < size; index += blockDim.x * gridDim.x) {
        buffer[index] = (T)(hiprand_uniform(&local_state) * 0.2f - 0.1f);
    }
}

template<>
__global__ void hip_random_uniform_kernel<int>(int* buffer, const size_t size, const int seq_offset)
{
    const int      idx = blockIdx.x * blockDim.x + threadIdx.x;
    hiprandState_t local_state;
    hiprand_init((float)1337.f, idx + seq_offset, 0, &local_state);
    for (size_t index = idx; index < size; index += blockDim.x * gridDim.x) {
        buffer[index] = hiprand(&local_state);
    }
}

template<>
__global__ void hip_random_uniform_kernel<bool>(bool* buffer, const size_t size, const int seq_offset)
{
    const int      idx = blockIdx.x * blockDim.x + threadIdx.x;
    hiprandState_t local_state;
    hiprand_init((float)1337.f, idx + seq_offset, 0, &local_state);
    for (size_t index = idx; index < size; index += blockDim.x * gridDim.x) {
        buffer[index] = (hiprand(&local_state) % 2 == 0);
    }
}

template<>
__global__ void hip_random_uniform_kernel<char>(char* buffer, const size_t size, const int seq_offset)
{
    const int      idx = blockIdx.x * blockDim.x + threadIdx.x;
    hiprandState_t local_state;
    hiprand_init((float)1337.f, idx + seq_offset, 0, &local_state);
    for (size_t index = idx; index < size; index += blockDim.x * gridDim.x) {
        buffer[index] = hiprand(&local_state) % 0xFF;
    }
}

template<typename T>
void HipRandomUniform(T* buffer, const size_t size)
{
    static int seq_offset = 0;
    hip_random_uniform_kernel<T><<<256, 256>>>(buffer, size, seq_offset);
    seq_offset += 256 * 256;
}

template void HipRandomUniform(float* buffer, const size_t size);
template void HipRandomUniform(_Float16* buffer, const size_t size);
template void HipRandomUniform(int* buffer, const size_t size);

std::vector<char> ReadBuffer(const std::string& filename, size_t offset, size_t nbytes)
{
    std::ifstream is(filename, std::ios::binary | std::ios::ate);
    if (not is.is_open())
        throw std::runtime_error("Error opening file: " + filename);
    if (nbytes == 0) {
        // if there is a non-zero offset and nbytes is not set,
        // calculate size of remaining bytes to read
        nbytes = is.tellg();
        if (offset > nbytes)
            throw std::runtime_error("offset is larger than file size");
        nbytes -= offset;
    }
    if (nbytes < 1)
        throw std::runtime_error("Invalid size for: " + filename);
    is.seekg(offset, std::ios::beg);

    std::vector<char> buffer(nbytes, 0);
    if (not is.read(&buffer[0], nbytes))
        throw std::runtime_error("Error reading file: " + filename);

    VLOG(1) << "buffer size: " << buffer.size() << " offset: " << offset << " nbytes: " << nbytes;
    return buffer;
}

/*-----------------------------------Weight-------------------------------*/
template<typename T>
std::vector<T> LoadWeightFromBinHelper(const std::vector<size_t>& shape, const std::string& filename)
{
    if (shape.size() > 2) {
        LOG(ERROR) << "shape should have less than two dims";
        return std::vector<T>();
    }

    size_t dim0 = shape[0], dim1 = 1;
    if (shape.size() == 2) {
        dim1 = shape[1];
    }

    size_t size = dim0 * dim1;
    if (size == 0) {
        LOG(WARNING) << "shape is zero, skip loading weight from file " << filename;
        return std::vector<T>();
    }

    std::vector<T> host_array(size);
    std::ifstream  in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        LOG(WARNING) << "file " << filename << " cannot be opened, loading model fails!";
        return std::vector<T>();
    }

    size_t loaded_data_size = sizeof(T) * size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);

    LOG(INFO) << "Read " << loaded_data_size << " bytes from " << filename;
    in.read((char*)host_array.data(), loaded_data_size);

    size_t in_get_size = in.gcount();
    if (in_get_size != loaded_data_size) {
        LOG(WARNING) << "file " << filename << " only has " << in_get_size << ", but request " << loaded_data_size
                     << ", loading model fails!";
        return std::vector<T>();
    }

    in.close();
    // If we succeed, return an array with values.
    return host_array;
}

template<typename T, typename T_IN>
int LoadWeightFromBinFunc(T* ptr, const std::vector<size_t>& shape, const std::string& filename)
{
    std::vector<T_IN> host_array = LoadWeightFromBinHelper<T_IN>(shape, filename);

    if (host_array.empty()) {
        return 0;
    }

    if (std::is_same<T, T_IN>::value == true) {
        HipH2DCpy(ptr, (T*)host_array.data(), host_array.size());
    }
    else {
        T_IN* ptr_2 = nullptr;
        DeviceMalloc(&ptr_2, host_array.size(), false);
        HipH2DCpy(ptr_2, host_array.data(), host_array.size());
        InvokeHipD2DCpyConvert(ptr, ptr_2, host_array.size());
        DeviceFree(ptr_2);
    }
    return 0;
}

template int
LoadWeightFromBinFunc<float, float>(float* ptr, const std::vector<size_t>& shape, const std::string& filename);
template int
LoadWeightFromBinFunc<_Float16, float>(_Float16* ptr, const std::vector<size_t>& shape, const std::string& filename);
template int
LoadWeightFromBinFunc<float, _Float16>(float* ptr, const std::vector<size_t>& shape, const std::string& filename);
template int
LoadWeightFromBinFunc<_Float16, _Float16>(_Float16* ptr, const std::vector<size_t>& shape, const std::string& filename);

template<typename T>
int LoadWeightFromBin(T*                         ptr,
                      const std::vector<size_t>& shape,
                      const std::string&         filename,
                      const DataType&            model_file_type)
{
    switch (model_file_type) {
        case DataType::FLOAT32:
            LoadWeightFromBinFunc<T, float>(ptr, shape, filename);
            break;
        case DataType::FLOAT16:
            LoadWeightFromBinFunc<T, _Float16>(ptr, shape, filename);
            break;
        default:
            ATER_THROW(Unavailable("Does not support DataType{}", static_cast<int>(model_file_type)));
    }
    return 0;
}

template<>
int LoadWeightFromBin(int*                       ptr,
                      const std::vector<size_t>& shape,
                      const std::string&         filename,
                      const DataType&            model_file_type)
{
    LoadWeightFromBinFunc<int, int>(ptr, shape, filename);
    return 0;
}

template int LoadWeightFromBin(float*                     ptr,
                               const std::vector<size_t>& shape,
                               const std::string&         filename,
                               const DataType&            model_file_type);
template int LoadWeightFromBin(_Float16*                  ptr,
                               const std::vector<size_t>& shape,
                               const std::string&         filename,
                               const DataType&            model_file_type);

template<typename T>
std::vector<T> GenHostRandomBuffer(size_t n, size_t seed)
{
    std::vector<T>                         result(n);
    std::mt19937                           gen(seed);
    std::uniform_real_distribution<double> dis(-1.0);
    std::generate(result.begin(), result.end(), [&] { return dis(gen); });

    return result;
}

template std::vector<float>    GenHostRandomBuffer(size_t n, size_t seed);
template std::vector<_Float16> GenHostRandomBuffer(size_t n, size_t seed);

}  // namespace ater