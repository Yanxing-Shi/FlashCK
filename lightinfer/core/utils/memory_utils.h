#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "hip/hip_runtime.h"

#include "lightinfer/core/utils/dtype.h"

namespace lightinfer {

/*-----------------------------device memory function ----------------------------*/
template<typename T>
void DeviceMalloc(
    T** ptr, size_t size, bool is_random_initialize = false, const int min_value = 0, const int max_value = 1);

template<typename T>
void DeviceMemSetZero(T* ptr, size_t size);

template<typename T>
void DeviceMemSetZeroAsync(T* ptr, size_t size, hipStream_t stream);

template<typename T>
void DeviceFree(T*& ptr);

template<typename T>
void DeviceFill(T* devptr, size_t size, T value, hipStream_t stream = 0);

template<typename T>
void HipD2HCpy(T* tgt, const T* src, const size_t size);

template<typename T>
void HipD2HCpyAsync(T* tgt, const T* src, const size_t size, hipStream_t stream);

template<typename T>
void HipH2DCpy(T* tgt, const T* src, const size_t size);

template<typename T>
void HipH2DCpyAsync(T* tgt, const T* src, const size_t size, hipStream_t stream);

template<typename T>
void HipD2DCpy(T* tgt, const T* src, const size_t size);

template<typename T>
void HipD2DCpyAsync(T* tgt, const T* src, const size_t size, hipStream_t stream);

template<typename T_OUT, typename T_IN>
void InvokeHipCast(T_OUT* dst, T_IN const* const src, const size_t size, hipStream_t stream);

template<typename T>
void HipAutoCpy(T* tgt, const T* src, const size_t size, hipStream_t stream = NULL);

template<typename T_IN, typename T_OUT>
void InvokeHipD2DCpyConvert(T_OUT* tgt, const T_IN* src, const size_t size, hipStream_t stream = NULL);

template<typename T>
void HipRandomUniform(T* buffer, const size_t size, const int min_value = 0, const int max_value = 1);

/*-----------------------------host memory function ----------------------------*/
template<typename T>
inline void HostInitRandom(T* ptr, size_t size, float minval, float maxval)
{
    for (size_t i = 0; i < size; ++i) {
        float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        val *= (maxval - minval);
        ptr[i] = static_cast<T>(minval + val);
    }
}

inline void initRandomInt(int* ptr, size_t size, int minval, int maxval)
{
    assert(minval < maxval);
    int mod = maxval - minval;
    for (size_t i = 0; i < size; ++i) {
        ptr[i] = minval + rand() % mod;
    }
}

std::string ReadBuffer(const std::string& filename, size_t offset = 0, size_t nbytes = 0);

struct ConcateSlice {
    std::vector<std::pair<size_t, size_t>> slices;
};

template<typename T>
int LoadWeightFromBin(T*                               ptr,
                      const std::vector<size_t>&       shape,
                      const std::string&               filename,
                      const DataType&                  model_file_type = DataType::FLOAT32,
                      const std::vector<ConcateSlice>& slices          = std::vector<ConcateSlice>());

std::vector<float> LoadArrayFromBin(const std::vector<size_t>&       shape,
                                    const std::string&               filename,
                                    const std::vector<ConcateSlice>& slices = std::vector<ConcateSlice>());

template<typename T>
std::vector<T> GenHostRandomBuffer(size_t n, size_t seed = 0);

inline bool CheckIfFileExist(const std::string& file_path)
{
    std::ifstream in(file_path, std::ios::in | std::ios::binary);
    if (in.is_open()) {
        in.close();
        return true;
    }
    return false;
}

}  // namespace lightinfer