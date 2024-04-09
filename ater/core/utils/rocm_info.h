#pragma once

#include <array>
#include <vector>

#include <hip/hip_runtime.h>

namespace ater {

// Get the total number of GPU devices in system.
int GetGPUDeviceCount();

// Get the compute capability of the ith GPU (format: major * 10 + minor)
int GetGPUComputeCapability(int i);

// Get the runtime version of the ith GPU
int GetGPURuntimeVersion(int id);

// Get the driver version of the ith GPU
int GetGPUDriverVersion(int id);

// Get the MultiProcessors of the ith GPU.
int GetGPUMultiProcessors(int id);

// Get the maximum number of threads per multiprocessor for the ith GPU.
int GetGPUMaxThreadsPerMultiProcessor(int id);

// Get the maximum number of threads per block for the ith GPU.
int GetGPUMaxThreadsPerBlock(int id);

// Get the maximum GridDim size for GPU buddy allocator.
std::array<int, 3> GetGpuMaxGridDimSize(int id);

// Get the properties of the ith GPU device.
const hipDeviceProp_t& GetDeviceProperties(int id);

// Get a list of device ids from environment variable or use all.
std::vector<int> GetSelectedDevices();

// Get the current GPU device id.
int GetCurrentDeviceId();

// Set the GPU device id for next execution.
void SetDeviceId(int device_id);

// Get the GPU device id for next execution.
void GetSetDevice(int input_device_id, int* output_device_id = nullptr);

// Get the GPU device name for i-th GPU.
std::string GetDeviceName(int device_id = 0);

// Check if the GPU device supports the specified XDL.
bool IsXdlSupported(int device_id = 0);

// Check if the GPU device supports the specified wmma.
bool IsWmmaSupported(int device_id = 0);

// Get device arch
std::string GetDeviceArch(int device_id = 0);

// GPU stream synchronization.
void GpuStreamSync(hipStream_t stream);

// GPU destroy stream.
void GpuStreamDestroy(hipStream_t stream);

// return if the pointer is on device
bool IsGpuPointer(void* ptr);
}  // namespace ater