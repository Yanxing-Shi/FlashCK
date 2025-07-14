#pragma once

#include <algorithm>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

namespace flashck {

int GetCurrentDeviceId();

void SetDeviceAndGetPrevious(int target_device_id, int* previous_device_id = nullptr);

std::string GetDeviceName(int device_id = 0);

bool IsGpuPointer(const void* ptr);

void LogGpuMemoryUsage(std::string_view device_name);

// HipStreamUniquePtr CreateHipStream()
// {
//     hipStream_t stream;
//     hipError_t  err = hipStreamCreate(&stream);
//     if (err != hipSuccess) {
//         throw std::runtime_error("HIP stream creation failed");
//     }
//     return HipStreamUniquePtr(stream);
// }

// HipEventUniquePtr CreateHipEvent(unsigned flags = hipEventDefault)
// {
//     hipEvent_t event;
//     hipError_t err = hipEventCreateWithFlags(&event, flags);
//     if (err != hipSuccess) {
//         throw std::runtime_error("HIP event creation failed");
//     }
//     return HipEventUniquePtr(event);
// }

}  // namespace flashck