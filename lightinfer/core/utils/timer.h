#pragma once

#include <chrono>

#include <hip/hip_runtime.h>

#include "lightinfer/core/utils/enforce.h"

namespace lightinfer {

// CPU Timer
class CPUTimer {
public:
    hipStream_t                                    stream_ = 0;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point startu;

    void Tic()
    {
        start = std::chrono::high_resolution_clock::now();
    }
    double Toc()
    {
        LI_ENFORCE_HIP_SUCCESS(hipStreamSynchronize(stream_));
        startu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(startu - start);
        double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
        return used_time_ms;
    }
};

// GPU Timer
class HipTimer {
private:
    hipEvent_t  event_start_;
    hipEvent_t  event_stop_;
    hipStream_t stream_;

public:
    explicit HipTimer(hipStream_t stream = 0)
    {
        stream_ = stream;
    }
    void Start()
    {
        LI_ENFORCE_HIP_SUCCESS(hipEventCreate(&event_start_));
        LI_ENFORCE_HIP_SUCCESS(hipEventCreate(&event_stop_));
        LI_ENFORCE_HIP_SUCCESS(hipEventRecord(event_start_, stream_));
    }
    float Stop()
    {
        float time;
        LI_ENFORCE_HIP_SUCCESS(hipEventRecord(event_stop_, stream_));
        LI_ENFORCE_HIP_SUCCESS(hipEventSynchronize(event_stop_));
        LI_ENFORCE_HIP_SUCCESS(hipEventElapsedTime(&time, event_start_, event_stop_));
        LI_ENFORCE_HIP_SUCCESS(hipEventDestroy(event_start_));
        LI_ENFORCE_HIP_SUCCESS(hipEventDestroy(event_stop_));
        return time;
    }
    ~HipTimer() {}
};

}  // namespace lightinfer