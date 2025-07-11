#pragma once

#include <chrono>

#include <hip/hip_runtime.h>

namespace flashck {

class CPUTimer {
public:
    void Tic();

    double Toc();

private:
    std::chrono::high_resolution_clock::time_point start_;
};

class HipTimer {
public:
    explicit HipTimer(hipStream_t stream = 0);

    void Start();

    float Stop();

private:
    hipEvent_t  start_event_;
    hipEvent_t  stop_event_;
    hipStream_t stream_;
};

}  // namespace flashck
