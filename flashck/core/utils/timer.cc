#include "flashck/core/utils/timer.h"

#include "flashck/core/utils/macros.h"

namespace flashck {

void CPUTimer::Tic()
{
    start_ = std::chrono::high_resolution_clock::now();
}

double CPUTimer::Toc()
{
    const auto                          end      = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - start_;
    return duration.count() * 1000.0;
}

HipTimer::HipTimer(hipStream_t stream): stream_(stream)
{
    HIP_ERROR_CHECK(hipEventCreate(&start_event_));
    HIP_ERROR_CHECK(hipEventCreate(&stop_event_));
}

void HipTimer::Start()
{
    HIP_ERROR_CHECK(hipEventRecord(start_event_, stream_));
}

float HipTimer::Stop()
{
    HIP_ERROR_CHECK(hipEventRecord(stop_event_, stream_));
    HIP_ERROR_CHECK(hipEventSynchronize(stop_event_));
    float time_ms = 0.0f;
    HIP_ERROR_CHECK(hipEventElapsedTime(&time_ms, start_event_, stop_event_));

    HIP_ERROR_CHECK(hipEventDestroy(start_event_));
    HIP_ERROR_CHECK(hipEventDestroy(stop_event_));

    return time_ms;
}

}  // namespace flashck
