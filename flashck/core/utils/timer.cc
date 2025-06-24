#include "flashck/core/utils/timer.h"

#include "flashck/core/utils/macros.h"

namespace flashck {

/**
 * @brief Records the start time point.
 */
void CPUTimer::Tic()
{
    start_ = std::chrono::high_resolution_clock::now();
}

/**
 * @brief Calculates the elapsed time since Tic() was called.
 *
 * @return The elapsed time in milliseconds as a double.
 */
double CPUTimer::Toc()
{
    const auto                          end      = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - start_;
    return duration.count() * 1000.0;
}

/**
 * @brief Constructs a HipTimer associated with a HIP stream.
 *
 * @param stream The HIP stream to time operations on. Default is 0 (default stream).
 */
HipTimer::HipTimer(hipStream_t stream): stream_(stream)
{
    HIP_ERROR_CHECK(hipEventCreate(&start_event_));
    HIP_ERROR_CHECK(hipEventCreate(&stop_event_));
}

/**
 * @brief Records the start event on the associated stream.
 */
void HipTimer::Start()
{
    HIP_ERROR_CHECK(hipEventRecord(start_event_, stream_));
}

/**
 * @brief Records the stop event, synchronizes, and calculates the elapsed time.
 *
 * @return The elapsed time in milliseconds as a float.
 */
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
