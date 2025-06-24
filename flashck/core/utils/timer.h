#pragma once

#include <chrono>

#include <hip/hip_runtime.h>

namespace flashck {

/**
 * @brief A timer for measuring CPU code execution time.
 *
 * This class uses the high-resolution clock from the C++ standard library
 * to measure the duration between Tic() and Toc() calls.
 */
class CPUTimer {
public:
    /**
     * @brief Records the start time point.
     */
    void Tic();

    /**
     * @brief Calculates the elapsed time since Tic() was called.
     *
     * @return The elapsed time in milliseconds as a double.
     */
    double Toc();

private:
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief A timer for measuring GPU code execution time using HIP events.
 *
 * This class uses HIP events to accurately measure the time taken by
 * GPU operations on a specified stream.
 */
class HipTimer {
public:
    /**
     * @brief Constructs a HipTimer associated with a HIP stream.
     *
     * @param stream The HIP stream to time operations on. Default is 0 (default stream).
     */
    explicit HipTimer(hipStream_t stream = 0);

    /**
     * @brief Records the start event on the associated stream.
     */
    void Start();

    /**
     * @brief Records the stop event, synchronizes, and calculates the elapsed time.
     *
     * @return The elapsed time in milliseconds as a float.
     */
    float Stop();

private:
    hipEvent_t  start_event_;  ///< The start event for timing
    hipEvent_t  stop_event_;   ///< The stop event for timing
    hipStream_t stream_;       ///< The associated HIP stream
};

}  // namespace flashck
