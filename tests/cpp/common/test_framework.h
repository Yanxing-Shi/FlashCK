/**
 * @file test_framework.h
 * @brief Unified test framework for FlashCK operations (correctness & performance)
 *
 * This framework provides a comprehensive testing infrastructure that supports:
 * - Correctness testing with reference implementations
 * - Performance benchmarking with throughput analysis
 * - GPU memory management and automatic cleanup
 * - Extensible configuration system for all operation types
 * - Unified reporting and comparison utilities
 *
 * Usage:
 * 1. Create operation-specific config classes (inherit from OpConfigBase)
 * 2. Implement reference functions and FlashCK wrappers
 * 3. Use UnifiedTestSuite<OpType> for both correctness and performance tests
 * 4. All operation inputs are automatically managed on GPU
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Helper macro for HIP error checking
#define HIP_CHECK(call)                                                                                                \
    do {                                                                                                               \
        hipError_t error = call;                                                                                       \
        if (error != hipSuccess) {                                                                                     \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(error)           \
                      << std::endl;                                                                                    \
            throw std::runtime_error("HIP error: " + std::string(hipGetErrorString(error)));                           \
        }                                                                                                              \
    } while (0)

namespace flashck {
namespace test {

// Forward declarations for config classes used in TFLOPs computation
template<typename T>
class LayerNormConfig;
template<typename T>
class RMSNormConfig;

// Forward declarations for framework classes
template<typename T>
class DataGenerator;
template<typename T>
class GpuMemoryManager;

// Traits for per-datatype test parameters (tolerances, etc.)
template<typename T>
struct TestTypeTraits {
    static constexpr float rtol       = 1e-3f;
    static constexpr float atol       = 1e-4f;
    static constexpr int   max_errors = 10;
};
// Specialization for HIP half-precision (if used)
template<>
struct TestTypeTraits<_Float16> {
    static constexpr float rtol       = 2e-2f;
    static constexpr float atol       = 2e-3f;
    static constexpr int   max_errors = 10;
};

// Specialization for float
template<>
struct TestTypeTraits<float> {
    static constexpr float rtol       = 1e-6f;
    static constexpr float atol       = 1e-8f;
    static constexpr int   max_errors = 10;
};
// Specialization for double
template<>
struct TestTypeTraits<double> {
    static constexpr float rtol       = 1e-8f;
    static constexpr float atol       = 1e-10f;
    static constexpr int   max_errors = 10;
};

/**
 * @brief Base interface for operation configurations
 * All operation-specific configs should inherit from this
 */
template<typename T>
class OpConfigBase {
public:
    virtual ~OpConfigBase() = default;

    // Core interface methods
    virtual std::string name() const           = 0;
    virtual std::string operation_type() const = 0;
    virtual size_t      output_size() const    = 0;
    virtual size_t      total_bytes() const    = 0;  // For throughput calculation

    // Data initialization
    virtual void init_test_data(DataGenerator<T>& data_gen) = 0;

    // GPU memory setup for FlashCK operations
    virtual void setup_gpu_inputs(GpuMemoryManager<T>& gpu_mem) = 0;
    virtual T*   get_gpu_output(GpuMemoryManager<T>& gpu_mem)   = 0;

    // CPU data access for reference implementations
    virtual void get_cpu_inputs_for_reference(std::vector<const T*>& inputs) const = 0;
};

/**
 * @brief Performance comparison metrics
 */
enum class PerformanceMetric {
    LATENCY,   // Lower is better
    TFLOPS,    // Higher is better
    BANDWIDTH  // Higher is better
};

/**
 * @brief Performance measurement result with extended metrics
 */
struct PerformanceResult {
    double      latency;         // Average latency in milliseconds
    double      tflops;          // TFLOPs performance
    double      bandwidth;       // Memory bandwidth in GB/s
    std::string config_name;     // Configuration name
    std::string operation_type;  // Operation type (e.g., LayerNorm, RMSNorm)

    // Static variable to control default comparison metric
    static PerformanceMetric default_metric;

    void print() const
    {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << operation_type << " - " << config_name << ":\n";
        std::cout << "  Latency: " << latency << " ms\n";
        std::cout << "  TFLOPs: " << tflops << " TFlops\n";
        std::cout << "  Bandwidth: " << bandwidth << " GB/s\n\n";
    }
};

/**
 * @brief GPU timer utility using HIP events for accurate performance measurements
 */
class GpuTimer {
public:
    GpuTimer()
    {
        HIP_CHECK(hipEventCreate(&start_event_));
        HIP_CHECK(hipEventCreate(&stop_event_));
    }

    ~GpuTimer()
    {
        hipEventDestroy(start_event_);
        hipEventDestroy(stop_event_);
    }

    void start()
    {
        HIP_CHECK(hipEventRecord(start_event_, 0));
    }

    double stop_ms()
    {
        HIP_CHECK(hipEventRecord(stop_event_, 0));
        HIP_CHECK(hipEventSynchronize(stop_event_));

        float elapsed_ms;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start_event_, stop_event_));
        return static_cast<double>(elapsed_ms);
    }

private:
    hipEvent_t start_event_;
    hipEvent_t stop_event_;
};

/**
 * @brief Enhanced GPU memory manager with automatic cleanup and debugging
 */
template<typename T>
class GpuMemoryManager {
public:
    /**
     * @brief Allocate GPU memory and copy data from CPU
     * @param host_data CPU data pointer
     * @param size Number of elements
     * @param name Optional name for debugging
     * @return GPU pointer
     */
    T* allocate_and_copy(const T* host_data, size_t size, const std::string& name = "")
    {
        T* gpu_data;
        HIP_CHECK(hipMalloc(&gpu_data, size * sizeof(T)));
        if (host_data) {
            HIP_CHECK(hipMemcpy(gpu_data, host_data, size * sizeof(T), hipMemcpyHostToDevice));
        }

        allocations_.push_back({gpu_data, size, name});
        total_allocated_bytes_ += size * sizeof(T);
        return gpu_data;
    }

    /**
     * @brief Allocate GPU memory without copying data
     * @param size Number of elements
     * @param name Optional name for debugging
     * @return GPU pointer
     */
    T* allocate(size_t size, const std::string& name = "")
    {
        return allocate_and_copy(nullptr, size, name);
    }

    /**
     * @brief Copy data from GPU to CPU
     * @param gpu_data GPU pointer
     * @param host_data CPU pointer
     * @param size Number of elements
     */
    void copy_to_host(T* gpu_data, T* host_data, size_t size)
    {
        HIP_CHECK(hipMemcpy(host_data, gpu_data, size * sizeof(T), hipMemcpyDeviceToHost));
    }

    /**
     * @brief Get total allocated bytes
     */
    size_t get_total_allocated_bytes() const
    {
        return total_allocated_bytes_;
    }

    /**
     * @brief Print allocation summary
     */
    void print_allocations() const
    {
        std::cout << "GPU Memory Allocations: " << allocations_.size() << " total, "
                  << (total_allocated_bytes_ / 1024.0 / 1024.0) << " MB\n";
        for (const auto& alloc : allocations_) {
            std::cout << "  - " << (alloc.name.empty() ? "unnamed" : alloc.name) << ": " << alloc.size << " elements\n";
        }
    }

    /**
     * @brief Destructor - automatically frees all allocated GPU memory
     */
    ~GpuMemoryManager()
    {
        for (const auto& alloc : allocations_) {
            hipFree(alloc.ptr);
        }
    }

private:
    struct Allocation {
        void*       ptr;
        size_t      size;
        std::string name;
    };

    std::vector<Allocation> allocations_;
    size_t                  total_allocated_bytes_ = 0;
};

/**
 * @brief Enhanced data generator with more distributions and utilities
 */
template<typename T>
class DataGenerator {
public:
    DataGenerator(unsigned seed = 42): gen_(seed) {}

    /**
     * @brief Generate random data with uniform distribution
     * @param data Output array
     * @param size Number of elements
     * @param min_val Minimum value
     * @param max_val Maximum value
     */
    void uniform(T* data, size_t size, float min_val = -2.0f, float max_val = 2.0f)
    {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<T>(dist(gen_));
        }
    }

    /**
     * @brief Generate constant data
     * @param data Output array
     * @param size Number of elements
     * @param value Constant value
     */
    void constant(T* data, size_t size, T value)
    {
        std::fill(data, data + size, value);
    }

    /**
     * @brief Generate normal distribution data
     * @param data Output array
     * @param size Number of elements
     * @param mean Mean value
     * @param stddev Standard deviation
     */
    void normal(T* data, size_t size, float mean = 0.0f, float stddev = 1.0f)
    {
        std::normal_distribution<float> dist(mean, stddev);
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<T>(dist(gen_));
        }
    }

    /**
     * @brief Generate special values for edge case testing
     * @param data Output array
     * @param size Number of elements
     * @param pattern Pattern type: "zeros", "ones", "sequential", "alternating"
     */
    void special_pattern(T* data, size_t size, const std::string& pattern)
    {
        if (pattern == "zeros") {
            std::fill(data, data + size, static_cast<T>(0));
        }
        else if (pattern == "ones") {
            std::fill(data, data + size, static_cast<T>(1));
        }
        else if (pattern == "sequential") {
            for (size_t i = 0; i < size; ++i) {
                data[i] = static_cast<T>(i);
            }
        }
        else if (pattern == "alternating") {
            for (size_t i = 0; i < size; ++i) {
                data[i] = static_cast<T>((i % 2) ? 1.0f : -1.0f);
            }
        }
    }

private:
    std::mt19937 gen_;
};

/**
 * @brief Enhanced tensor comparison utilities with detailed error reporting
 */
template<typename T>
class TensorComparator {
public:
    /**
     * @brief Compare two tensors with relative and absolute tolerance
     * @param a First tensor
     * @param b Second tensor
     * @param size Number of elements
     * @param rtol Relative tolerance
     * @param atol Absolute tolerance
     * @param max_errors Maximum number of errors to report
     * @return true if tensors are close, false otherwise
     */
    static bool
    allclose(const T* a, const T* b, size_t size, float rtol = 1e-3f, float atol = 1e-5f, int max_errors = 10)
    {
        bool all_close   = true;
        int  error_count = 0;

        for (size_t i = 0; i < size && error_count < max_errors; ++i) {
            float a_val     = static_cast<float>(a[i]);
            float b_val     = static_cast<float>(b[i]);
            float diff      = std::abs(a_val - b_val);
            float threshold = atol + rtol * std::abs(b_val);

            if (diff > threshold) {
                if (error_count == 0) {
                    std::cout << "Tensor comparison failed. First " << max_errors << " mismatches:\n";
                }
                std::cout << "  [" << i << "]: got " << a_val << ", expected " << b_val << ", diff=" << diff
                          << ", threshold=" << threshold << std::endl;
                all_close = false;
                error_count++;
            }
        }

        if (!all_close && error_count == max_errors) {
            std::cout << "  ... (truncated, showing first " << max_errors << " errors only)\n";
        }

        return all_close;
    }

    /**
     * @brief Compute comprehensive error metrics
     * @param a First tensor
     * @param b Second tensor
     * @param size Number of elements
     * @return Error metrics structure
     */
    struct ErrorMetrics {
        float  mae;             // Mean absolute error
        float  rmse;            // Root mean square error
        float  max_abs_err;     // Maximum absolute error
        float  max_rel_err;     // Maximum relative error
        size_t mismatch_count;  // Number of mismatched elements
    };

    static ErrorMetrics
    compute_error_metrics(const T* a, const T* b, size_t size, float rtol = 1e-3f, float atol = 1e-5f)
    {
        ErrorMetrics metrics = {0.0f, 0.0f, 0.0f, 0.0f, 0};

        for (size_t i = 0; i < size; ++i) {
            float a_val    = static_cast<float>(a[i]);
            float b_val    = static_cast<float>(b[i]);
            float abs_diff = std::abs(a_val - b_val);
            float rel_diff = (std::abs(b_val) > 1e-8f) ? abs_diff / std::abs(b_val) : abs_diff;

            metrics.mae += abs_diff;
            metrics.rmse += abs_diff * abs_diff;
            metrics.max_abs_err = std::max(metrics.max_abs_err, abs_diff);
            metrics.max_rel_err = std::max(metrics.max_rel_err, rel_diff);

            float threshold = atol + rtol * std::abs(b_val);
            if (abs_diff > threshold) {
                metrics.mismatch_count++;
            }
        }

        metrics.mae /= size;
        metrics.rmse = std::sqrt(metrics.rmse / size);

        return metrics;
    }

    /**
     * @brief Print error metrics summary
     */
    static void print_error_metrics(const ErrorMetrics& metrics, size_t total_elements)
    {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Error Metrics:\n";
        std::cout << "  MAE: " << metrics.mae << "\n";
        std::cout << "  RMSE: " << metrics.rmse << "\n";
        std::cout << "  Max Abs Error: " << metrics.max_abs_err << "\n";
        std::cout << "  Max Rel Error: " << metrics.max_rel_err << "\n";
        std::cout << "  Mismatches: " << metrics.mismatch_count << "/" << total_elements << " ("
                  << (100.0 * metrics.mismatch_count / total_elements) << "%)\n";
    }
};

/**
 * @brief Unified test suite for both correctness and performance testing
 *
 * This class provides a comprehensive testing framework that can handle any operation type
 * by using the OpConfigBase interface. It supports:
 * - Correctness testing with detailed error reporting
 * - Performance benchmarking with statistical analysis
 * - Automatic GPU memory management
 * - Extensible configuration system
 *
 * Template parameter T: Data type (float, half, etc.)
 */
template<typename T>
class UnifiedTestSuite: public ::testing::Test {
public:
    UnifiedTestSuite(): data_gen_(42) {}

protected:
    void SetUp() override
    {
        // Initialize GPU context if needed
        HIP_CHECK(hipDeviceSynchronize());
    }

    void TearDown() override
    {
        // Cleanup is handled by GpuMemoryManager destructor
    }

    /**
     * @brief Run correctness test for a set of configurations
     * @param test_configs Vector of operation configurations
     * @param reference_impl Reference implementation function
     * @param flashck_impl FlashCK implementation function
     * @param tolerance_rtol Relative tolerance for comparison
     * @param tolerance_atol Absolute tolerance for comparison
     * @param verbose Enable detailed error reporting
     */
    template<typename ConfigType>
    void run_correctness_test(const std::vector<std::shared_ptr<ConfigType>>&            test_configs,
                              std::function<void(const ConfigType&, T*)>                 reference_impl,
                              std::function<T*(const ConfigType&, GpuMemoryManager<T>&)> flashck_impl,
                              float tolerance_rtol = TestTypeTraits<T>::rtol,
                              float tolerance_atol = TestTypeTraits<T>::atol,
                              bool  verbose        = false)
    {
        std::cout << "\n=== CORRECTNESS TESTING ===\n";

        for (const auto& config : test_configs) {
            SCOPED_TRACE("Config: " + config->name());

            if (verbose) {
                std::cout << "\nTesting: " << config->name() << " (" << config->operation_type() << ")\n";
            }

            // Initialize test data
            config->init_test_data(data_gen_);

            // Setup GPU memory for FlashCK implementation
            GpuMemoryManager<T> gpu_mem;
            config->setup_gpu_inputs(gpu_mem);

            // Run FlashCK implementation
            T* flashck_output = flashck_impl(*config, gpu_mem);
            ASSERT_NE(flashck_output, nullptr) << "FlashCK implementation failed for " << config->name();

            // Copy result back to CPU for comparison
            auto cpu_output = std::make_unique<T[]>(config->output_size());
            gpu_mem.copy_to_host(flashck_output, cpu_output.get(), config->output_size());

            // Run reference implementation
            auto reference_output = std::make_unique<T[]>(config->output_size());
            reference_impl(*config, reference_output.get());

            // Compare results
            if (verbose) {
                auto metrics = TensorComparator<T>::compute_error_metrics(
                    cpu_output.get(), reference_output.get(), config->output_size(), tolerance_rtol, tolerance_atol);
                TensorComparator<T>::print_error_metrics(metrics, config->output_size());
            }

            EXPECT_TRUE(TensorComparator<T>::allclose(cpu_output.get(),
                                                      reference_output.get(),
                                                      config->output_size(),
                                                      tolerance_rtol,
                                                      tolerance_atol,
                                                      TestTypeTraits<T>::max_errors))
                << "Output mismatch for config: " << config->name();

            if (verbose) {
                std::cout << "✓ " << config->name() << " passed\n";
            }
        }

        std::cout << "Correctness testing completed: " << test_configs.size() << " configurations\n";
    }

    /**
     * @brief Run performance benchmark for a set of configurations
     * @param test_configs Vector of operation configurations
     * @param flashck_impl FlashCK implementation function
     * @param num_runs Number of benchmark runs
     * @param warmup_runs Number of warmup runs
     * @return Vector of performance results
     *
     * Example usage:
     *   auto results1 = run_performance_test(configs, impl);
     *
     */

    template<typename ConfigType>
    std::vector<PerformanceResult>
    run_performance_test(const std::vector<std::shared_ptr<ConfigType>>&            test_configs,
                         std::function<T*(const ConfigType&, GpuMemoryManager<T>&)> flashck_impl,
                         int                                                        num_runs    = 10,
                         int                                                        warmup_runs = 3)
    {
        std::cout << "\n=== PERFORMANCE BENCHMARKING ===\n";
        std::vector<PerformanceResult> results;

        for (const auto& config : test_configs) {
            std::cout << "Benchmarking: " << config->name() << "...";

            // Initialize test data
            config->init_test_data(data_gen_);

            // Setup GPU memory (reused across runs)
            GpuMemoryManager<T> gpu_mem;
            config->setup_gpu_inputs(gpu_mem);

            GpuTimer            gpu_timer;
            std::vector<double> times;
            times.reserve(num_runs);

            // Warmup runs
            for (int i = 0; i < warmup_runs; ++i) {
                try {
                    T* result = flashck_impl(*config, gpu_mem);
                    HIP_CHECK(hipDeviceSynchronize());
                    if (result == nullptr)
                        break;
                }
                catch (...) {
                    // Skip failed warmup runs
                }
            }

            // Benchmark runs using GPU timer
            for (int i = 0; i < num_runs; ++i) {
                HIP_CHECK(hipDeviceSynchronize());
                gpu_timer.start();
                T*     result  = flashck_impl(*config, gpu_mem);
                double time_ms = gpu_timer.stop_ms();  // This includes synchronization

                if (result != nullptr) {
                    times.push_back(time_ms);
                }
            }

            if (!times.empty()) {
                // Compute statistics
                double min_time = *std::min_element(times.begin(), times.end());
                double max_time = *std::max_element(times.begin(), times.end());
                double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

                // Use average latency as the primary metric
                double latency = avg_time;

                // Compute memory bandwidth (GB/s)
                size_t total_bytes = config->total_bytes();
                double bandwidth   = (total_bytes / 1e9) / (avg_time / 1000.0);

                // Compute TFLOPs - this needs to be operation-specific
                // For now, use a basic estimate based on operation type
                double tflops = compute_tflops(*config, avg_time);

                results.push_back({latency, tflops, bandwidth, config->name(), config->operation_type()});

                std::cout << " DONE\n";
                std::cout << "  Latency: " << std::fixed << std::setprecision(3) << latency << " ms";
                std::cout << ", TFLOPs: " << std::fixed << std::setprecision(3) << tflops;
                std::cout << ", Bandwidth: " << std::fixed << std::setprecision(3) << bandwidth << " GB/s\n";
            }
            else {
                std::cout << " FAILED\n";
            }
        }

        // Print summary
        print_performance_summary(results);
        return results;
    }

    /**
     * @brief Print performance summary table
     * @param results Performance results to display
     */
    void print_performance_summary(const std::vector<PerformanceResult>& results)
    {
        if (results.empty())
            return;

        std::cout << "\n=== PERFORMANCE SUMMARY ===\n";
        std::cout << std::left << std::setw(25) << "Configuration" << std::right << std::setw(12) << "Latency (ms)"
                  << std::setw(12) << "TFLOPs" << std::setw(15) << "Bandwidth" << "\n";
        std::cout << std::string(75, '-') << "\n";

        for (const auto& result : results) {
            std::cout << std::left << std::setw(25) << result.config_name << std::right << std::fixed
                      << std::setprecision(3) << std::setw(12) << result.latency << std::setw(12) << result.tflops
                      << std::setw(12) << result.bandwidth << " GB/s" << "\n";
        }
        std::cout << "\n";
    }

private:
    /**
     * @brief Compute TFLOPs for different operation types
     * @param config Operation configuration
     * @param time_ms Execution time in milliseconds
     * @return TFLOPs performance
     */
    template<typename ConfigType>
    double compute_tflops(const ConfigType& config, double time_ms)
    {
        std::string op_type = config.operation_type();
        double      time_s  = time_ms / 1000.0;

        if (op_type == "LayerNorm") {
            // LayerNorm FLOPs: 2 * m * n (mean) + 2 * m * n (variance) + 3 * m * n (normalize + scale + bias)
            // Approximate: 7 * m * n operations
            auto* ln_config = dynamic_cast<const LayerNormConfig<T>*>(&config);
            if (ln_config) {
                double flops = 7.0 * ln_config->m() * ln_config->n();
                return (flops / 1e12) / time_s;
            }
        }
        else if (op_type == "RMSNorm") {
            // RMSNorm FLOPs: 2 * m * n (square + sum) + 2 * m * n (normalize + scale)
            // Approximate: 4 * m * n operations
            auto* rms_config = dynamic_cast<const RMSNormConfig<T>*>(&config);
            if (rms_config) {
                double flops = 4.0 * rms_config->m() * rms_config->n();
                return (flops / 1e12) / time_s;
            }
        }

        // Default: return 0 for unknown operations
        return 0.0;
    }

protected:
    DataGenerator<T> data_gen_;
};

}  // namespace test
}  // namespace flashck
