
#include <chrono>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "flashck/wrapper/cpp/norm/rms_norm.h"
#include <hip/hip_runtime.h>

// Helper function for HIP error checking
#define HIP_CHECK(call)                                                                                                \
    do {                                                                                                               \
        hipError_t error = call;                                                                                       \
        if (error != hipSuccess) {                                                                                     \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(error)           \
                      << std::endl;                                                                                    \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

namespace {

// Timer utility class
class Timer {
public:
    void start()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double stop_ms()
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;  // Convert to milliseconds
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

// Performance measurement helper
struct BenchmarkResult {
    double min_time_ms;
    double max_time_ms;
    double avg_time_ms;
    double throughput_gb_s;
    int    m, n;

    void print(const std::string& test_name) const
    {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << test_name << " [" << m << "x" << n << "]:\n";
        std::cout << "  Min: " << min_time_ms << " ms\n";
        std::cout << "  Max: " << max_time_ms << " ms\n";
        std::cout << "  Avg: " << avg_time_ms << " ms\n";
        std::cout << "  Throughput: " << throughput_gb_s << " GB/s\n\n";
    }
};

// Helper function to allocate and copy data to GPU
template<typename T>
T* allocate_and_copy_to_gpu(const T* host_data, int size)
{
    T* gpu_data;
    HIP_CHECK(hipMalloc(&gpu_data, size * sizeof(T)));
    HIP_CHECK(hipMemcpy(gpu_data, host_data, size * sizeof(T), hipMemcpyHostToDevice));
    return gpu_data;
}

template<typename T>
BenchmarkResult benchmark_rms_norm(int m, int n, int num_runs = 10, int warmup_runs = 3)
{
    std::mt19937                          gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Allocate CPU memory for initialization
    auto input_cpu = std::make_unique<T[]>(m * n);
    auto gamma_cpu = std::make_unique<T[]>(n);

    // Initialize data on CPU
    for (int i = 0; i < m * n; ++i) {
        input_cpu[i] = static_cast<T>(dist(gen));
    }
    for (int i = 0; i < n; ++i) {
        gamma_cpu[i] = static_cast<T>(dist(gen) * 0.1f + 1.0f);
    }

    // Allocate GPU memory and copy data
    T* input_gpu = allocate_and_copy_to_gpu(input_cpu.get(), m * n);
    T* gamma_gpu = allocate_and_copy_to_gpu(gamma_cpu.get(), n);

    Timer               timer;
    std::vector<double> times;
    times.reserve(num_runs);

    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        try {
            flashck::rms_norm_fwd(input_gpu, gamma_gpu, m, n);
            HIP_CHECK(hipDeviceSynchronize());  // Ensure GPU computation completes
        }
        catch (...) {
            // Skip failed runs during warmup
        }
    }

    // Benchmark runs
    for (int i = 0; i < num_runs; ++i) {
        HIP_CHECK(hipDeviceSynchronize());  // Ensure clean start
        timer.start();
        T* result = flashck::rms_norm_fwd(input_gpu, gamma_gpu, m, n);
        HIP_CHECK(hipDeviceSynchronize());  // Ensure GPU computation completes
        double time_ms = timer.stop_ms();

        if (result != nullptr) {
            times.push_back(time_ms);
        }
    }

    // Clean up GPU memory
    HIP_CHECK(hipFree(input_gpu));
    HIP_CHECK(hipFree(gamma_gpu));

    if (times.empty()) {
        return {0.0, 0.0, 0.0, 0.0, m, n};
    }

    // Calculate statistics
    double min_time = *std::min_element(times.begin(), times.end());
    double max_time = *std::max_element(times.begin(), times.end());
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    // Calculate throughput (assume reading input + gamma, writing output)
    double bytes_per_op    = sizeof(T) * (2 * m * n + n);  // input, output, gamma (no beta for RMSNorm)
    double throughput_gb_s = (bytes_per_op / 1e9) / (min_time / 1000.0);

    return {min_time, max_time, avg_time, throughput_gb_s, m, n};
}

}  // anonymous namespace

class RMSNormBenchmarkTest: public ::testing::Test {
protected:
    void SetUp() override
    {
        std::cout << "\n=== RMSNorm Performance Benchmarks ===\n\n";
    }
};

TEST_F(RMSNormBenchmarkTest, SmallSequences)
{
    // Test typical small sequence lengths
    std::vector<std::pair<int, int>> configs = {
        {32, 768},   // Small batch, BERT-base hidden size
        {64, 768},   // Medium batch, BERT-base hidden size
        {128, 768},  // Large batch, BERT-base hidden size
    };

    std::cout << "Small Sequence Length Benchmarks:\n";
    for (auto [m, n] : configs) {
        auto result = benchmark_rms_norm<float>(m, n);
        result.print("RMSNorm Float32");
    }
}

TEST_F(RMSNormBenchmarkTest, LargeSequences)
{
    // Test typical large sequence lengths
    std::vector<std::pair<int, int>> configs = {
        {32, 1024},  // GPT-small hidden size
        {64, 1024},  // GPT-small hidden size
        {32, 4096},  // GPT-large hidden size
        {16, 4096},  // GPT-large hidden size
    };

    std::cout << "Large Sequence Length Benchmarks:\n";
    for (auto [m, n] : configs) {
        auto result = benchmark_rms_norm<float>(m, n);
        result.print("RMSNorm Float32");
    }
}

TEST_F(RMSNormBenchmarkTest, VaryingBatchSizes)
{
    const int        n           = 768;  // Fixed hidden dimension
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256};

    std::cout << "Varying Batch Size Benchmarks (n=" << n << "):\n";
    for (int m : batch_sizes) {
        auto result = benchmark_rms_norm<float>(m, n, 5, 2);  // Fewer runs for large batches
        result.print("BatchSize=" + std::to_string(m));
    }
}

TEST_F(RMSNormBenchmarkTest, VaryingHiddenSizes)
{
    const int        m            = 32;  // Fixed batch size
    std::vector<int> hidden_sizes = {256, 512, 768, 1024, 1536, 2048, 4096};

    std::cout << "Varying Hidden Size Benchmarks (m=" << m << "):\n";
    for (int n : hidden_sizes) {
        auto result = benchmark_rms_norm<float>(m, n, 5, 2);
        result.print("HiddenSize=" + std::to_string(n));
    }
}

TEST_F(RMSNormBenchmarkTest, MemoryIntensiveWorkloads)
{
    // Test memory-intensive configurations
    std::vector<std::pair<int, int>> configs = {
        {512, 1024},  // Large batch
        {256, 2048},  // Very large hidden
        {1024, 768},  // Very large batch
        {128, 4096},  // Large both
    };

    std::cout << "Memory-Intensive Workload Benchmarks:\n";
    for (auto [m, n] : configs) {
        auto result = benchmark_rms_norm<float>(m, n, 3, 1);  // Fewer runs
        result.print("RMSNorm Float32");
    }
}

TEST_F(RMSNormBenchmarkTest, ConsistencyCheck)
{
    // Test multiple runs for consistency
    const int m = 64, n = 768;
    const int num_tests = 5;

    std::cout << "Consistency Check (multiple runs):\n";
    std::vector<double> avg_times;

    for (int i = 0; i < num_tests; ++i) {
        auto result = benchmark_rms_norm<float>(m, n, 10, 3);
        avg_times.push_back(result.avg_time_ms);
        std::cout << "Run " << (i + 1) << ": " << std::fixed << std::setprecision(3) << result.avg_time_ms << " ms\n";
    }

    // Calculate coefficient of variation
    double mean     = std::accumulate(avg_times.begin(), avg_times.end(), 0.0) / avg_times.size();
    double variance = 0.0;
    for (double time : avg_times) {
        variance += (time - mean) * (time - mean);
    }
    variance /= avg_times.size();
    double cv = std::sqrt(variance) / mean;

    std::cout << "Mean: " << mean << " ms, CV: " << (cv * 100) << "%\n";

    // Expect reasonable consistency (CV < 20%)
    EXPECT_LT(cv, 0.20) << "Performance varies too much between runs";
}

// Comparison test between LayerNorm and RMSNorm
TEST_F(RMSNormBenchmarkTest, ComparisonWithLayerNorm)
{
    const int m = 64, n = 768;

    std::cout << "Performance Comparison (LayerNorm vs RMSNorm):\n";

    // Benchmark RMSNorm
    auto rms_result = benchmark_rms_norm<float>(m, n, 10, 3);
    rms_result.print("RMSNorm Float32");

    // Note: LayerNorm benchmark would require separate implementation
    // This is just a placeholder for comparison
    std::cout << "Note: RMSNorm should be faster than LayerNorm due to:\n";
    std::cout << "  - No mean calculation required\n";
    std::cout << "  - No bias (beta) parameter\n";
    std::cout << "  - Simpler computation graph\n\n";
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
