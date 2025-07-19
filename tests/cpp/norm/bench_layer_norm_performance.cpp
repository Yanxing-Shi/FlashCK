/**
 * @file bench_layer_norm_performance.cpp
 * @brief Performance benchmarks for the header-only LayerNorm wrapper
 */

#include <chrono>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "flashck/wrapper/cpp/norm/layer_norm.h"

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

template<typename T>
BenchmarkResult benchmark_layer_norm(int m, int n, int num_runs = 10, int warmup_runs = 3)
{
    std::mt19937                          gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Allocate memory
    auto input = std::make_unique<T[]>(m * n);
    auto gamma = std::make_unique<T[]>(n);
    auto beta  = std::make_unique<T[]>(n);

    // Initialize data
    for (int i = 0; i < m * n; ++i) {
        input[i] = static_cast<T>(dist(gen));
    }
    for (int i = 0; i < n; ++i) {
        gamma[i] = static_cast<T>(dist(gen) * 0.1f + 1.0f);
        beta[i]  = static_cast<T>(dist(gen) * 0.1f);
    }

    Timer               timer;
    std::vector<double> times;
    times.reserve(num_runs);

    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        try {
            flashck::layer_norm_fwd(input.get(), gamma.get(), beta.get(), m, n);
        }
        catch (...) {
            // Skip failed runs during warmup
        }
    }

    // Benchmark runs
    for (int i = 0; i < num_runs; ++i) {
        timer.start();
        T*     result  = flashck::layer_norm_fwd(input.get(), gamma.get(), beta.get(), m, n);
        double time_ms = timer.stop_ms();

        if (result != nullptr) {
            times.push_back(time_ms);
        }
    }

    if (times.empty()) {
        return {0.0, 0.0, 0.0, 0.0, m, n};
    }

    // Calculate statistics
    double min_time = *std::min_element(times.begin(), times.end());
    double max_time = *std::max_element(times.begin(), times.end());
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    // Calculate throughput (assume reading input + gamma + beta, writing output)
    double bytes_per_op    = sizeof(T) * (2 * m * n + 2 * n);  // input, output, gamma, beta
    double throughput_gb_s = (bytes_per_op / 1e9) / (min_time / 1000.0);

    return {min_time, max_time, avg_time, throughput_gb_s, m, n};
}

}  // anonymous namespace

class LayerNormBenchmarkTest: public ::testing::Test {
protected:
    void SetUp() override
    {
        std::cout << "\n=== LayerNorm Performance Benchmarks ===\n\n";
    }
};

TEST_F(LayerNormBenchmarkTest, SmallSequences)
{
    // Test typical small sequence lengths
    std::vector<std::pair<int, int>> configs = {
        {32, 768},   // Small batch, BERT-base hidden size
        {64, 768},   // Medium batch, BERT-base hidden size
        {128, 768},  // Large batch, BERT-base hidden size
    };

    std::cout << "Small Sequence Length Benchmarks:\n";
    for (auto [m, n] : configs) {
        auto result = benchmark_layer_norm<float>(m, n);
        result.print("LayerNorm Float32");
    }
}

TEST_F(LayerNormBenchmarkTest, LargeSequences)
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
        auto result = benchmark_layer_norm<float>(m, n);
        result.print("LayerNorm Float32");
    }
}

TEST_F(LayerNormBenchmarkTest, VaryingBatchSizes)
{
    const int        n           = 768;  // Fixed hidden dimension
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256};

    std::cout << "Varying Batch Size Benchmarks (n=" << n << "):\n";
    for (int m : batch_sizes) {
        auto result = benchmark_layer_norm<float>(m, n, 5, 2);  // Fewer runs for large batches
        result.print("BatchSize=" + std::to_string(m));
    }
}

TEST_F(LayerNormBenchmarkTest, VaryingHiddenSizes)
{
    const int        m            = 32;  // Fixed batch size
    std::vector<int> hidden_sizes = {256, 512, 768, 1024, 1536, 2048, 4096};

    std::cout << "Varying Hidden Size Benchmarks (m=" << m << "):\n";
    for (int n : hidden_sizes) {
        auto result = benchmark_layer_norm<float>(m, n, 5, 2);
        result.print("HiddenSize=" + std::to_string(n));
    }
}

TEST_F(LayerNormBenchmarkTest, MemoryIntensiveWorkloads)
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
        auto result = benchmark_layer_norm<float>(m, n, 3, 1);  // Fewer runs
        result.print("LayerNorm Float32");
    }
}

TEST_F(LayerNormBenchmarkTest, ConsistencyCheck)
{
    // Test multiple runs for consistency
    const int m = 64, n = 768;
    const int num_tests = 5;

    std::cout << "Consistency Check (multiple runs):\n";
    std::vector<double> avg_times;

    for (int i = 0; i < num_tests; ++i) {
        auto result = benchmark_layer_norm<float>(m, n, 10, 3);
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

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
