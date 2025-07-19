#include "flashck_wrapper.h"
#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <vector>

class LayerNormBenchmark: public ::testing::Test {
protected:
    void SetUp() override
    {
        // Setup larger tensors for benchmarking
        batch_size  = 32;
        seq_len     = 128;
        hidden_size = 768;
        epsilon     = 1e-5f;

        // Initialize test data
        std::random_device                    rd;
        std::mt19937                          gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        input_data.resize(batch_size * seq_len * hidden_size);
        gamma.resize(hidden_size);
        beta.resize(hidden_size);

        for (auto& val : input_data) {
            val = dis(gen);
        }

        for (auto& val : gamma) {
            val = 1.0f;
        }

        for (auto& val : beta) {
            val = 0.0f;
        }
    }

    void TearDown() override
    {
        // Cleanup if needed
    }

    int                batch_size, seq_len, hidden_size;
    float              epsilon;
    std::vector<float> input_data, gamma, beta;
};

TEST_F(LayerNormBenchmark, StaticPerformance)
{
    EXPECT_TRUE(flashck::wrapper::is_available());

    const int           num_iterations = 10;
    std::vector<double> times;

    for (int i = 0; i < num_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        float* result = flashck::layer_norm_fwd_static<float>(
            input_data.data(), gamma.data(), beta.data(), batch_size * seq_len, hidden_size, hidden_size, epsilon);

        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        ASSERT_NE(result, nullptr);
        times.push_back(duration.count());
    }

    // Calculate statistics
    double total_time = 0.0;
    for (double time : times) {
        total_time += time;
    }
    double avg_time = total_time / num_iterations;

    std::cout << "Static LayerNorm Performance:" << std::endl;
    std::cout << "  Input shape: [" << batch_size << ", " << seq_len << ", " << hidden_size << "]" << std::endl;
    std::cout << "  Average time: " << avg_time << " μs" << std::endl;
    std::cout << "  Throughput: " << (batch_size * seq_len * hidden_size) / avg_time << " elements/μs" << std::endl;

    // Performance should be reasonable (this is a basic sanity check)
    EXPECT_LT(avg_time, 10000.0) << "Performance seems too slow";
}

TEST_F(LayerNormBenchmark, DynamicPerformance)
{
    EXPECT_TRUE(flashck::wrapper::is_available());

    std::vector<int>    m_range        = {64, 256};
    const int           num_iterations = 10;
    std::vector<double> times;

    for (int i = 0; i < num_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        float* result = flashck::layer_norm_fwd_dynamic<float>(input_data.data(),
                                                               gamma.data(),
                                                               beta.data(),
                                                               m_range,
                                                               batch_size * seq_len,
                                                               hidden_size,
                                                               hidden_size,
                                                               epsilon);

        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        ASSERT_NE(result, nullptr);
        times.push_back(duration.count());
    }

    // Calculate statistics
    double total_time = 0.0;
    for (double time : times) {
        total_time += time;
    }
    double avg_time = total_time / num_iterations;

    std::cout << "Dynamic LayerNorm Performance:" << std::endl;
    std::cout << "  Input shape: [" << batch_size << ", " << seq_len << ", " << hidden_size << "]" << std::endl;
    std::cout << "  Dynamic range: [" << m_range[0] << ", " << m_range[1] << "]" << std::endl;
    std::cout << "  Average time: " << avg_time << " μs" << std::endl;
    std::cout << "  Throughput: " << (batch_size * seq_len * hidden_size) / avg_time << " elements/μs" << std::endl;

    // Performance should be reasonable
    EXPECT_LT(avg_time, 10000.0) << "Performance seems too slow";
}

TEST_F(LayerNormBenchmark, ScalabilityTest)
{
    EXPECT_TRUE(flashck::wrapper::is_available());

    std::vector<std::tuple<int, int, int>> test_sizes = {
        {8, 32, 256},     // Small
        {16, 64, 512},    // Medium
        {32, 128, 768},   // Large
        {64, 256, 1024},  // Extra large
    };

    std::cout << "LayerNorm Scalability Test:" << std::endl;
    std::cout << "Format: [batch_size, seq_len, hidden_size] -> avg_time (μs)" << std::endl;

    for (auto [bs, sl, hs] : test_sizes) {
        std::vector<float> test_input(bs * sl * hs);
        std::vector<float> test_gamma(hs, 1.0f);
        std::vector<float> test_beta(hs, 0.0f);

        // Initialize with test pattern
        for (int i = 0; i < test_input.size(); ++i) {
            test_input[i] = static_cast<float>(i % 100) * 0.01f;
        }

        const int           num_iterations = 5;
        std::vector<double> times;

        for (int i = 0; i < num_iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();

            float* result = flashck::layer_norm_fwd_static<float>(
                test_input.data(), test_gamma.data(), test_beta.data(), bs * sl, hs, hs, epsilon);

            auto end      = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            ASSERT_NE(result, nullptr);
            times.push_back(duration.count());
        }

        double total_time = 0.0;
        for (double time : times) {
            total_time += time;
        }
        double avg_time = total_time / num_iterations;

        std::cout << "  [" << bs << ", " << sl << ", " << hs << "] -> " << avg_time << " μs" << std::endl;

        // Basic sanity check
        EXPECT_LT(avg_time, 50000.0) << "Performance too slow for size " << bs << "x" << sl << "x" << hs;
    }
}
