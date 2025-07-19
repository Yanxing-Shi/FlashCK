/**
 * @file test_layer_norm_correctness.cpp
 * @brief Correctness tests for the header-only LayerNorm wrapper
 */

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <vector>

#include "flashck/wrapper/cpp/norm/layer_norm.h"

namespace {

// Reference CPU implementation for validation
template<typename T>
void reference_layer_norm(const T* input, const T* gamma, const T* beta, T* output, int m, int n, float epsilon = 1e-5f)
{
    for (int i = 0; i < m; ++i) {
        const T* row     = input + i * n;
        T*       out_row = output + i * n;

        // Compute mean
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += static_cast<double>(row[j]);
        }
        double mean = sum / n;

        // Compute variance
        double var_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            double diff = static_cast<double>(row[j]) - mean;
            var_sum += diff * diff;
        }
        double variance = var_sum / n;
        double inv_std  = 1.0 / std::sqrt(variance + epsilon);

        // Apply normalization
        for (int j = 0; j < n; ++j) {
            double normalized = (static_cast<double>(row[j]) - mean) * inv_std;
            out_row[j] = static_cast<T>(static_cast<double>(gamma[j]) * normalized + static_cast<double>(beta[j]));
        }
    }
}

// Helper function to generate random data
template<typename T>
void generate_random_data(T* data, int size, std::mt19937& gen, float min_val = -2.0f, float max_val = 2.0f)
{
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<T>(dist(gen));
    }
}

// Helper function to compare tensors with tolerance
template<typename T>
bool compare_tensors(const T* a, const T* b, int size, float rtol = 1e-3f, float atol = 1e-5f)
{
    for (int i = 0; i < size; ++i) {
        float diff      = std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
        float threshold = atol + rtol * std::abs(static_cast<float>(b[i]));
        if (diff > threshold) {
            return false;
        }
    }
    return true;
}

}  // anonymous namespace

class LayerNormCorrectnessTest: public ::testing::Test {
protected:
    void SetUp() override
    {
        gen.seed(42);  // Fixed seed for reproducibility
    }

    std::mt19937 gen;
};

TEST_F(LayerNormCorrectnessTest, BasicFloat32Test)
{
    const int   m = 32, n = 768;
    const float epsilon = 1e-5f;

    // Allocate memory
    auto input    = std::make_unique<float[]>(m * n);
    auto gamma    = std::make_unique<float[]>(n);
    auto beta     = std::make_unique<float[]>(n);
    auto expected = std::make_unique<float[]>(m * n);

    // Generate test data
    generate_random_data(input.get(), m * n, gen);
    generate_random_data(gamma.get(), n, gen, 0.5f, 1.5f);
    generate_random_data(beta.get(), n, gen, -0.5f, 0.5f);

    // Compute reference result
    reference_layer_norm(input.get(), gamma.get(), beta.get(), expected.get(), m, n, epsilon);

    // Test the wrapper
    float* result = nullptr;
    ASSERT_NO_THROW({ result = flashck::layer_norm_fwd(input.get(), gamma.get(), beta.get(), m, n, epsilon); });

    ASSERT_NE(result, nullptr);

    // Compare results
    EXPECT_TRUE(compare_tensors(result, expected.get(), m * n, 1e-3f, 1e-4f))
        << "LayerNorm output does not match reference implementation";
}

TEST_F(LayerNormCorrectnessTest, SmallDimensionsTest)
{
    const int   m = 4, n = 8;
    const float epsilon = 1e-5f;

    auto input    = std::make_unique<float[]>(m * n);
    auto gamma    = std::make_unique<float[]>(n);
    auto beta     = std::make_unique<float[]>(n);
    auto expected = std::make_unique<float[]>(m * n);

    // Generate test data
    generate_random_data(input.get(), m * n, gen);
    std::fill(gamma.get(), gamma.get() + n, 1.0f);
    std::fill(beta.get(), beta.get() + n, 0.0f);

    // Compute reference
    reference_layer_norm(input.get(), gamma.get(), beta.get(), expected.get(), m, n, epsilon);

    // Test wrapper
    float* result = flashck::layer_norm_fwd(input.get(), gamma.get(), beta.get(), m, n, epsilon);

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(compare_tensors(result, expected.get(), m * n, 1e-3f, 1e-4f));
}

TEST_F(LayerNormCorrectnessTest, LargeDimensionsTest)
{
    const int   m = 128, n = 4096;
    const float epsilon = 1e-5f;

    auto input    = std::make_unique<float[]>(m * n);
    auto gamma    = std::make_unique<float[]>(n);
    auto beta     = std::make_unique<float[]>(n);
    auto expected = std::make_unique<float[]>(m * n);

    // Generate test data
    generate_random_data(input.get(), m * n, gen);
    generate_random_data(gamma.get(), n, gen, 0.8f, 1.2f);
    generate_random_data(beta.get(), n, gen, -0.1f, 0.1f);

    // Compute reference
    reference_layer_norm(input.get(), gamma.get(), beta.get(), expected.get(), m, n, epsilon);

    // Test wrapper
    float* result = flashck::layer_norm_fwd(input.get(), gamma.get(), beta.get(), m, n, epsilon);

    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(compare_tensors(result, expected.get(), m * n, 1e-3f, 1e-4f));
}

TEST_F(LayerNormCorrectnessTest, ErrorHandlingTest)
{
    const int m = 32, n = 768;
    auto      input = std::make_unique<float[]>(m * n);
    auto      gamma = std::make_unique<float[]>(n);
    auto      beta  = std::make_unique<float[]>(n);

    // Test null pointer inputs
    EXPECT_THROW(flashck::layer_norm_fwd<float>(nullptr, gamma.get(), beta.get(), m, n), std::runtime_error);
    EXPECT_THROW(flashck::layer_norm_fwd<float>(input.get(), nullptr, beta.get(), m, n), std::runtime_error);
    EXPECT_THROW(flashck::layer_norm_fwd<float>(input.get(), gamma.get(), nullptr, m, n), std::runtime_error);

    // Test invalid dimensions
    EXPECT_THROW(flashck::layer_norm_fwd<float>(input.get(), gamma.get(), beta.get(), 0, n), std::runtime_error);
    EXPECT_THROW(flashck::layer_norm_fwd<float>(input.get(), gamma.get(), beta.get(), m, 0), std::runtime_error);
    EXPECT_THROW(flashck::layer_norm_fwd<float>(input.get(), gamma.get(), beta.get(), -1, n), std::runtime_error);
}

TEST_F(LayerNormCorrectnessTest, BackwardCompatibilityTest)
{
    const int   m = 16, n = 64;
    const float epsilon = 1e-5f;

    auto input = std::make_unique<float[]>(m * n);
    auto gamma = std::make_unique<float[]>(n);
    auto beta  = std::make_unique<float[]>(n);

    generate_random_data(input.get(), m * n, gen);
    generate_random_data(gamma.get(), n, gen, 0.5f, 1.5f);
    generate_random_data(beta.get(), n, gen, -0.5f, 0.5f);

    // Test both functions return same result
    float* result1 = flashck::layer_norm_fwd(input.get(), gamma.get(), beta.get(), m, n, epsilon);
    float* result2 = flashck::layer_norm_fwd_static(input.get(), gamma.get(), beta.get(), m, n, epsilon);

    ASSERT_NE(result1, nullptr);
    ASSERT_NE(result2, nullptr);
    EXPECT_TRUE(compare_tensors(result1, result2, m * n, 1e-6f, 1e-8f));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
