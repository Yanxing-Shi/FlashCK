#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "flashck_wrapper.h"

class LayerNormTest: public ::testing::Test {
protected:
    void SetUp() override
    {
        batch_size  = 4;
        seq_len     = 8;
        hidden_size = 16;
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
            val = 1.0f;  // Standard weight initialization
        }

        for (auto& val : beta) {
            val = 0.0f;  // Standard bias initialization
        }
    }

    void TearDown() override
    {
        // Cleanup if needed
    }

    // Helper function to compute reference layer norm
    std::vector<float> compute_reference_layer_norm(const std::vector<float>& input,
                                                    const std::vector<float>& weight,
                                                    const std::vector<float>& bias,
                                                    int                       m,
                                                    int                       n,
                                                    float                     eps)
    {

        std::vector<float> output(m * n);

        for (int i = 0; i < m; ++i) {
            // Compute mean
            float mean = 0.0f;
            for (int j = 0; j < n; ++j) {
                mean += input[i * n + j];
            }
            mean /= n;

            // Compute variance
            float variance = 0.0f;
            for (int j = 0; j < n; ++j) {
                float diff = input[i * n + j] - mean;
                variance += diff * diff;
            }
            variance /= n;

            // Normalize
            float inv_std = 1.0f / std::sqrt(variance + eps);
            for (int j = 0; j < n; ++j) {
                float normalized  = (input[i * n + j] - mean) * inv_std;
                output[i * n + j] = normalized * weight[j] + bias[j];
            }
        }

        return output;
    }

    int                batch_size, seq_len, hidden_size;
    float              epsilon;
    std::vector<float> input_data, gamma, beta;
};

TEST_F(LayerNormTest, BasicFunctionality)
{
    EXPECT_TRUE(flashck::wrapper::is_available());

    // Test static layer normalization
    float* result = flashck::layer_norm_fwd_static<float>(
        input_data.data(), gamma.data(), beta.data(), batch_size * seq_len, hidden_size, hidden_size, epsilon);

    ASSERT_NE(result, nullptr);

    // Compute reference result
    auto reference = compute_reference_layer_norm(input_data, gamma, beta, batch_size * seq_len, hidden_size, epsilon);

    // Compare results (with some tolerance for floating point errors)
    const float tolerance = 1e-4f;
    for (int i = 0; i < batch_size * seq_len * hidden_size; ++i) {
        EXPECT_NEAR(result[i], reference[i], tolerance) << "Mismatch at index " << i;
    }
}

TEST_F(LayerNormTest, DifferentSizes)
{
    std::vector<std::tuple<int, int, int>> test_cases = {
        {1, 1, 4},    // Minimal case
        {2, 4, 8},    // Small case
        {4, 8, 16},   // Medium case
        {8, 16, 32},  // Larger case
    };

    for (auto [bs, sl, hs] : test_cases) {
        std::vector<float> test_input(bs * sl * hs, 1.0f);
        std::vector<float> test_gamma(hs, 1.0f);
        std::vector<float> test_beta(hs, 0.0f);

        // Initialize with some test pattern
        for (int i = 0; i < test_input.size(); ++i) {
            test_input[i] = static_cast<float>(i % 10) * 0.1f;
        }

        float* result = flashck::layer_norm_fwd_static<float>(
            test_input.data(), test_gamma.data(), test_beta.data(), bs * sl, hs, hs, epsilon);

        ASSERT_NE(result, nullptr) << "Failed for size " << bs << "x" << sl << "x" << hs;

        // Basic sanity check: output should be finite
        for (int i = 0; i < bs * sl * hs; ++i) {
            EXPECT_TRUE(std::isfinite(result[i]))
                << "Non-finite result at index " << i << " for size " << bs << "x" << sl << "x" << hs;
        }
    }
}

TEST_F(LayerNormTest, ZeroInput)
{
    std::vector<float> zero_input(batch_size * seq_len * hidden_size, 0.0f);

    float* result = flashck::layer_norm_fwd_static<float>(
        zero_input.data(), gamma.data(), beta.data(), batch_size * seq_len, hidden_size, hidden_size, epsilon);

    ASSERT_NE(result, nullptr);

    // For zero input, normalized output should be zero, so result should equal bias
    const float tolerance = 1e-5f;
    for (int i = 0; i < batch_size * seq_len; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            EXPECT_NEAR(result[i * hidden_size + j], beta[j], tolerance)
                << "Mismatch at position [" << i << "][" << j << "]";
        }
    }
}

TEST_F(LayerNormTest, IdentityTransform)
{
    // Test with gamma=1, beta=0 (identity transform)
    std::fill(gamma.begin(), gamma.end(), 1.0f);
    std::fill(beta.begin(), beta.end(), 0.0f);

    float* result = flashck::layer_norm_fwd_static<float>(
        input_data.data(), gamma.data(), beta.data(), batch_size * seq_len, hidden_size, hidden_size, epsilon);

    ASSERT_NE(result, nullptr);

    // Verify that each sequence is normalized (mean ≈ 0, variance ≈ 1)
    const float tolerance = 1e-4f;
    for (int i = 0; i < batch_size * seq_len; ++i) {
        float mean     = 0.0f;
        float variance = 0.0f;

        // Compute mean
        for (int j = 0; j < hidden_size; ++j) {
            mean += result[i * hidden_size + j];
        }
        mean /= hidden_size;

        // Compute variance
        for (int j = 0; j < hidden_size; ++j) {
            float diff = result[i * hidden_size + j] - mean;
            variance += diff * diff;
        }
        variance /= hidden_size;

        EXPECT_NEAR(mean, 0.0f, tolerance) << "Mean not close to 0 for sequence " << i;
        EXPECT_NEAR(variance, 1.0f, tolerance) << "Variance not close to 1 for sequence " << i;
    }
}
