#include "flashck_wrapper.h"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <vector>

class LayerNormDynamicTest: public ::testing::Test {
protected:
    void SetUp() override
    {
        batch_size  = 4;
        seq_len     = 8;
        hidden_size = 16;
        epsilon     = 1e-5f;

        // Set up dynamic range
        min_seq_len = 4;
        max_seq_len = 16;
        m_range     = {min_seq_len, max_seq_len};

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

    int                batch_size, seq_len, hidden_size;
    int                min_seq_len, max_seq_len;
    float              epsilon;
    std::vector<int>   m_range;
    std::vector<float> input_data, gamma, beta;
};

TEST_F(LayerNormDynamicTest, BasicFunctionality)
{
    EXPECT_TRUE(flashck::wrapper::is_available());

    // Test dynamic layer normalization
    float* result = flashck::layer_norm_fwd_dynamic<float>(
        input_data.data(), gamma.data(), beta.data(), m_range, batch_size * seq_len, hidden_size, hidden_size, epsilon);

    ASSERT_NE(result, nullptr);

    // Basic sanity check: output should be finite
    for (int i = 0; i < batch_size * seq_len * hidden_size; ++i) {
        EXPECT_TRUE(std::isfinite(result[i])) << "Non-finite result at index " << i;
    }
}

TEST_F(LayerNormDynamicTest, DifferentSequenceLengths)
{
    std::vector<int> test_seq_lens = {4, 6, 8, 12, 16};

    for (int test_seq_len : test_seq_lens) {
        if (test_seq_len < min_seq_len || test_seq_len > max_seq_len) {
            continue;  // Skip out of range
        }

        std::vector<float> test_input(batch_size * test_seq_len * hidden_size);

        // Initialize with test pattern
        for (int i = 0; i < test_input.size(); ++i) {
            test_input[i] = static_cast<float>(i % 10) * 0.1f;
        }

        float* result = flashck::layer_norm_fwd_dynamic<float>(test_input.data(),
                                                               gamma.data(),
                                                               beta.data(),
                                                               m_range,
                                                               batch_size * test_seq_len,
                                                               hidden_size,
                                                               hidden_size,
                                                               epsilon);

        ASSERT_NE(result, nullptr) << "Failed for sequence length " << test_seq_len;

        // Verify finite output
        for (int i = 0; i < batch_size * test_seq_len * hidden_size; ++i) {
            EXPECT_TRUE(std::isfinite(result[i]))
                << "Non-finite result at index " << i << " for sequence length " << test_seq_len;
        }
    }
}

TEST_F(LayerNormDynamicTest, CompareWithStatic)
{
    // Test that dynamic version produces same results as static version
    // for the same input

    float* static_result = flashck::layer_norm_fwd_static<float>(
        input_data.data(), gamma.data(), beta.data(), batch_size * seq_len, hidden_size, hidden_size, epsilon);

    float* dynamic_result = flashck::layer_norm_fwd_dynamic<float>(
        input_data.data(), gamma.data(), beta.data(), m_range, batch_size * seq_len, hidden_size, hidden_size, epsilon);

    ASSERT_NE(static_result, nullptr);
    ASSERT_NE(dynamic_result, nullptr);

    // Compare results with some tolerance
    const float tolerance = 1e-5f;
    for (int i = 0; i < batch_size * seq_len * hidden_size; ++i) {
        EXPECT_NEAR(static_result[i], dynamic_result[i], tolerance) << "Mismatch at index " << i;
    }
}
