#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <vector>

#include "flashck/wrapper/cpp/norm/rms_norm.h"

namespace {

// Reference CPU implementation for validation
template<typename T>
void reference_rms_norm(const T* input, const T* gamma, T* output, int m, int n, float epsilon = 1e-5f)
{
    for (int i = 0; i < m; ++i) {
        const T* row     = input + i * n;
        T*       out_row = output + i * n;

        // Compute RMS (Root Mean Square)
        double sum_sq = 0.0;
        for (int j = 0; j < n; ++j) {
            double val = static_cast<double>(row[j]);
            sum_sq += val * val;
        }
        double rms     = std::sqrt(sum_sq / n + epsilon);
        double inv_rms = 1.0 / rms;

        // Apply normalization
        for (int j = 0; j < n; ++j) {
            double normalized = static_cast<double>(row[j]) * inv_rms;
            out_row[j]        = static_cast<T>(static_cast<double>(gamma[j]) * normalized);
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

// Helper function to allocate and copy data to GPU
template<typename T>
T* allocate_and_copy_to_gpu(const T* host_data, int size)
{
    T* gpu_data;
    hipMalloc(&gpu_data, size * sizeof(T));
    hipMemcpy(gpu_data, host_data, size * sizeof(T), hipMemcpyHostToDevice);
    return gpu_data;
}

// Helper function to copy data from GPU and free GPU memory
template<typename T>
void copy_from_gpu_and_free(T* host_data, T* gpu_data, int size)
{
    hipMemcpy(host_data, gpu_data, size * sizeof(T), hipMemcpyDeviceToHost);
    hipFree(gpu_data);
}

// Test fixture for RMSNorm correctness tests
class RMSNormCorrectnessTest: public ::testing::Test {
protected:
    void SetUp() override
    {
        gen_.seed(42);  // Fixed seed for reproducibility
    }

    void TearDown() override {}

    std::mt19937 gen_;
};

// Test basic correctness with small dimensions
TEST_F(RMSNormCorrectnessTest, SmallDimensions)
{
    const int   m       = 4;
    const int   n       = 8;
    const float epsilon = 1e-5f;

    // Allocate host memory
    auto input_host  = std::make_unique<float[]>(m * n);
    auto gamma_host  = std::make_unique<float[]>(n);
    auto output_host = std::make_unique<float[]>(m * n);
    auto ref_output  = std::make_unique<float[]>(m * n);

    // Generate test data
    generate_random_data(input_host.get(), m * n, gen_);
    generate_random_data(gamma_host.get(), n, gen_, 0.5f, 1.5f);

    // Compute reference result
    reference_rms_norm(input_host.get(), gamma_host.get(), ref_output.get(), m, n, epsilon);

    // Allocate GPU memory and copy data
    float* input_gpu = allocate_and_copy_to_gpu(input_host.get(), m * n);
    float* gamma_gpu = allocate_and_copy_to_gpu(gamma_host.get(), n);

    // Execute RMSNorm
    float* output_gpu = flashck::rms_norm_fwd(input_gpu, gamma_gpu, m, n, epsilon);
    ASSERT_NE(output_gpu, nullptr) << "RMSNorm execution failed";

    // Copy result back to host
    copy_from_gpu_and_free(output_host.get(), output_gpu, m * n);

    // Verify results with tolerance
    const float tolerance = 1e-4f;
    for (int i = 0; i < m * n; ++i) {
        EXPECT_NEAR(output_host[i], ref_output[i], tolerance)
            << "Mismatch at index " << i << ": got " << output_host[i] << ", expected " << ref_output[i];
    }

    // Clean up
    hipFree(input_gpu);
    hipFree(gamma_gpu);
}

// Test with larger, realistic dimensions
TEST_F(RMSNormCorrectnessTest, RealisticDimensions)
{
    const int   m       = 32;
    const int   n       = 768;
    const float epsilon = 1e-5f;

    // Allocate host memory
    auto input_host  = std::make_unique<float[]>(m * n);
    auto gamma_host  = std::make_unique<float[]>(n);
    auto output_host = std::make_unique<float[]>(m * n);
    auto ref_output  = std::make_unique<float[]>(m * n);

    // Generate test data
    generate_random_data(input_host.get(), m * n, gen_);
    generate_random_data(gamma_host.get(), n, gen_, 0.8f, 1.2f);

    // Compute reference result
    reference_rms_norm(input_host.get(), gamma_host.get(), ref_output.get(), m, n, epsilon);

    // Allocate GPU memory and copy data
    float* input_gpu = allocate_and_copy_to_gpu(input_host.get(), m * n);
    float* gamma_gpu = allocate_and_copy_to_gpu(gamma_host.get(), n);

    // Execute RMSNorm
    float* output_gpu = flashck::rms_norm_fwd(input_gpu, gamma_gpu, m, n, epsilon);
    ASSERT_NE(output_gpu, nullptr) << "RMSNorm execution failed";

    // Copy result back to host
    copy_from_gpu_and_free(output_host.get(), output_gpu, m * n);

    // Verify results with tolerance (slightly higher for larger tensors)
    const float tolerance = 2e-4f;
    for (int i = 0; i < m * n; ++i) {
        EXPECT_NEAR(output_host[i], ref_output[i], tolerance)
            << "Mismatch at index " << i << ": got " << output_host[i] << ", expected " << ref_output[i];
    }

    // Clean up
    hipFree(input_gpu);
    hipFree(gamma_gpu);
}

// Test with different epsilon values
TEST_F(RMSNormCorrectnessTest, DifferentEpsilon)
{
    const int m = 8;
    const int n = 16;

    // Test different epsilon values
    std::vector<float> epsilons = {1e-6f, 1e-5f, 1e-4f, 1e-3f};

    for (float epsilon : epsilons) {
        // Allocate host memory
        auto input_host  = std::make_unique<float[]>(m * n);
        auto gamma_host  = std::make_unique<float[]>(n);
        auto output_host = std::make_unique<float[]>(m * n);
        auto ref_output  = std::make_unique<float[]>(m * n);

        // Generate test data
        generate_random_data(input_host.get(), m * n, gen_);
        generate_random_data(gamma_host.get(), n, gen_, 0.9f, 1.1f);

        // Compute reference result
        reference_rms_norm(input_host.get(), gamma_host.get(), ref_output.get(), m, n, epsilon);

        // Allocate GPU memory and copy data
        float* input_gpu = allocate_and_copy_to_gpu(input_host.get(), m * n);
        float* gamma_gpu = allocate_and_copy_to_gpu(gamma_host.get(), n);

        // Execute RMSNorm
        float* output_gpu = flashck::rms_norm_fwd(input_gpu, gamma_gpu, m, n, epsilon);
        ASSERT_NE(output_gpu, nullptr) << "RMSNorm execution failed for epsilon = " << epsilon;

        // Copy result back to host
        copy_from_gpu_and_free(output_host.get(), output_gpu, m * n);

        // Verify results
        const float tolerance = 1e-4f;
        for (int i = 0; i < m * n; ++i) {
            EXPECT_NEAR(output_host[i], ref_output[i], tolerance)
                << "Mismatch at index " << i << " for epsilon = " << epsilon << ": got " << output_host[i]
                << ", expected " << ref_output[i];
        }

        // Clean up
        hipFree(input_gpu);
        hipFree(gamma_gpu);
    }
}

// Test error handling for null pointers
TEST_F(RMSNormCorrectnessTest, ErrorHandling)
{
    const int m = 4;
    const int n = 8;

    // Test null input
    EXPECT_THROW(flashck::rms_norm_fwd<float>(nullptr, nullptr, m, n), std::runtime_error);

    // Allocate valid gamma for further tests
    auto gamma_host = std::make_unique<float[]>(n);
    generate_random_data(gamma_host.get(), n, gen_);
    float* gamma_gpu = allocate_and_copy_to_gpu(gamma_host.get(), n);

    // Test null gamma
    auto input_host = std::make_unique<float[]>(m * n);
    generate_random_data(input_host.get(), m * n, gen_);
    float* input_gpu = allocate_and_copy_to_gpu(input_host.get(), m * n);

    EXPECT_THROW(flashck::rms_norm_fwd<float>(input_gpu, nullptr, m, n), std::runtime_error);

    // Test invalid dimensions
    EXPECT_THROW(flashck::rms_norm_fwd<float>(input_gpu, gamma_gpu, 0, n), std::runtime_error);
    EXPECT_THROW(flashck::rms_norm_fwd<float>(input_gpu, gamma_gpu, m, 0), std::runtime_error);
    EXPECT_THROW(flashck::rms_norm_fwd<float>(input_gpu, gamma_gpu, -1, n), std::runtime_error);

    // Clean up
    hipFree(input_gpu);
    hipFree(gamma_gpu);
}

}  // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
