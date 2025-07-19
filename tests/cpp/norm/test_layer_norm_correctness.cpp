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

// Helper function to allocate and copy data to GPU
template<typename T>
T* allocate_and_copy_to_gpu(const T* host_data, int size)
{
    T* gpu_data;
    HIP_CHECK(hipMalloc(&gpu_data, size * sizeof(T)));
    HIP_CHECK(hipMemcpy(gpu_data, host_data, size * sizeof(T), hipMemcpyHostToDevice));
    return gpu_data;
}

// Helper function to copy data from GPU and free GPU memory
template<typename T>
void copy_from_gpu_and_free(T* host_data, T* gpu_data, int size)
{
    HIP_CHECK(hipMemcpy(host_data, gpu_data, size * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(gpu_data));
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

    // Allocate CPU memory
    auto input_cpu    = std::make_unique<float[]>(m * n);
    auto gamma_cpu    = std::make_unique<float[]>(n);
    auto beta_cpu     = std::make_unique<float[]>(n);
    auto expected_cpu = std::make_unique<float[]>(m * n);
    auto output_cpu   = std::make_unique<float[]>(m * n);

    // Generate test data
    generate_random_data(input_cpu.get(), m * n, gen);
    generate_random_data(gamma_cpu.get(), n, gen, 0.5f, 1.5f);
    generate_random_data(beta_cpu.get(), n, gen, -0.5f, 0.5f);

    // Compute reference result
    reference_layer_norm(input_cpu.get(), gamma_cpu.get(), beta_cpu.get(), expected_cpu.get(), m, n, epsilon);

    // Allocate GPU memory and copy data
    float* input_gpu = allocate_and_copy_to_gpu(input_cpu.get(), m * n);
    float* gamma_gpu = allocate_and_copy_to_gpu(gamma_cpu.get(), n);
    float* beta_gpu  = allocate_and_copy_to_gpu(beta_cpu.get(), n);

    // Test the wrapper
    float* result_gpu = nullptr;
    ASSERT_NO_THROW({ result_gpu = flashck::layer_norm_fwd(input_gpu, gamma_gpu, beta_gpu, m, n, epsilon); });

    ASSERT_NE(result_gpu, nullptr);

    // Copy result back to CPU for comparison
    copy_from_gpu_and_free(output_cpu.get(), result_gpu, m * n);

    // Compare results
    EXPECT_TRUE(compare_tensors(output_cpu.get(), expected_cpu.get(), m * n, 1e-3f, 1e-4f))
        << "LayerNorm output does not match reference implementation";

    // Clean up GPU memory
    HIP_CHECK(hipFree(input_gpu));
    HIP_CHECK(hipFree(gamma_gpu));
    HIP_CHECK(hipFree(beta_gpu));
}

TEST_F(LayerNormCorrectnessTest, SmallDimensionsTest)
{
    const int   m = 4, n = 8;
    const float epsilon = 1e-5f;

    // Allocate CPU memory
    auto input_cpu    = std::make_unique<float[]>(m * n);
    auto gamma_cpu    = std::make_unique<float[]>(n);
    auto beta_cpu     = std::make_unique<float[]>(n);
    auto expected_cpu = std::make_unique<float[]>(m * n);
    auto output_cpu   = std::make_unique<float[]>(m * n);

    // Generate test data
    generate_random_data(input_cpu.get(), m * n, gen);
    std::fill(gamma_cpu.get(), gamma_cpu.get() + n, 1.0f);
    std::fill(beta_cpu.get(), beta_cpu.get() + n, 0.0f);

    // Compute reference
    reference_layer_norm(input_cpu.get(), gamma_cpu.get(), beta_cpu.get(), expected_cpu.get(), m, n, epsilon);

    // Allocate GPU memory and copy data
    float* input_gpu = allocate_and_copy_to_gpu(input_cpu.get(), m * n);
    float* gamma_gpu = allocate_and_copy_to_gpu(gamma_cpu.get(), n);
    float* beta_gpu  = allocate_and_copy_to_gpu(beta_cpu.get(), n);

    // Test wrapper
    float* result_gpu = flashck::layer_norm_fwd(input_gpu, gamma_gpu, beta_gpu, m, n, epsilon);

    ASSERT_NE(result_gpu, nullptr);

    // Copy result back to CPU for comparison
    copy_from_gpu_and_free(output_cpu.get(), result_gpu, m * n);

    EXPECT_TRUE(compare_tensors(output_cpu.get(), expected_cpu.get(), m * n, 1e-3f, 1e-4f));

    // Clean up GPU memory
    HIP_CHECK(hipFree(input_gpu));
    HIP_CHECK(hipFree(gamma_gpu));
    HIP_CHECK(hipFree(beta_gpu));
}

TEST_F(LayerNormCorrectnessTest, LargeDimensionsTest)
{
    const int   m = 128, n = 4096;
    const float epsilon = 1e-5f;

    // Allocate CPU memory
    auto input_cpu    = std::make_unique<float[]>(m * n);
    auto gamma_cpu    = std::make_unique<float[]>(n);
    auto beta_cpu     = std::make_unique<float[]>(n);
    auto expected_cpu = std::make_unique<float[]>(m * n);
    auto output_cpu   = std::make_unique<float[]>(m * n);

    // Generate test data
    generate_random_data(input_cpu.get(), m * n, gen);
    generate_random_data(gamma_cpu.get(), n, gen, 0.8f, 1.2f);
    generate_random_data(beta_cpu.get(), n, gen, -0.1f, 0.1f);

    // Compute reference
    reference_layer_norm(input_cpu.get(), gamma_cpu.get(), beta_cpu.get(), expected_cpu.get(), m, n, epsilon);

    // Allocate GPU memory and copy data
    float* input_gpu = allocate_and_copy_to_gpu(input_cpu.get(), m * n);
    float* gamma_gpu = allocate_and_copy_to_gpu(gamma_cpu.get(), n);
    float* beta_gpu  = allocate_and_copy_to_gpu(beta_cpu.get(), n);

    // Test wrapper
    float* result_gpu = flashck::layer_norm_fwd(input_gpu, gamma_gpu, beta_gpu, m, n, epsilon);

    ASSERT_NE(result_gpu, nullptr);

    // Copy result back to CPU for comparison
    copy_from_gpu_and_free(output_cpu.get(), result_gpu, m * n);

    EXPECT_TRUE(compare_tensors(output_cpu.get(), expected_cpu.get(), m * n, 1e-3f, 1e-4f));

    // Clean up GPU memory
    HIP_CHECK(hipFree(input_gpu));
    HIP_CHECK(hipFree(gamma_gpu));
    HIP_CHECK(hipFree(beta_gpu));
}

TEST_F(LayerNormCorrectnessTest, ErrorHandlingTest)
{
    const int m = 32, n = 768;

    // Allocate CPU memory
    auto input_cpu = std::make_unique<float[]>(m * n);
    auto gamma_cpu = std::make_unique<float[]>(n);
    auto beta_cpu  = std::make_unique<float[]>(n);

    // Generate test data
    generate_random_data(input_cpu.get(), m * n, gen);
    generate_random_data(gamma_cpu.get(), n, gen);
    generate_random_data(beta_cpu.get(), n, gen);

    // Allocate GPU memory
    float* input_gpu = allocate_and_copy_to_gpu(input_cpu.get(), m * n);
    float* gamma_gpu = allocate_and_copy_to_gpu(gamma_cpu.get(), n);
    float* beta_gpu  = allocate_and_copy_to_gpu(beta_cpu.get(), n);

    // Test null pointer inputs
    EXPECT_THROW(flashck::layer_norm_fwd<float>(nullptr, gamma_gpu, beta_gpu, m, n), std::runtime_error);
    EXPECT_THROW(flashck::layer_norm_fwd<float>(input_gpu, nullptr, beta_gpu, m, n), std::runtime_error);
    EXPECT_THROW(flashck::layer_norm_fwd<float>(input_gpu, gamma_gpu, nullptr, m, n), std::runtime_error);

    // Test invalid dimensions
    EXPECT_THROW(flashck::layer_norm_fwd<float>(input_gpu, gamma_gpu, beta_gpu, 0, n), std::runtime_error);
    EXPECT_THROW(flashck::layer_norm_fwd<float>(input_gpu, gamma_gpu, beta_gpu, m, 0), std::runtime_error);
    EXPECT_THROW(flashck::layer_norm_fwd<float>(input_gpu, gamma_gpu, beta_gpu, -1, n), std::runtime_error);

    // Clean up GPU memory
    HIP_CHECK(hipFree(input_gpu));
    HIP_CHECK(hipFree(gamma_gpu));
    HIP_CHECK(hipFree(beta_gpu));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
