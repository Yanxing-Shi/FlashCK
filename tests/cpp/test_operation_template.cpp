/**
 * @file test_operation_template.cpp
 * @brief Template for adding new operations to the unified test framework
 *
 * This file serves as a template for adding new operations (FMHA, GEMM, etc.)
 * to the unified test framework. Copy and modify this file for new operations.
 *
 * Steps to adapt this template:
 * 1. Replace "Operation" with your operation name (e.g., "Gemm", "FlashAttention")
 * 2. Implement the configuration class (see op_test_configs.h for examples)
 * 3. Implement the reference implementation
 * 4. Create the FlashCK wrapper function
 * 5. Update the test names and parameters
 */

#include <gtest/gtest.h>
#include <memory>

// Include your operation's header here
// #include "wrapper/cpp/your_operation/your_operation.h"

// Unified test framework
#include "../common/op_test_configs.h"  // For GEMM/FlashAttention examples
#include "../common/test_framework.h"

using namespace flashck;
using namespace flashck::test;

// Test fixture for your operation
class OperationUnifiedTestFloat: public UnifiedTestSuite<float> {};

/**
 * @brief Example: FlashCK GEMM wrapper for the unified test framework
 * Replace this with your operation's wrapper
 */
template<typename T>
T* run_flashck_gemm_example(const GemmConfig<T>& config, GpuMemoryManager<T>& gpu_mem)
{
    try {
        // Replace this with your actual FlashCK operation call
        // T* result = your_operation_fwd(
        //     config.gpu_input1(),
        //     config.gpu_input2(),
        //     /* other parameters */
        // );
        //
        // For now, return nullptr to indicate not implemented
        std::cerr << "GEMM operation not implemented yet" << std::endl;
        return nullptr;
    }
    catch (const std::exception& e) {
        std::cerr << "Operation execution failed: " << e.what() << std::endl;
        return nullptr;
    }
}

/**
 * @brief Example: Reference GEMM implementation
 * Replace this with your operation's reference implementation
 */
template<typename T>
class GemmReferenceExample {
public:
    static void forward(const GemmConfig<T>& config, T* output)
    {
        // Implement your reference CPU implementation here
        // For GEMM: C = alpha * A * B + beta * C

        const int m     = config.m();
        const int n     = config.n();
        const int k     = config.k();
        const T   alpha = config.alpha();
        const T   beta  = config.beta();

        const T* A = config.a_data();
        const T* B = config.b_data();
        const T* C = config.c_data();

        // Simple reference GEMM implementation (not optimized)
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                T sum = static_cast<T>(0);
                for (int l = 0; l < k; ++l) {
                    sum += A[i * k + l] * B[l * n + j];
                }
                output[i * n + j] = alpha * sum + beta * C[i * n + j];
            }
        }
    }
};

/**
 * @brief Correctness test example
 * Modify this for your operation
 */
TEST_F(OperationUnifiedTestFloat, GemmCorrectnessTest)
{
    // Create test configurations (adapt for your operation)
    auto configs = OpTestConfigFactory<float>::create_gemm_configs();

    // Reference implementation wrapper
    auto reference_impl = [](const GemmConfig<float>& config, float* output) {
        GemmReferenceExample<float>::forward(config, output);
    };

    // FlashCK implementation wrapper
    auto flashck_impl = [](const GemmConfig<float>& config, GpuMemoryManager<float>& gpu_mem) -> float* {
        return run_flashck_gemm_example(config, gpu_mem);
    };

    // Run correctness test (adjust tolerances as needed)
    run_correctness_test(configs, reference_impl, flashck_impl, 1e-3f, 1e-4f, true);
}

/**
 * @brief Performance test example
 * Modify this for your operation
 */
TEST_F(OperationUnifiedTestFloat, GemmPerformanceTest)
{
    // Create test configurations for performance testing
    std::vector<std::shared_ptr<GemmConfig<float>>> perf_configs;

    // Add various sizes for performance comparison (adapt for your operation)
    perf_configs.push_back(std::make_shared<GemmConfig<float>>(64, 64, 64, false, false, 1.0f, 0.0f, "Small_64x64x64"));
    perf_configs.push_back(
        std::make_shared<GemmConfig<float>>(128, 128, 128, false, false, 1.0f, 0.0f, "Medium_128x128x128"));
    perf_configs.push_back(
        std::make_shared<GemmConfig<float>>(256, 256, 256, false, false, 1.0f, 0.0f, "Large_256x256x256"));
    perf_configs.push_back(
        std::make_shared<GemmConfig<float>>(512, 512, 512, false, false, 1.0f, 0.0f, "XLarge_512x512x512"));

    // FlashCK implementation wrapper
    auto flashck_impl = [](const GemmConfig<float>& config, GpuMemoryManager<float>& gpu_mem) -> float* {
        return run_flashck_gemm_example(config, gpu_mem);
    };

    // Run performance test
    auto results = run_performance_test(perf_configs, flashck_impl, 20, 5, true);

    // Verify we got results (will be 0 until operation is implemented)
    // EXPECT_GT(results.size(), 0) << "No performance results obtained";

    // Print best performer (when implemented)
    if (!results.empty()) {
        std::cout << "\nBest GEMM performance: " << results[0].config_name << " with " << results[0].throughput_gb_s
                  << " GB/s\n";
    }
}

/**
 * @brief Example of testing different configurations/variants
 * Modify this for your operation's specific variants
 */
TEST_F(OperationUnifiedTestFloat, GemmVariantsTest)
{
    // Test different GEMM variants (transpose options, different sizes, etc.)
    std::vector<std::shared_ptr<GemmConfig<float>>> variant_configs;

    const int size = 256;

    // Different transpose combinations
    variant_configs.push_back(
        std::make_shared<GemmConfig<float>>(size, size, size, false, false, 1.0f, 0.0f, "NN_NoTrans"));
    variant_configs.push_back(
        std::make_shared<GemmConfig<float>>(size, size, size, true, false, 1.0f, 0.0f, "TN_TransA"));
    variant_configs.push_back(
        std::make_shared<GemmConfig<float>>(size, size, size, false, true, 1.0f, 0.0f, "NT_TransB"));
    variant_configs.push_back(
        std::make_shared<GemmConfig<float>>(size, size, size, true, true, 1.0f, 0.0f, "TT_TransBoth"));

    // Different alpha/beta combinations
    variant_configs.push_back(
        std::make_shared<GemmConfig<float>>(size, size, size, false, false, 2.0f, 0.0f, "Alpha2_Beta0"));
    variant_configs.push_back(
        std::make_shared<GemmConfig<float>>(size, size, size, false, false, 1.0f, 1.0f, "Alpha1_Beta1"));

    // Reference implementation
    auto reference_impl = [](const GemmConfig<float>& config, float* output) {
        GemmReferenceExample<float>::forward(config, output);
    };

    // FlashCK implementation
    auto flashck_impl = [](const GemmConfig<float>& config, GpuMemoryManager<float>& gpu_mem) -> float* {
        return run_flashck_gemm_example(config, gpu_mem);
    };

    // Test correctness for all variants
    run_correctness_test(variant_configs, reference_impl, flashck_impl, 1e-3f, 1e-4f, true);
}

/**
 * @brief Example test for edge cases
 * Add tests for edge cases specific to your operation
 */
TEST_F(OperationUnifiedTestFloat, GemmEdgeCasesTest)
{
    std::vector<std::shared_ptr<GemmConfig<float>>> edge_configs;

    // Small matrices
    edge_configs.push_back(std::make_shared<GemmConfig<float>>(1, 1, 1, false, false, 1.0f, 0.0f, "Minimal_1x1x1"));
    edge_configs.push_back(std::make_shared<GemmConfig<float>>(1, 16, 1, false, false, 1.0f, 0.0f, "Vector_1x16x1"));
    edge_configs.push_back(std::make_shared<GemmConfig<float>>(16, 1, 16, false, false, 1.0f, 0.0f, "Vector_16x1x16"));

    // Rectangular matrices
    edge_configs.push_back(
        std::make_shared<GemmConfig<float>>(1024, 64, 256, false, false, 1.0f, 0.0f, "Tall_1024x64x256"));
    edge_configs.push_back(
        std::make_shared<GemmConfig<float>>(64, 1024, 256, false, false, 1.0f, 0.0f, "Wide_64x1024x256"));

    // Reference and FlashCK implementations
    auto reference_impl = [](const GemmConfig<float>& config, float* output) {
        GemmReferenceExample<float>::forward(config, output);
    };

    auto flashck_impl = [](const GemmConfig<float>& config, GpuMemoryManager<float>& gpu_mem) -> float* {
        return run_flashck_gemm_example(config, gpu_mem);
    };

    // Test edge cases
    run_correctness_test(edge_configs, reference_impl, flashck_impl, 1e-3f, 1e-4f, true);
}

// Main function for standalone execution
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

/*
 * Instructions for using this template:
 *
 * 1. Copy this file to test_<your_operation>_unified.cpp
 * 2. Replace all instances of "Operation", "Gemm", etc. with your operation name
 * 3. Include your operation's header file
 * 4. Implement your configuration class (see op_test_configs.h for examples)
 * 5. Implement your reference implementation
 * 6. Implement your FlashCK wrapper function
 * 7. Adapt the test configurations to your operation's parameters
 * 8. Add the new test to your CMakeLists.txt
 * 9. Update the TEST_FRAMEWORK_GUIDE.md with your operation specifics
 *
 * The unified framework will handle:
 * - GPU memory management
 * - Data generation and initialization
 * - Error comparison and reporting
 * - Performance measurement and analysis
 * - Test result formatting and display
 */
