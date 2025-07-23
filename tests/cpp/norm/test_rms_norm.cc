// test_rms_norm.cc
// RMSNorm correctness and performance tests split from test_norm_unified.cpp

#include "../common/norm_reference_impl.h"
#include "../common/norm_test_config.h"
#include "../common/test_framework.h"
#include "wrapper/cpp/norm/rms_norm.h"
#include <gtest/gtest.h>
#include <memory>

using namespace flashck;
using namespace flashck::test;

class NormUnifiedTestFloat: public UnifiedTestSuite<float> {};

// RMSNorm correctness test
TEST_F(NormUnifiedTestFloat, RMSNormCorrectnessTest)
{
    auto configs        = TestConfigFactory<float>::create_rmsnorm_configs();
    auto reference_impl = [](const RMSNormConfig<float>& config, float* output) {
        RMSNormReference<float>::forward(config, output);
    };
    auto flashck_impl = [](const RMSNormConfig<float>& config, GpuMemoryManager<float>& gpu_mem) -> float* {
        try {
            return rms_norm_fwd(config.gpu_input(), config.gpu_gamma(), config.m(), config.n(), config.epsilon());
        }
        catch (const std::exception& e) {
            std::cerr << "RMSNorm execution failed: " << e.what() << std::endl;
            return nullptr;
        }
    };
    run_correctness_test(configs,
                         std::function<void(const RMSNormConfig<float>&, float*)>(reference_impl),
                         std::function<float*(const RMSNormConfig<float>&, GpuMemoryManager<float>&)>(flashck_impl),
                         1e-3f,
                         1e-4f,
                         true);
}

// RMSNorm performance test
TEST_F(NormUnifiedTestFloat, RMSNormPerformanceTest)
{
    std::vector<std::shared_ptr<RMSNormConfig<float>>> perf_configs;
    perf_configs.push_back(std::make_shared<RMSNormConfig<float>>(64, 512, 1e-5f, "Small_64x512"));
    perf_configs.push_back(std::make_shared<RMSNormConfig<float>>(128, 768, 1e-5f, "Medium_128x768"));
    perf_configs.push_back(std::make_shared<RMSNormConfig<float>>(256, 1024, 1e-5f, "Large_256x1024"));
    perf_configs.push_back(std::make_shared<RMSNormConfig<float>>(512, 2048, 1e-5f, "XLarge_512x2048"));
    perf_configs.push_back(std::make_shared<RMSNormConfig<float>>(1024, 4096, 1e-5f, "XXLarge_1024x4096"));
    auto flashck_impl = [](const RMSNormConfig<float>& config, GpuMemoryManager<float>& gpu_mem) -> float* {
        try {
            return rms_norm_fwd(config.gpu_input(), config.gpu_gamma(), config.m(), config.n(), config.epsilon());
        }
        catch (const std::exception& e) {
            std::cerr << "RMSNorm execution failed: " << e.what() << std::endl;
            return nullptr;
        }
    };
    auto results =
        run_performance_test(perf_configs,
                             std::function<float*(const RMSNormConfig<float>&, GpuMemoryManager<float>&)>(flashck_impl),
                             20,
                             5);
    EXPECT_GT(results.size(), 0) << "No performance results obtained";
    if (!results.empty()) {
        std::cout << "\nBest RMSNorm performance: " << results[0].config_name << " - Latency: " << results[0].latency
                  << " ms"
                  << ", TFLOPs: " << results[0].tflops << ", Bandwidth: " << results[0].bandwidth << " GB/s\n";
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
