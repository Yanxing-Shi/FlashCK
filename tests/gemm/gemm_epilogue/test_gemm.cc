
#include "tests/utils/benchmark_utils.h"
#include "tests/utils/gtest_utils.h"
#include "tests/utils/torch_utils.h"

#include "lightinfer/core/module/layers/gemm_layers/linear_layer.h"

template<typename T>
class GemmTest: public TestBase {
public:
    void SetUp() override {};
    void TearDown() override {};

    void InitTestData(const std::vector<int64_t>& m, const int64_t n, const int64_t k)
    {
        if (m.size() == 1) {
            x_pt_ = GetRandomTorchTensor<T>({m[0], k});
        }
        else {
            x_pt_ = GetRandomTorchTensor<T>({m[0], m[1], k});
        }
        w_pt_ = GetRandomTorchTensor<T>({n, k});
        b_pt_ = GetZerosTorchTensor<T>({n});
    }

    void BuildTorchModel(bool is_benchmark)
    {
        auto torch_func = [&](torch::Tensor x, torch::Tensor w, torch::Tensor b) {
            return torch::nn::functional::linear(x, w, b).to(torch::kFloat32);
        };

        if (is_benchmark) {
            benchmark_.BenchmarkTorchFunc(torch_func, x_pt_, w_pt_, b_pt_);
        }

        y_out_pt_ = torch_func(x_pt_, w_pt_, b_pt_);
    }

    void BuildLightInferModel(const std::vector<int64_t>& m_max,
                              const std::vector<int64_t>& m_runtime,
                              const int64_t               n_runtime,
                              const int64_t               k_runtime,
                              bool                        is_split_k,
                              bool                        is_benchmark,
                              const std::string&          test_name)
    {
        std::string context_name =
            test_name + "_" + lightinfer::DataTypeToShortString(lightinfer::CppTypeToDataType<T>::Type());
        lightinfer::Context::CreateGlobalContext(context_name, lightinfer::Mode::Inference);
        auto context_ptr = lightinfer::Context::GetGlobalInstance();

        auto x = std::make_unique<lightinfer::Variable>("x_var", lightinfer::CppTypeToDataType<T>::Type());
        if (m_max.size() == 1) {
            x->SetShape({lightinfer::DDim({1, m_max[0]}), lightinfer::DDim(k_runtime)});
        }
        else {
            x->SetShape(
                {lightinfer::DDim({1, m_max[0]}), lightinfer::DDim({1, m_max[1]}), lightinfer::DDim(k_runtime)});
        }

        auto linear_layer = std::make_unique<lightinfer::LinearLayer<T>>(k_runtime, n_runtime, is_split_k, false);
        y_out_ater_       = (*linear_layer)(x.get());
        context_ptr->CodegenAndProfileKernel();
        context_ptr->BuildContext();

        linear_layer->LoadParam(reinterpret_cast<T*>(w_pt_.data_ptr()));
        x->SetValue((char*)x_pt_.data_ptr());
        torch::Tensor y_pt;
        if (m_max.size() == 1) {
            x->SetShape({m_runtime[0], k_runtime});
            y_pt = GetZerosTorchTensor<T>({m_runtime[0], n_runtime});
        }
        else {
            x->SetShape({m_runtime[0], m_runtime[1], k_runtime});
            y_pt = GetZerosTorchTensor<T>({m_runtime[0], m_runtime[1], n_runtime});
        }
        y_out_ater_->SetValue((char*)y_pt.data_ptr());

        if (is_benchmark) {
            benchmark_.BenchmarkLightInferFunc(linear_layer.get());
        }

        linear_layer->Forward();
    }

    void RunTestGemm(const std::vector<int64_t>& m_max,
                     const std::vector<int64_t>& m_runtime,
                     const int64_t               n_runtime,
                     const int64_t               k_runtime,
                     bool                        is_split_k,
                     bool                        is_benchmark,
                     const std::string&          test_name)
    {
        // m_max == m_runtime
        if (m_max.size() != m_runtime.size()) {
            throw std::runtime_error("RunTestGemm: invalid input shape");
        }

        InitTestData(m_runtime, n_runtime, k_runtime);

        BuildTorchModel(is_benchmark);

        BuildLightInferModel(m_max, m_runtime, n_runtime, k_runtime, is_split_k, is_benchmark, test_name);

        if (is_benchmark) {
            benchmark_.SetTestName(test_name);
            benchmark_.PrintProfileResult();
        }

        bool passed = CheckResult(
            test_name, reinterpret_cast<T*>(y_out_ater_->GetValue()), y_out_pt_.data_ptr<float>(), y_out_pt_.numel());
        EXPECT_TRUE(passed);
    }

private:
    // torch
    torch::Tensor x_pt_, w_pt_, b_pt_;

    ProfileBenchmark benchmark_{};

    // torch and lightinfer out
    torch::Tensor         y_out_pt_;
    lightinfer::Variable* y_out_ater_ = nullptr;

    int test_id_ = 0;
};

TYPED_TEST_SUITE(GemmTest, KernelTestTypes);

TYPED_TEST(GemmTest, test_gemm_2d_static)
{
    this->RunTestGemm({64}, {64}, 3 * 512, 512, false, true, "test_gemm_2d_static");
}

TYPED_TEST(GemmTest, test_gemm_3d_static)
{
    this->RunTestGemm({2, 1024}, {2, 1024}, 256, 512, false, true, "test_gemm_3d_static");
}

// TYPED_TEST(GemmTest, test_split_k_gemm_2d_static)
// {
//     this->RunTestGemm({1024}, {1024}, 1024, 4096, true, true, "test_split_k_gemm_2d_static");
// }

// TYPED_TEST(GemmTest, test_split_k_gemm_3d_static)
// {
//     this->RunTestGemm({4, 1024}, {4, 1024}, 4096, 4096, true, true, "test_split_k_gemm_3d_static");
// }