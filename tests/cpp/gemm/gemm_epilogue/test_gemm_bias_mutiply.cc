
#include "tests/utils/benchmark_utils.h"
#include "tests/utils/gtest_utils.h"
#include "tests/utils/torch_utils.h"

#include "flashck/core/module/layers/gemm_layers/linear_layer.h"

template<typename T>
class GemmBiasMulTest: public TestBase {
public:
    void SetUp() override {};
    void TearDown() override {};

    void InitTestData(const std::vector<int64_t>& m, const int64_t n, const int64_t k)
    {
        if (m.size() == 1) {
            x_pt_  = GetRandomTorchTensor<T>({m[0], k});
            d0_pt_ = GetRandomTorchTensor<T>({m[0], n});
        }
        else {
            x_pt_  = GetRandomTorchTensor<T>({m[0], m[1], k});
            d0_pt_ = GetRandomTorchTensor<T>({m[0], m[1], n});
        }
        w_pt_ = GetRandomTorchTensor<T>({n, k});
        b_pt_ = GetRandomTorchTensor<T>({n});
    }

    void BuildTorchModel(bool is_benchmark)
    {
        auto torch_func = [&](torch::Tensor x_pt, torch::Tensor w_pt, torch::Tensor b_pt, torch::Tensor d0_pt) {
            return (torch::nn::functional::linear(x_pt, w_pt, b_pt) * d0_pt).to(torch::kFloat32);
        };

        if (is_benchmark) {
            benchmark_.BenchmarkTorchFunc(torch_func, x_pt_, w_pt_, b_pt_, d0_pt_);
        }

        y_mul_out_pt_ = torch_func(x_pt_, w_pt_, b_pt_, d0_pt_);
    }

    void BuildflashckModel(const std::vector<int64_t>& m_max,
                           const std::vector<int64_t>& m_runtime,
                           const int64_t               n_runtime,
                           const int64_t               k_runtime,
                           bool                        is_benchmark,
                           const std::string&          test_name)
    {
        std::string context_name =
            test_name + "_" + flashck::DataTypeToShortString(flashck::CppTypeToDataType<T>::Type());
        flashck::Context::CreateGlobalContext(context_name, flashck::Mode::Inference);
        auto context_ptr = flashck::Context::GetGlobalInstance();

        auto x  = std::make_unique<flashck::Variable>("x_var", flashck::CppTypeToDataType<T>::Type());
        auto d0 = std::make_unique<flashck::Variable>("d0_var", flashck::CppTypeToDataType<T>::Type());
        if (m_max.size() == 1) {
            x->SetShape({flashck::DDim({1, m_max[0]}), flashck::DDim(k_runtime)});
            d0->SetShape({flashck::DDim({1, m_max[0]}), flashck::DDim(n_runtime)});
        }
        else {
            x->SetShape({flashck::DDim({1, m_max[0]}), flashck::DDim({1, m_max[1]}), flashck::DDim(k_runtime)});
            d0->SetShape({flashck::DDim({1, m_max[0]}), flashck::DDim({1, m_max[1]}), flashck::DDim(n_runtime)});
        }
        auto linear_layer = std::make_unique<flashck::LinearLayer<T>>(k_runtime, n_runtime, false, true, "multiply");
        y_out_ater_       = (*linear_layer)(x.get(), d0.get());
        context_ptr->CodeGenAndProfiling();
        context_ptr->BuildContext();

        linear_layer->LoadParam(reinterpret_cast<T*>(w_pt_.data_ptr()), reinterpret_cast<T*>(b_pt_.data_ptr()));
        x->SetValue((char*)x_pt_.data_ptr());
        d0->SetValue((char*)d0_pt_.data_ptr());
        torch::Tensor y_pt;
        if (m_max.size() == 1) {
            x->SetShape({m_runtime[0], k_runtime});
            d0->SetShape({m_runtime[0], n_runtime});
            y_pt = GetZerosTorchTensor<T>({m_runtime[0], n_runtime});
        }
        else {
            x->SetShape({m_runtime[0], m_runtime[1], k_runtime});
            d0->SetShape({m_runtime[0], m_runtime[1], n_runtime});
            y_pt = GetZerosTorchTensor<T>({m_runtime[0], m_runtime[1], n_runtime});
        }
        y_out_ater_->SetValue((char*)y_pt.data_ptr());

        if (is_benchmark) {
            benchmark_.BenchmarkflashckFunc(linear_layer.get());
        }

        linear_layer->Forward();
    }

    void RunTestGemmBiasMul(const std::vector<int64_t>& m_max,
                            const std::vector<int64_t>& m_runtime,
                            const int64_t               n_runtime,
                            const int64_t               k_runtime,
                            bool                        is_benchmark,
                            const std::string&          test_name)
    {
        // m_max == m_runtime
        if (m_max.size() != m_runtime.size()) {
            throw std::runtime_error("RunTestGemmBiasMul: invalid input shape");
        }

        InitTestData(m_runtime, n_runtime, k_runtime);

        BuildTorchModel(is_benchmark);

        BuildflashckModel(m_max, m_runtime, n_runtime, k_runtime, is_benchmark, test_name);

        if (is_benchmark) {
            benchmark_.SetTestName(test_name);
            benchmark_.PrintProfileResult();
        }

        bool passed = CheckResult(test_name,
                                  reinterpret_cast<T*>(y_out_ater_->GetValue()),
                                  y_mul_out_pt_.data_ptr<float>(),
                                  y_mul_out_pt_.numel());
        EXPECT_TRUE(passed);
    }

private:
    // torch
    torch::Tensor x_pt_, w_pt_, b_pt_, d0_pt_;

    ProfileBenchmark benchmark_{};

    // torch and flashck out
    torch::Tensor      y_mul_out_pt_;
    flashck::Variable* y_out_ater_ = nullptr;

    int test_id_ = 0;
};

TYPED_TEST_SUITE(GemmBiasMulTest, KernelTestTypes);

TYPED_TEST(GemmBiasMulTest, test_gemm_bias_mul_2d_static)
{
    this->RunTestGemmBiasMul({1024}, {1024}, 256, 512, true, "test_gemm_bias_mul_2d_static");
}

TYPED_TEST(GemmBiasMulTest, test_gemm_bias_mul_3d_static)
{
    this->RunTestGemmBiasMul({2, 1024}, {2, 1024}, 256, 512, true, "test_gemm_bias_mul_3d_static");
}