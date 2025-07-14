
#include "tests/utils/benchmark_utils.h"
#include "tests/utils/gtest_utils.h"
#include "tests/utils/torch_utils.h"

#include "flashck/core/module/layers/gemm_layers/linear_layer.h"

template<typename T>
class GemmBiasTanhTest: public TestBase {
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
        b_pt_ = GetRandomTorchTensor<T>({n});
    }

    void BuildTorchModel(bool is_benchmark)
    {
        auto torch_func = [&](torch::Tensor x, torch::Tensor w, torch::Tensor b) {
            return torch::tanh(torch::nn::functional::linear(x, w, b)).to(torch::kFloat32);
        };

        if (is_benchmark) {
            benchmark_.BenchmarkTorchFunc(torch_func, x_pt_, w_pt_, b_pt_);
        }

        y_tanh_out_pt_ = torch_func(x_pt_, w_pt_, b_pt_);
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

        auto x = std::make_unique<flashck::Variable>("x_var", flashck::CppTypeToDataType<T>::Type());
        if (m_max.size() == 1) {
            x->SetShape({flashck::DDim({1, m_max[0]}), flashck::DDim(k_runtime)});
        }
        else {
            x->SetShape({flashck::DDim({1, m_max[0]}), flashck::DDim({1, m_max[1]}), flashck::DDim(k_runtime)});
        }
        auto linear_layer = std::make_unique<flashck::LinearLayer<T>>(k_runtime, n_runtime, false, true, "tanh");
        y_out_ater_       = (*linear_layer)(x.get());
        context_ptr->CodeGenAndProfiling();
        context_ptr->BuildContext();

        linear_layer->LoadParam(reinterpret_cast<T*>(w_pt_.data_ptr()), reinterpret_cast<T*>(b_pt_.data_ptr()));
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
            benchmark_.BenchmarkflashckFunc(linear_layer.get());
        }

        linear_layer->Forward();
    }

    void RunTestGemmBiasTanh(const std::vector<int64_t>& m_max,
                             const std::vector<int64_t>& m_runtime,
                             const int64_t               n_runtime,
                             const int64_t               k_runtime,
                             bool                        is_benchmark,
                             const std::string&          test_name)
    {
        // m_max == m_runtime
        if (m_max.size() != m_runtime.size()) {
            throw std::runtime_error("RunTestGemmBiasTanh: invalid input shape");
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
                                  y_tanh_out_pt_.data_ptr<float>(),
                                  y_tanh_out_pt_.numel());
        EXPECT_TRUE(passed);
    }

private:
    // torch
    torch::Tensor x_pt_, w_pt_, b_pt_;

    ProfileBenchmark benchmark_{};

    // torch and flashck out
    torch::Tensor      y_tanh_out_pt_;
    flashck::Variable* y_out_ater_ = nullptr;

    int test_id_ = 0;
};

TYPED_TEST_SUITE(GemmBiasTanhTest, KernelTestTypes);

TYPED_TEST(GemmBiasTanhTest, test_gemm_bias_tanh_2d_static)
{
    this->RunTestGemmBiasTanh({1024}, {1024}, 256, 512, true, "test_gemm_bias_tanh_2d_static");
}

TYPED_TEST(GemmBiasTanhTest, test_gemm_bias_tanh_3d_static)
{
    this->RunTestGemmBiasTanh({4, 1024}, {4, 1024}, 256, 512, true, "test_gemm_bias_tanh_3d_static");
}