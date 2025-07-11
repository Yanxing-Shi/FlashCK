
#include "tests/utils/benchmark_utils.h"
#include "tests/utils/gtest_utils.h"
#include "tests/utils/torch_utils.h"

#include "flashck/core/module/layers/gemm_layers/linear_layer.h"

template<typename T>
class GemmBiasPermuteTest: public TestBase {
public:
    void SetUp() override {};
    void TearDown() override {};

    void InitTestData(const std::vector<int64_t>& m, const std::vector<int64_t>& n, const int64_t k)
    {
        // permute m2n3
        if (m.size() == 2 && n.size() == 3) {
            m_mul_res_ = m[0] * m[1];
            n_mul_res_ = n[0] * n[1] * n[2];

            x_pt_ = GetRandomTorchTensor<T>({m_mul_res_, k});
            w_pt_ = GetRandomTorchTensor<T>({n_mul_res_, k});
            b_pt_ = GetRandomTorchTensor<T>({n_mul_res_});
        }
        else {
            throw std::runtime_error("InitTestData: invalid input shape");
        }
    }

    void
    BuildTorchM2N3Model(const std::vector<int64_t>& m_runtime, const std::vector<int64_t>& n_runtime, bool is_benchmark)
    {
        auto torch_func = [&](torch::Tensor x, torch::Tensor w, torch::Tensor b) {
            auto linear_out_pt = torch::nn::functional::linear(x_pt_, w_pt_, b_pt_);
            auto reshape_out_pt =
                torch::reshape(linear_out_pt, {m_runtime[0], m_runtime[1], n_runtime[0], n_runtime[1], n_runtime[2]});
            return torch::permute(reshape_out_pt, {2, 0, 3, 1, 4}).contiguous().to(torch::kFloat32);
        };

        if (is_benchmark) {
            benchmark_.BenchmarkTorchFunc(torch_func, x_pt_, w_pt_, b_pt_);
        }

        y_permute_out_pt_ = torch_func(x_pt_, w_pt_, b_pt_);
    }

    void BuildAterM2N3Model(const std::vector<int64_t>& m_max,
                            const std::vector<int64_t>& m_runtime,
                            const std::vector<int64_t>& n_runtime,
                            const int64_t               k_runtime,
                            bool                        is_benchmark,
                            const std::string&          test_name)
    {
        std::string context_name =
            test_name + "_" + flashck::DataTypeToShortString(flashck::CppTypeToDataType<T>::Type());
        flashck::Context::CreateGlobalContext(context_name, flashck::Mode::Inference);
        auto context_ptr = flashck::Context::GetGlobalInstance();

        auto x = std::make_unique<flashck::Variable>("x_var", flashck::CppTypeToDataType<T>::Type());
        x->SetShape({flashck::DDim({1, m_max[0]}), flashck::DDim({1, m_max[1]}), flashck::DDim(k_runtime)});

        auto linear_layer = std::make_unique<flashck::LinearLayer<T>>(
            k_runtime, n_mul_res_, false, true, "permute", flashck::Shape({n_runtime[0], n_runtime[1]}));

        y_out_ater_ = (*linear_layer)(x.get());
        context_ptr->CodegenAndProfileKernel();
        context_ptr->BuildContext();

        linear_layer->LoadParam(reinterpret_cast<T*>(w_pt_.data_ptr()), reinterpret_cast<T*>(b_pt_.data_ptr()));
        x->SetValue((char*)x_pt_.data_ptr());
        x->SetShape({m_runtime[0], m_runtime[1], k_runtime});

        auto y_pt = GetZerosTorchTensor<T>({m_mul_res_, n_mul_res_});
        y_out_ater_->SetValue((char*)y_pt.data_ptr());

        if (is_benchmark) {
            benchmark_.BenchmarkflashckFunc(linear_layer.get());
        }

        linear_layer->Forward();
    }

    void RunTestGemmBiasPermute(const std::vector<int64_t>& m_max,
                                const std::vector<int64_t>& m_runtime,
                                const std::vector<int64_t>& n_runtime,
                                const int64_t               k_runtime,
                                bool                        is_benchmark,
                                const std::string&          test_name)
    {
        // m_max == m_runtime
        if (m_max.size() != m_runtime.size()) {
            throw std::runtime_error("RunTestGemmBiasPermute: invalid input shape");
        }

        InitTestData(m_runtime, n_runtime, k_runtime);

        BuildTorchM2N3Model(m_runtime, n_runtime, is_benchmark);

        BuildAterM2N3Model(m_max, m_runtime, n_runtime, k_runtime, is_benchmark, test_name);

        if (is_benchmark) {
            benchmark_.SetTestName(test_name);
            benchmark_.PrintProfileResult();
        }

        bool passed = CheckResult(test_name,
                                  reinterpret_cast<T*>(y_out_ater_->GetValue()),
                                  y_permute_out_pt_.data_ptr<float>(),
                                  y_permute_out_pt_.numel());
        EXPECT_TRUE(passed);
    }

private:
    int64_t m_mul_res_ = 1;
    int64_t n_mul_res_ = 1;

    torch::Tensor    x_pt_, w_pt_, b_pt_;
    ProfileBenchmark benchmark_{};

    torch::Tensor      y_permute_out_pt_;
    flashck::Variable* y_out_ater_ = nullptr;

    int test_id_ = 0;
};

TYPED_TEST_SUITE(GemmBiasPermuteTest, KernelTestTypes);

TYPED_TEST(GemmBiasPermuteTest, test_permute_m2n3_static)
{
    this->RunTestGemmBiasPermute({2, 256}, {2, 256}, {4, 16, 128}, 256, true, "test_permute_m2n3_static");
}