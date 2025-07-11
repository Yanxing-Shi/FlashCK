
#include "tests/utils/benchmark_utils.h"
#include "tests/utils/gtest_utils.h"
#include "tests/utils/torch_utils.h"

#include "flashck/core/module/layers/gemm_layers/bmm_layer.h"

template<typename T>
class BmmTest: public TestBase {
public:
    void SetUp() override {};
    void TearDown() override {};

    void InitTestData(const int64_t b, const int64_t m, const int64_t n, const int64_t k)
    {
        x_pt_ = GetRandomTorchTensor<T>({b, m, k});
        w_pt_ = GetRandomTorchTensor<T>({n, k});
        b_pt_ = GetRandomTorchTensor<T>({b, n});
    }

    void BuildTorchModel(bool is_benchmark)
    {
        if (is_benchmark) {
            auto torch_func = [&](torch::Tensor x, torch::Tensor w, torch::Tensor b) {
                return torch::nn::functional::linear(x, w, b);
            };
            benchmark_.BenchmarkTorchFunc(torch_func, x_pt_, w_pt_, b_pt_);
        }
        y_out_pt_ = torch::nn::functional::linear(x_pt_, w_pt_, b_pt_);
    }

    void BuildflashckModel(const int64_t      b_max,
                           const int64_t      m_max,
                           const int64_t      b_runtime,
                           const int64_t      m_runtime,
                           const int64_t      n_runtime,
                           const int64_t      k_runtime,
                           bool               is_benchmark,
                           const std::string& test_name)
    {
        std::string context_name =
            test_name + "_" + flashck::DataTypeToShortString(flashck::CppTypeToDataType<T>::Type());
        flashck::Context::CreateGlobalContext(context_name, flashck::Mode::Inference);
        auto context_ptr = flashck::Context::GetGlobalInstance();

        auto x = std::make_unique<flashck::Variable>("x_var", flashck::CppTypeToDataType<T>::Type());
        x->SetShape({flashck::DDim({1, b_max}), flashck::DDim({1, m_max}), flashck::DDim(k_runtime)});

        auto bmm_layer = std::make_unique<flashck::BmmLayer<T>>(k_runtime, n_runtime, false);
        y_out_ater_    = (*bmm_layer)(x.get());
        context_ptr->CodegenAndProfileKernel();
        context_ptr->BuildContext();

        bmm_layer->LoadParam(reinterpret_cast<T*>(w_pt_.data_ptr()));
        x->SetValue((char*)x_pt_.data_ptr());

        x->SetShape({b_runtime, m_runtime, k_runtime});
        auto y_pt = GetZerosTorchTensor<T>({b_runtime, m_runtime, n_runtime});

        y_out_ater_->SetValue((char*)y_pt.data_ptr());

        if (is_benchmark) {
            benchmark_.BenchmarkflashckFunc(bmm_layer.get());
        }

        bmm_layer->Forward();
    }

    void RunTestGemm(const int64_t      b_max,
                     const int64_t      m_max,
                     const int64_t      b_runtime,
                     const int64_t      m_runtime,
                     const int64_t      n_runtime,
                     const int64_t      k_runtime,
                     bool               is_benchmark,
                     const std::string& test_name)
    {
        InitTestData(b_runtime, m_runtime, n_runtime, k_runtime);

        BuildTorchModel(is_benchmark);

        BuildflashckModel(b_max, m_max, b_runtime, m_runtime, n_runtime, k_runtime, is_benchmark, test_name);

        if (is_benchmark) {
            benchmark_.SetTestName(test_name);
            benchmark_.PrintProfileResult();
        }

        bool passed = CheckResult(test_name,
                                  reinterpret_cast<T*>(y_out_ater_->GetValue()),
                                  reinterpret_cast<T*>(y_out_pt_.data_ptr()),
                                  y_out_pt_.numel());
        EXPECT_TRUE(passed);
    }

private:
    // torch
    torch::Tensor x_pt_, w_pt_, b_pt_;

    ProfileBenchmark benchmark_{};

    // torch and flashck out
    torch::Tensor      y_out_pt_;
    flashck::Variable* y_out_ater_ = nullptr;

    int test_id_ = 0;
};

TYPED_TEST_SUITE(BmmTest, KernelTestTypes);

TYPED_TEST(BmmTest, test_bmm_static)
{
    this->RunTestGemm(1, 128, 1, 128, 3 * 128, 128, true, "test_bmm_static");
}
