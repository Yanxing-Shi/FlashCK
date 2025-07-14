
#include "tests/cpp/utils/benchmark_utils.h"
#include "tests/cpp/utils/gtest_utils.h"
#include "tests/cpp/utils/torch_utils.h"

#include "flashck/core/module/layers/norm_layers/layer_norm_layer.h"

template<typename T>
class LayerNormTest: public TestBase {
public:
    void SetUp() override {};
    void TearDown() override {};

    void InitTestData(const std::vector<int64_t>& m, const int64_t emb_dims)
    {
        if (m.size() == 1) {
            x_pt_ = GetRandomTorchTensor<T>({m[0], emb_dims});
        }
        else {
            x_pt_ = GetRandomTorchTensor<T>({m[0], m[1], emb_dims});
        }

        gamma_pt_ = GetRandomTorchTensor<T>({emb_dims});
        beta_pt_  = GetRandomTorchTensor<T>({emb_dims});
    }

    void BuildTorchModel(const int64_t emb_dims, const float epsilon, bool is_benchmark)
    {
        if (is_benchmark) {
            auto torch_func = [&](torch::Tensor x, torch::Tensor gamma, torch::Tensor beta) {
                return torch::nn::functional::layer_norm(x,
                                                         torch::nn::functional::LayerNormFuncOptions({emb_dims})
                                                             .eps(epsilon)
                                                             .weight(gamma)
                                                             .bias(beta))
                    .to(torch::kFloat32);
            };
            benchmark_.BenchmarkTorchFunc(torch_func, x_pt_, gamma_pt_, beta_pt_);
        }

        y_layernorm_out_pt_ =
            torch::nn::functional::layer_norm(
                x_pt_,
                torch::nn::functional::LayerNormFuncOptions({emb_dims}).eps(epsilon).weight(gamma_pt_).bias(beta_pt_))
                .to(torch::kFloat32);
    }

    void BuildflashckModel(const std::vector<int64_t>& m_max,
                           const std::vector<int64_t>& m_runtime,
                           const int64_t               emb_dims,
                           const float                 epsilon,
                           bool                        is_benchmark,
                           const std::string&          test_name)
    {
        std::string context_name = test_name + "_" + flashck::DataTypeToString(flashck::CppTypeToDataType<T>::Type());
        flashck::Context::CreateGlobalContext(context_name);
        auto context_ptr = flashck::Context::GetGlobalInstance();

        auto x = std::make_unique<flashck::Variable>("x_var", flashck::CppTypeToDataType<T>::Type());

        if (m_max.size() == 1) {
            x->SetShape({flashck::DDim({1, m_max[0]}), flashck::DDim(emb_dims)});
        }
        else {
            x->SetShape({flashck::DDim({1, m_max[0]}), flashck::DDim({1, m_max[1]}), flashck::DDim(emb_dims)});
        }

        auto layer_norm_layer = std::make_unique<flashck::LayerNormLayer<T>>(flashck::Shape({flashck::DDim(emb_dims)}));
        y_out_ater_           = (*layer_norm_layer)(x.get(), epsilon);
        context_ptr->BuildContext();

        flashck::ProfilingEngine::GetInstance()->GetGraphCodeGen()->CodeGenAndProfiling(context_ptr->GetModelOps(),
                                                                                        context_name);

        layer_norm_layer->LoadParam(reinterpret_cast<T*>(gamma_pt_.data_ptr()),
                                    reinterpret_cast<T*>(beta_pt_.data_ptr()));
        x->SetValue((char*)x_pt_.data_ptr());
        torch::Tensor y_pt;
        if (m_max.size() == 1) {
            x->SetShape({m_runtime[0], emb_dims});
            y_pt = GetZerosTorchTensor<T>({m_runtime[0], emb_dims});
        }
        else {
            x->SetShape({m_runtime[0], m_runtime[1], emb_dims});
            y_pt = GetZerosTorchTensor<T>({m_runtime[0], m_runtime[1], emb_dims});
        }
        y_out_ater_->SetValue((char*)y_pt.data_ptr());

        if (is_benchmark) {
            benchmark_.BenchmarkflashckFunc(layer_norm_layer.get());
        }

        layer_norm_layer->Forward();
    }

    void RunTestLayerNorm(const std::vector<int64_t>& m_max,
                          const std::vector<int64_t>& m_runtime,
                          const int64_t               emb_dims,
                          bool                        is_benchmark,
                          const std::string&          test_name,
                          const float                 epsilon = 1e-5)
    {
        // m_max == m_runtime
        if (m_max.size() != m_runtime.size()) {
            throw std::runtime_error("RunTestLayerNorm: invalid input shape");
        }

        InitTestData(m_runtime, emb_dims);

        BuildTorchModel(emb_dims, epsilon, is_benchmark);

        BuildflashckModel(m_max, m_runtime, emb_dims, epsilon, is_benchmark, test_name);

        if (is_benchmark) {
            benchmark_.PrintProfileResult();
        }

        bool passed = CheckResult(test_name,
                                  reinterpret_cast<T*>(y_out_ater_->GetValue()),
                                  y_layernorm_out_pt_.data_ptr<float>(),
                                  y_layernorm_out_pt_.numel());
        EXPECT_TRUE(passed);
    }

private:
    // torch
    torch::Tensor x_pt_, gamma_pt_, beta_pt_;

    ProfileBenchmark benchmark_{};

    // torch and flashck out
    torch::Tensor      y_layernorm_out_pt_;
    flashck::Variable* y_out_ater_ = nullptr;

    int test_id_ = 0;
};

TYPED_TEST_SUITE(LayerNormTest, KernelTestTypes);

TYPED_TEST(LayerNormTest, test_layernorm_2d_d768)
{
    this->RunTestLayerNorm({64}, {64}, 768, true, "test_layernorm_2d_d768_m64");
    this->RunTestLayerNorm({128}, {128}, 768, true, "test_layernorm_2d_d768_m128");
    this->RunTestLayerNorm({256}, {256}, 768, true, "test_layernorm_2d_d768_m256");
    this->RunTestLayerNorm({512}, {512}, 768, true, "test_layernorm_2d_d768_m512");
    this->RunTestLayerNorm({1024}, {1024}, 768, true, "test_layernorm_2d_d768_m1024");
}

// TYPED_TEST(LayerNormTest, test_layernorm_3d_static)
// {
//     this->RunTestLayerNorm({2, 1024}, {2, 1024}, 512, true, "test_layernorm_3d_static");
// }