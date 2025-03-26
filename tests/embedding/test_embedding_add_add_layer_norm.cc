
#include "tests/utils/benchmark_utils.h"
#include "tests/utils/gtest_utils.h"
#include "tests/utils/torch_utils.h"

#include "lightinfer/core/module/layers/embedding_layers/embedding_layer.h"

template<typename T>
class EmbeddingAddAddLayerNormTest: public TestBase {
public:
    void SetUp() override {};
    void TearDown() override {};

    void InitTestData(const std::vector<int64_t>& m,
                      const int64_t               vocab_size,
                      const int64_t               type_vocab_size,
                      const int64_t               max_position_embeddings,
                      const int64_t               emb_dims)
    {
        if (m.size() == 1) {
            x0_pt_ = GetRandomIntTorchTensor({m[0]}, 0, vocab_size - 1);
            x1_pt_ = GetRandomIntTorchTensor({m[0]}, 0, type_vocab_size - 1);
            x2_pt_ = GetRandomIntTorchTensor({m[0]}, 0, max_position_embeddings - 1);
        }
        else {
            x0_pt_ = GetRandomIntTorchTensor({m[0], m[1]}, 0, vocab_size - 1);
            x1_pt_ = GetRandomIntTorchTensor({m[0], m[1]}, 0, type_vocab_size - 1);
            x2_pt_ = GetRandomIntTorchTensor({m[0], m[1]}, 0, max_position_embeddings - 1);
        }

        w0_pt_ = GetRandomTorchTensor<T>({vocab_size, emb_dims});
        w1_pt_ = GetRandomTorchTensor<T>({type_vocab_size, emb_dims});
        w2_pt_ = GetRandomTorchTensor<T>({max_position_embeddings, emb_dims});

        gamma_pt_ = GetRandomTorchTensor<T>({emb_dims});
        beta_pt_  = GetRandomTorchTensor<T>({emb_dims});
    }

    void BuildTorchModel(const int64_t emb_dims, const float epsilon, bool is_benchmark)
    {
        if (is_benchmark) {
            auto torch_func = [&](torch::Tensor x0_pt,
                                  torch::Tensor x1_pt,
                                  torch::Tensor x2_pt,
                                  torch::Tensor w0_pt,
                                  torch::Tensor w1_pt,
                                  torch::Tensor w2_pt,
                                  torch::Tensor gamma_pt,
                                  torch::Tensor beta_pt) {
                auto emb_out_pt = torch::nn::functional::embedding(x0_pt, w0_pt)
                                  + torch::nn::functional::embedding(x1_pt, w1_pt)
                                  + torch::nn::functional::embedding(x2_pt, w2_pt);

                return torch::nn::functional::layer_norm(emb_out_pt,
                                                         torch::nn::functional::LayerNormFuncOptions({emb_dims})
                                                             .eps(epsilon)
                                                             .weight(gamma_pt)
                                                             .bias(beta_pt))
                    .to(torch::kFloat32);
            };

            benchmark_.BenchmarkTorchFunc(
                torch_func, x0_pt_, x1_pt_, x2_pt_, w0_pt_, w1_pt_, w2_pt_, gamma_pt_, beta_pt_);
        }

        // bert embedding
        // vocab size embedding + type vocab size embedding + position embedding
        auto emb_out_pt = torch::nn::functional::embedding(x0_pt_, w0_pt_)
                          + torch::nn::functional::embedding(x1_pt_, w1_pt_)
                          + torch::nn::functional::embedding(x2_pt_, w2_pt_);
        // layernorm: gamma, beta
        y_emb_layernorm_out_pt_ =
            torch::nn::functional::layer_norm(
                emb_out_pt,
                torch::nn::functional::LayerNormFuncOptions({emb_dims}).eps(epsilon).weight(gamma_pt_).bias(beta_pt_))
                .to(torch::kFloat32);
    }

    void BuildLightInferModel(const std::vector<int64_t>& m_max,
                              const std::vector<int64_t>& m_runtime,
                              const int64_t               vocab_size,
                              const int64_t               type_vocab_size,
                              const int64_t               max_position_embeddings,
                              const int64_t               emb_dims,
                              const float                 epsilon,
                              bool                        is_benchmark,
                              const std::string&          test_name)
    {
        std::string context_name =
            test_name + "_" + lightinfer::DataTypeToShortString(lightinfer::CppTypeToDataType<T>::Type());
        lightinfer::Context::CreateGlobalContext(context_name, lightinfer::Mode::Inference);
        auto context_ptr = lightinfer::Context::GetGlobalInstance();

        auto x0 = std::make_unique<lightinfer::Variable>("x0_var", lightinfer::DataType::INT64);
        auto x1 = std::make_unique<lightinfer::Variable>("x1_var", lightinfer::DataType::INT64);
        auto x2 = std::make_unique<lightinfer::Variable>("x2_var", lightinfer::DataType::INT64);

        if (m_max.size() == 1) {
            x0->SetShape({lightinfer::DDim({1, m_max[0]})});
            x1->SetShape({lightinfer::DDim({1, m_max[0]})});
            x2->SetShape({lightinfer::DDim({1, m_max[0]})});
        }
        else {
            x0->SetShape({lightinfer::DDim({1, m_max[0]}), lightinfer::DDim({1, m_max[1]})});
            x1->SetShape({lightinfer::DDim({1, m_max[0]}), lightinfer::DDim({1, m_max[1]})});
            x2->SetShape({lightinfer::DDim({1, m_max[0]}), lightinfer::DDim({1, m_max[1]})});
        }

        auto embedding_layer = std::make_unique<lightinfer::EmbeddingLayer<T>>(
            vocab_size, type_vocab_size, max_position_embeddings, emb_dims, epsilon);
        y_out_ater_ = (*embedding_layer)(x0.get(), x1.get(), x2.get());
        context_ptr->CodegenAndProfileKernel();
        context_ptr->BuildContext();

        embedding_layer->LoadParam(reinterpret_cast<T*>(w0_pt_.data_ptr()),
                                   reinterpret_cast<T*>(w1_pt_.data_ptr()),
                                   reinterpret_cast<T*>(w2_pt_.data_ptr()),
                                   reinterpret_cast<T*>(gamma_pt_.data_ptr()),
                                   reinterpret_cast<T*>(beta_pt_.data_ptr()));
        x0->SetValue((char*)x0_pt_.data_ptr());
        x1->SetValue((char*)x1_pt_.data_ptr());
        x2->SetValue((char*)x2_pt_.data_ptr());

        torch::Tensor y_pt;
        if (m_max.size() == 1) {
            x0->SetShape({m_runtime[0]});
            x1->SetShape({m_runtime[0]});
            x2->SetShape({m_runtime[0]});
            y_pt = GetZerosTorchTensor<T>({m_runtime[0], emb_dims});
        }
        else {
            x0->SetShape({m_runtime[0], m_runtime[1]});
            x1->SetShape({m_runtime[0], m_runtime[1]});
            x2->SetShape({m_runtime[0], m_runtime[1]});
            y_pt = GetZerosTorchTensor<T>({m_runtime[0], m_runtime[1], emb_dims});
        }
        y_out_ater_->SetValue((char*)y_pt.data_ptr());

        if (is_benchmark) {
            benchmark_.BenchmarkLightInferFunc(embedding_layer.get());
        }

        embedding_layer->Forward();
    }

    void RunTestEmbeddingAddAddLayerNorm(const std::vector<int64_t>& m_max,
                                         const std::vector<int64_t>& m_runtime,
                                         const int64_t               vocab_size,
                                         const int64_t               type_vocab_size,
                                         const int64_t               max_position_embeddings,
                                         const int64_t               emb_dims,
                                         bool                        is_benchmark,
                                         const std::string&          test_name,
                                         const float                 epsilon = 1e-12)
    {
        // m_max == m_runtime
        if (m_max.size() != m_runtime.size()) {
            throw std::runtime_error("RunTestEmbeddingAddAddLayerNorm: invalid input shape");
        }

        InitTestData(m_runtime, vocab_size, type_vocab_size, max_position_embeddings, emb_dims);

        BuildTorchModel(emb_dims, epsilon, is_benchmark);

        BuildLightInferModel(m_max,
                             m_runtime,
                             vocab_size,
                             type_vocab_size,
                             max_position_embeddings,
                             emb_dims,
                             epsilon,
                             is_benchmark,
                             test_name);

        if (is_benchmark) {
            benchmark_.SetTestName(test_name);
            benchmark_.PrintProfileResult();
        }

        bool passed = CheckResult(test_name,
                                  reinterpret_cast<T*>(y_out_ater_->GetValue()),
                                  y_emb_layernorm_out_pt_.data_ptr<float>(),
                                  y_emb_layernorm_out_pt_.numel());
        EXPECT_TRUE(passed);
    }

private:
    // torch
    torch::Tensor x0_pt_, x1_pt_, x2_pt_;
    torch::Tensor w0_pt_, w1_pt_, w2_pt_;
    torch::Tensor gamma_pt_, beta_pt_;

    ProfileBenchmark benchmark_{};

    // torch and lightinfer out
    torch::Tensor         y_emb_layernorm_out_pt_;
    lightinfer::Variable* y_out_ater_ = nullptr;

    int test_id_ = 0;
};

TYPED_TEST_SUITE(EmbeddingAddAddLayerNormTest, KernelTestTypes);

// TYPED_TEST(EmbeddingAddAddLayerNormTest, test_embedding_add_add_layernorm_2d_static)
// {
//     this->RunTestEmbeddingAddAddLayerNorm(
//         {1024}, {1024}, 30522, 2, 512, 1024, true, "test_embedding_add_add_layernorm_2d_static");
// }

TYPED_TEST(EmbeddingAddAddLayerNormTest, test_embedding_add_add_layernorm_3d_d1024_static)
{
    this->RunTestEmbeddingAddAddLayerNorm(
        {1, 1024}, {1, 1024}, 30522, 2, 512, 768, true, "test_embedding_add_add_layernorm_3d_d1024_bs1");

    this->RunTestEmbeddingAddAddLayerNorm(
        {2, 1024}, {2, 1024}, 30522, 2, 512, 768, true, "test_embedding_add_add_layernorm_3d_d1024_bs2");

    this->RunTestEmbeddingAddAddLayerNorm(
        {4, 1024}, {4, 1024}, 30522, 2, 512, 768, true, "test_embedding_add_add_layernorm_3d_d1024_bs4");

    this->RunTestEmbeddingAddAddLayerNorm(
        {8, 1024}, {8, 1024}, 30522, 2, 512, 768, true, "test_embedding_add_add_layernorm_3d_d1024_bs8");

    this->RunTestEmbeddingAddAddLayerNorm(
        {16, 1024}, {16, 1024}, 30522, 2, 512, 768, true, "test_embedding_add_add_layernorm_3d_d1024_bs16");

    this->RunTestEmbeddingAddAddLayerNorm(
        {32, 1024}, {32, 1024}, 30522, 2, 512, 768, true, "test_embedding_add_add_layernorm_3d_d1024_bs32");
}
