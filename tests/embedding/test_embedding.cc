
#include "tests/utils/gtest_utils.h"
#include "tests/utils/torch_utils.h"

#include "lightinfer/core/module/layers/embedding_layers/embedding_layer.h"

template<typename T>
class EmbeddingTest: public TestBase {
public:
    void SetUp() override {};
    void TearDown() override {};

    void InitTestData(const std::vector<int64_t>& m, const int64_t num_embeddings, const int64_t emb_dims)
    {
        if (m.size() == 1) {
            x_pt_ = GetRandomIntTorchTensor({m[0]}, 0, num_embeddings - 1);
        }
        else {
            x_pt_ = GetRandomIntTorchTensor({m[0], m[1]}, 0, num_embeddings - 1);
        }

        w_pt_ = GetRandomTorchTensor<T>({num_embeddings, emb_dims});

        gamma_pt_ = GetOnesTorchTensor<T>({emb_dims});
        beta_pt_  = GetZerosTorchTensor<T>({emb_dims});
    }

    void BuildTorchModel()
    {
        y_out_pt_ = torch::nn::functional::embedding(x_pt_, w_pt_);
    }

    void BuildLightInferModel(const std::vector<int64_t>& m_max,
                              const std::vector<int64_t>& m_runtime,
                              const int64_t               num_embeddings,
                              const int64_t               emb_dims,
                              const std::string&          test_name)
    {
        std::string context_name =
            test_name + "_" + lightinfer::DataTypeToShortString(lightinfer::CppTypeToDataType<T>::Type());
        lightinfer::Context::CreateGlobalContext(context_name, lightinfer::Mode::Inference);
        auto context_ptr = lightinfer::Context::GetGlobalInstance();

        auto x = std::make_unique<lightinfer::Variable>("x_var", lightinfer::DataType::INT64);
        if (m_max.size() == 1) {
            x->SetShape({lightinfer::DDim({1, m_max[0]})});
        }
        else {
            x->SetShape({lightinfer::DDim({1, m_max[0]}), lightinfer::DDim({1, m_max[1]})});
        }

        auto embedding_layer = std::make_unique<lightinfer::EmbeddingLayer<T>>(num_embeddings, emb_dims, 0.f);
        y_out_ater_          = (*embedding_layer)(x.get());
        context_ptr->CodegenAndProfileKernel();
        context_ptr->BuildContext();

        embedding_layer->LoadParam(reinterpret_cast<T*>(w_pt_.data_ptr()),
                                   reinterpret_cast<T*>(gamma_pt_.data_ptr()),
                                   reinterpret_cast<T*>(beta_pt_.data_ptr()));
        x->SetValue((char*)x_pt_.data_ptr());
        torch::Tensor y_pt;
        if (m_max.size() == 1) {
            x->SetShape({m_runtime[0]});
            y_pt = GetZerosTorchTensor<T>({m_runtime[0], emb_dims});
        }
        else {
            x->SetShape({m_runtime[0], m_runtime[1]});
            y_pt = GetZerosTorchTensor<T>({m_runtime[0], m_runtime[1], emb_dims});
        }
        y_out_ater_->SetValue((char*)y_pt.data_ptr());

        embedding_layer->Forward();
    }

    void RunTestEmbedding(const std::vector<int64_t>& m_max,
                          const std::vector<int64_t>& m_runtime,
                          const int64_t               num_embeddings,
                          const int64_t               emb_dims,
                          const std::string&          test_name)
    {
        // m_max == m_runtime
        if (m_max.size() != m_runtime.size()) {
            throw std::runtime_error("RunTestEmbedding: invalid input shape");
        }

        InitTestData(m_runtime, num_embeddings, emb_dims);

        BuildTorchModel();

        BuildLightInferModel(m_max, m_runtime, num_embeddings, emb_dims, test_name);

        bool passed = CheckResult(test_name,
                                  reinterpret_cast<T*>(y_out_ater_->GetValue()),
                                  reinterpret_cast<T*>(y_out_pt_.data_ptr()),
                                  y_out_pt_.numel());
        EXPECT_TRUE(passed);
    }

private:
    // torch
    torch::Tensor x_pt_, w_pt_, gamma_pt_, beta_pt_;

    // torch and lightinfer out
    torch::Tensor         y_out_pt_;
    lightinfer::Variable* y_out_ater_ = nullptr;

    int test_id_ = 0;
};

TYPED_TEST_SUITE(EmbeddingTest, KernelTestTypes);

TYPED_TEST(EmbeddingTest, test_embedding_2d_static)
{
    this->RunTestEmbedding({1024}, {1024}, 30522, 512, "test_embedding_2d_static");
}

TYPED_TEST(EmbeddingTest, test_embedding_3d_static)
{
    this->RunTestEmbedding({1, 1024}, {1, 1024}, 30522, 512, "test_embedding_3d_static");
}