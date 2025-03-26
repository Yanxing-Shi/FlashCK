#include "tests/utils/benchmark_utils.h"
#include "tests/utils/gtest_utils.h"
#include "tests/utils/torch_utils.h"

#include "tests/attention/test_fmha_configs.h"
#include "tests/attention/test_ref_utils.h"

#include "lightinfer/core/module/layers/attention_layers/fuse_multi_head_attention_layer.h"

template<typename T>
class FuseMultiHeadAttentionTest: public TestBase {
public:
    void SetUp() override {};
    void TearDown() override {};

    void BuildLightInferModel(const FuseMultiHeadAttentionConfigs& configs,
                              bool                                 is_benchmark = false,
                              const std::string&                   test_name    = "")
    {
        lightinfer::Context::CreateGlobalContext(test_name, lightinfer::Mode::Inference);
        auto context_ptr = lightinfer::Context::GetGlobalInstance();

        int64_t hidden_dim = configs.q_num_heads_ * configs.qk_head_dim_;
        auto    x          = std::make_unique<lightinfer::Variable>("x_var", lightinfer::CppTypeToDataType<T>::Type());
        x->SetShape({
            lightinfer::DDim({1, configs.batch_size_}),
            lightinfer::DDim({1, configs.q_seq_len_}),
            lightinfer::DDim(hidden_dim),
        });

        auto attn_layer = std::make_unique<lightinfer::FuseMultiHeadAttentionLayer<T>>(
            configs.q_num_heads_,
            configs.kv_num_heads_,
            configs.qk_head_dim_,
            configs.v_head_dim_,
            configs.scale_,
            configs.bias_enum_,
            configs.window_size_,
            configs.mask_enum_,
            configs.is_pre_layer_norm_ ? lightinfer::LayerNormType::PreLayerNorm :
                                         lightinfer::LayerNormType::PostLayerNorm,
            configs.epsilon_);
        y_out_ater_ = (*attn_layer)(x.get());
        context_ptr->CodegenAndProfileKernel();
        context_ptr->BuildContext();

        attn_layer->LoadParam(reinterpret_cast<T*>(gamma_.data_ptr()),
                              reinterpret_cast<T*>(beta_.data_ptr()),
                              reinterpret_cast<T*>(qkv_weight_.data_ptr()),
                              reinterpret_cast<T*>(qkv_bias_.data_ptr()),
                              reinterpret_cast<T*>(out_weight_.data_ptr()),
                              reinterpret_cast<T*>(out_bias_.data_ptr()));

        x->SetValue((char*)x_.data_ptr());
        x->SetShape({
            configs.batch_size_,
            configs.q_seq_len_,
            hidden_dim,
        });

        auto y = GetZerosTorchTensor<T>({configs.batch_size_, configs.q_seq_len_, hidden_dim});
        y_out_ater_->SetValue((char*)y.data_ptr());

        if (is_benchmark) {
            benchmark_.BenchmarkLightInferFunc(attn_layer.get());
        }

        attn_layer->Forward();
    }

    void RunTestFuseMultiHeadAttnQKVPacked(const FuseMultiHeadAttentionConfigs& configs,
                                           bool                                 is_benchmark,
                                           const std::string&                   test_name)
    {
        int64_t hidden_dim = configs.q_num_heads_ * configs.qk_head_dim_;
        x_                 = GetRandomTorchTensor<T>({configs.batch_size_, configs.q_seq_len_, hidden_dim});

        gamma_ = GetRandomTorchTensor<T>({hidden_dim});
        beta_  = GetRandomTorchTensor<T>({hidden_dim});

        qkv_weight_ = GetRandomTorchTensor<T>({hidden_dim * 3, hidden_dim});
        qkv_bias_   = GetRandomTorchTensor<T>({hidden_dim * 3});

        out_weight_ = GetRandomTorchTensor<T>({hidden_dim, hidden_dim});
        out_bias_   = GetRandomTorchTensor<T>({hidden_dim});

        // create attn_bias_ tensor
        std::tie(attn_bias_, alibi_slopes_) = CreateAttentionBiasTensor<T>(configs.batch_size_,
                                                                           configs.q_seq_len_,
                                                                           configs.kv_seq_len_,
                                                                           configs.q_num_heads_,
                                                                           configs.bias_enum_,
                                                                           configs.bias_rank_info_,
                                                                           configs.window_size_);

        // create attention mask tensor
        local_mask_ = GetLocalMaskFromSlidingWindow<T>(
            configs.batch_size_, configs.q_seq_len_, configs.kv_seq_len_, configs.window_size_);

        // execute torch reference
        auto torch_func = [&](const float                         epsilon,
                              const PackedType&                   packed_type,
                              const torch::Tensor&                x,
                              const torch::Tensor&                gamma,
                              const torch::Tensor&                beta,
                              const torch::Tensor&                out_proj_weight,
                              const torch::Tensor&                out_proj_bias,
                              const std::optional<torch::Tensor>& proj_weight   = std::nullopt,
                              const std::optional<torch::Tensor>& proj_bias     = std::nullopt,
                              const std::optional<torch::Tensor>& q_proj_weight = std::nullopt,
                              const std::optional<torch::Tensor>& k_proj_weight = std::nullopt,
                              const std::optional<torch::Tensor>& v_proj_weight = std::nullopt,
                              const std::optional<torch::Tensor>& q_proj_bias   = std::nullopt,
                              const std::optional<torch::Tensor>& k_proj_bias   = std::nullopt,
                              const std::optional<torch::Tensor>& v_proj_bias   = std::nullopt,
                              const std::optional<torch::Tensor>& attn_bias     = std::nullopt,
                              const std::optional<torch::Tensor>& mask          = std::nullopt,
                              const std::optional<float>          scale         = std::nullopt) {
            return RefFuseMultiHeadAttention(epsilon,
                                             packed_type,
                                             x,
                                             gamma,
                                             beta,
                                             out_proj_weight,
                                             out_proj_bias,
                                             qkv_weight,
                                             qkv_bias,
                                             q_proj_weight,
                                             k_proj_weight,
                                             v_proj_weight,
                                             q_proj_bias,
                                             k_proj_bias,
                                             v_proj_bias,
                                             attn_bias,
                                             mask,
                                             scale);
        };

        if (is_benchmark) {
            benchmark_.BenchmarkTorchFunc(torch_func,
                                          configs.epsilon_,
                                          configs.packed_type_,
                                          x_,
                                          gamma_,
                                          beta_,
                                          out_weight_,
                                          out_bias_,
                                          qkv_weight_,
                                          qkv_bias_,
                                          std::nullopt,
                                          std::nullopt,
                                          std::nullopt,
                                          std::nullopt,
                                          std::nullopt,
                                          std::nullopt,
                                          attn_bias_,
                                          local_mask_,
                                          configs.scale_);
        }

        ref_out_ = torch_func(configs.epsilon_,
                              configs.packed_type_,
                              x_,
                              gamma_,
                              beta_,
                              qkv_weight_,
                              qkv_bias_,
                              out_weight_,
                              out_bias_,
                              qkv_weight_,
                              qkv_bias_,
                              std::nullopt,
                              std::nullopt,
                              std::nullopt,
                              std::nullopt,
                              std::nullopt,
                              std::nullopt,
                              attn_bias_,
                              local_mask_,
                              configs.scale_);

        BuildLightInferModel(configs, is_benchmark, test_name);

        if (is_benchmark) {
            benchmark_.SetTestName(test_name);
            benchmark_.PrintProfileResult();
        }

        bool passed = CheckResult(
            test_name, reinterpret_cast<T*>(y_out_ater_->GetValue()), ref_out_.data_ptr<float>(), ref_out_.numel());
        EXPECT_TRUE(passed);
    }

private:
    // Test data
    torch::Tensor x_, gamma_, beta_, qkv_weight_, qkv_bias_, out_weight_, out_bias_;
    torch::Tensor attn_bias_, alibi_slopes_, local_mask_;

    ProfileBenchmark benchmark_{};

    // torch and lightinfer out
    torch::Tensor         ref_out_;
    lightinfer::Variable* y_out_ater_;

    int test_id_ = 0;
};

TYPED_TEST_SUITE(FuseMultiHeadAttentionTest, KernelTestTypes);

TYPED_TEST(FuseMultiHeadAttentionTest, test_fuse_multi_head_attn)
{
    this->RunTestFuseMultiHeadAttnQKVPacked({{1,
                                              32,
                                              32,
                                              8,
                                              8,
                                              32,
                                              32,
                                              1,
                                              1.0f,
                                              lightinfer::BiasEnum::NO_BIAS,
                                              -1,
                                              lightinfer::GenericAttentionMaskEnum::NO_MASK,
                                              {-1, -1}},
                                             false,
                                             1e-5f,
                                             PackedType::QKVPacked},
                                            true,
                                            "test_fuse_multi_head_attn");
}

// TYPED_TEST(FuseMultiHeadAttentionTest, test_multi_head_attn_dynamic)
// {
//     this->RunTestFuseMultiHeadAttn(2, 64, 2, 64, 4, 64, 1e-5f, false, true, "test_multi_head_attn_static");
// }
