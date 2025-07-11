#include "tests/utils/benchmark_utils.h"
#include "tests/utils/gtest_utils.h"
#include "tests/utils/torch_utils.h"

#include "tests/attention/test_bias_utils.h"
#include "tests/attention/test_fmha_configs.h"
#include "tests/attention/test_mask_utils.h"
#include "tests/attention/test_ref_utils.h"
#include "tests/attention/test_tensor_utils.h"

#include "flashck/core/module/layers/attention_layers/memory_efficient_attention_layer.h"

template<typename T>
class MemoryEfficientAttentionTest: public TestBase {
public:
    void SetUp() override {};
    void TearDown() override {};

    void
    BuildflashckModel(const AttentionConfigs& configs, bool is_benchmark = false, const std::string& test_name = "")
    {

        flashck::Context::CreateGlobalContext(test_name, flashck::Mode::Inference);
        auto context_ptr = flashck::Context::GetGlobalInstance();

        auto q_var    = std::make_unique<flashck::Variable>("q_var", flashck::CppTypeToDataType<T>::Type());
        auto k_var    = std::make_unique<flashck::Variable>("k_var", flashck::CppTypeToDataType<T>::Type());
        auto v_var    = std::make_unique<flashck::Variable>("v_var", flashck::CppTypeToDataType<T>::Type());
        auto bias_var = configs.bias_enum_ != flashck::BiasEnum::NO_BIAS ?
                            std::make_unique<flashck::Variable>("bias_var", flashck::CppTypeToDataType<T>::Type()) :
                            nullptr;

        q_var->SetShape({
            flashck::DDim({1, configs.batch_size_}),
            flashck::DDim({1, configs.q_seq_len_}),
            flashck::DDim(configs.q_num_heads_),
            flashck::DDim(configs.qk_head_dim_),
        });
        k_var->SetShape({
            flashck::DDim({1, configs.batch_size_}),
            flashck::DDim({1, configs.kv_seq_len_}),
            flashck::DDim(configs.kv_num_heads_),
            flashck::DDim(configs.qk_head_dim_),
        });
        v_var->SetShape({
            flashck::DDim({1, configs.batch_size_}),
            flashck::DDim({1, configs.kv_seq_len_}),
            flashck::DDim(configs.kv_num_heads_),
            flashck::DDim(configs.v_head_dim_),
        });
        if (configs.bias_enum_ != flashck::BiasEnum::NO_BIAS) {
            if (configs.bias_enum_ == flashck::BiasEnum::ELEMENTWISE_BIAS) {
                if (configs.bias_rank_info_ == 0) {
                    bias_var->SetShape({
                        flashck::DDim(1),
                        flashck::DDim(1),
                        flashck::DDim({1, configs.q_seq_len_}),
                        flashck::DDim({1, configs.kv_seq_len_}),
                    });
                }
                else if (configs.bias_rank_info_ == 1) {
                    bias_var->SetShape({
                        flashck::DDim(1),
                        flashck::DDim(configs.q_num_heads_),
                        flashck::DDim({1, configs.q_seq_len_}),
                        flashck::DDim({1, configs.kv_seq_len_}),
                    });
                }
                else if (configs.bias_rank_info_ == 2) {
                    bias_var->SetShape({
                        flashck::DDim({1, configs.batch_size_}),
                        flashck::DDim(configs.q_num_heads_),
                        flashck::DDim({1, configs.q_seq_len_}),
                        flashck::DDim({1, configs.kv_seq_len_}),
                    });
                }
            }
            else if (configs.bias_enum_ == flashck::BiasEnum::ALIBI) {
                if (configs.bias_rank_info_ == 0) {
                    bias_var->SetShape({flashck::DDim(1), flashck::DDim(configs.q_num_heads_)});
                }
                else {
                    bias_var->SetShape({flashck::DDim({1, configs.batch_size_}), flashck::DDim(configs.q_num_heads_)});
                }
            }
            else {
                throw std::runtime_error("Invalid bias type");
            }
        }

        auto attn_layer = std::make_unique<flashck::MemoryEfficientAttentionLayer<T>>(flashck::FmhaOperationMode::Batch,
                                                                                      configs.q_num_heads_,
                                                                                      configs.kv_num_heads_,
                                                                                      configs.qk_head_dim_,
                                                                                      configs.v_head_dim_,
                                                                                      configs.scale_,
                                                                                      configs.bias_enum_,
                                                                                      configs.window_size_,
                                                                                      configs.mask_enum_,
                                                                                      false);
        ater_out_       = (*attn_layer)(q_var.get(), k_var.get(), v_var.get(), bias_var.get());
        context_ptr->CodegenAndProfileKernel();
        context_ptr->BuildContext();

        q_var->SetValue((char*)q_.data_ptr());
        q_var->SetShape({
            configs.batch_size_,
            configs.q_seq_len_,
            configs.q_num_heads_,
            configs.qk_head_dim_,
        });
        k_var->SetValue((char*)k_.data_ptr());
        k_var->SetShape({
            configs.batch_size_,
            configs.kv_seq_len_,
            configs.kv_num_heads_,
            configs.qk_head_dim_,
        });
        v_var->SetValue((char*)v_.data_ptr());
        v_var->SetShape({
            configs.batch_size_,
            configs.kv_seq_len_,
            configs.kv_num_heads_,
            configs.v_head_dim_,
        });
        if (configs.bias_enum_ != flashck::BiasEnum::NO_BIAS) {
            if (configs.bias_enum_ == flashck::BiasEnum::ELEMENTWISE_BIAS) {
                bias_var->SetValue((char*)attn_bias_.data_ptr());
                if (configs.bias_rank_info_ == 0) {
                    bias_var->SetShape({
                        1,
                        1,
                        configs.q_seq_len_,
                        configs.kv_seq_len_,
                    });
                }
                else if (configs.bias_rank_info_ == 1) {
                    bias_var->SetShape({
                        1,
                        configs.q_num_heads_,
                        configs.q_seq_len_,
                        configs.qk_head_dim_,
                    });
                }
                else if (configs.bias_rank_info_ == 2) {
                    bias_var->SetShape({
                        configs.batch_size_,
                        configs.q_num_heads_,
                        configs.q_seq_len_,
                        configs.kv_num_heads_,
                    });
                }
                else {
                    throw std::runtime_error("Invalid bias rank info");
                }
            }
            else if (configs.bias_enum_ == flashck::BiasEnum::ALIBI) {
                bias_var->SetValue((char*)alibi_slopes_.data_ptr());
                if (configs.bias_rank_info_ == 0) {
                    bias_var->SetShape({1, configs.q_num_heads_});
                }
                else {
                    bias_var->SetShape({configs.batch_size_, configs.q_num_heads_});
                }
            }
            else {
                throw std::runtime_error("Invalid bias type");
            }
        }

        auto y = GetZerosTorchTensor<T>(
            {configs.batch_size_, configs.q_seq_len_, configs.q_num_heads_, configs.v_head_dim_});
        ater_out_->SetValue((char*)y.data_ptr());

        if (is_benchmark) {
            benchmark_.BenchmarkflashckFunc(attn_layer.get());
        }

        attn_layer->Forward();
    }

    void RunTestMemoryEfficientAttention(const AttentionConfigs& configs,
                                         bool                    is_benchmark = false,
                                         const std::string&      test_name    = "")
    {
        // create q_, k_, v_
        std::tie(q_, k_, v_) = CreateBatchQKVTensor<T>(configs.batch_size_,
                                                       configs.q_seq_len_,
                                                       configs.kv_seq_len_,
                                                       configs.q_num_heads_,
                                                       configs.kv_num_heads_,
                                                       configs.qk_head_dim_,
                                                       configs.v_head_dim_);

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

        //  execute torch reference
        auto torch_func = [&](torch::Tensor q,
                              torch::Tensor k,
                              torch::Tensor v,
                              torch::Tensor attn_bias,
                              torch::Tensor local_mask,
                              float scale) { return RefAttentionBMHK(q, k, v, attn_bias, local_mask, scale, true); };

        if (is_benchmark) {
            benchmark_.BenchmarkTorchFunc(torch_func, q_, k_, v_, attn_bias_, local_mask_, configs.scale_);
        }

        torch::Tensor ref_out = torch_func(q_, k_, v_, attn_bias_, local_mask_, configs.scale_);

        // execute flashck inference
        BuildflashckModel(configs, is_benchmark, test_name);

        if (is_benchmark) {
            benchmark_.SetTestName(test_name);
            benchmark_.PrintProfileResult();
        }

        // check result
        bool passed = CheckResult(
            test_name, reinterpret_cast<T*>(ater_out_->GetValue()), ref_out.data_ptr<float>(), ref_out.numel());
        EXPECT_TRUE(passed);
    }

private:
    torch::Tensor q_, k_, v_;
    torch::Tensor attn_bias_, alibi_slopes_;
    torch::Tensor local_mask_;

    flashck::Variable* ater_out_;

    ProfileBenchmark benchmark_;

    int test_id_ = 0;
};

TYPED_TEST_SUITE(MemoryEfficientAttentionTest, KernelTestTypes);

// TYPED_TEST(MemoryEfficientAttentionTest, test_memory_efficient_attn_llama)
// {
//     this->RunTestMemoryEfficientAttention({1,
//                                            32,
//                                            32,
//                                            32,
//                                            32,
//                                            128,
//                                            128,
//                                            1,
//                                            1e-5f,
//                                            flashck::BiasEnumInfo{flashck::BiasEnum::NO_BIAS, 0},
//                                            flashck::GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT,
//                                            {-1, 0}},
//                                           true,
//                                           "test_memory_efficient_attn_s32");
//     this->RunTestMemoryEfficientAttention({1,
//                                            64,
//                                            64,
//                                            32,
//                                            32,
//                                            128,
//                                            128,
//                                            1,
//                                            1e-5f,
//                                            flashck::BiasEnumInfo{flashck::BiasEnum::NO_BIAS, 0},
//                                            flashck::GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT,
//                                            {-1, 0}},
//                                           true,
//                                           "test_memory_efficient_attn_s64");
//     this->RunTestMemoryEfficientAttention({1,
//                                            128,
//                                            128,
//                                            32,
//                                            32,
//                                            128,
//                                            128,
//                                            1,
//                                            1e-5f,
//                                            flashck::BiasEnumInfo{flashck::BiasEnum::NO_BIAS, 0},
//                                            flashck::GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT,
//                                            {-1, 0}},
//                                           true,
//                                           "test_memory_efficient_attn_s128");
//     this->RunTestMemoryEfficientAttention({1,
//                                            256,
//                                            256,
//                                            32,
//                                            32,
//                                            128,
//                                            128,
//                                            1,
//                                            1e-5f,
//                                            flashck::BiasEnumInfo{flashck::BiasEnum::NO_BIAS, 0},
//                                            flashck::GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT,
//                                            {-1, 0}},
//                                           true,
//                                           "test_memory_efficient_attn_s256");
//     this->RunTestMemoryEfficientAttention({1,
//                                            512,
//                                            512,
//                                            32,
//                                            32,
//                                            128,
//                                            128,
//                                            1,
//                                            1e-5f,
//                                            flashck::BiasEnumInfo{flashck::BiasEnum::NO_BIAS, 0},
//                                            flashck::GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT,
//                                            {-1, 0}},
//                                           true,
//                                           "test_memory_efficient_attn_s512");
//     this->RunTestMemoryEfficientAttention({1,
//                                            1024,
//                                            1024,
//                                            32,
//                                            32,
//                                            128,
//                                            128,
//                                            1,
//                                            1e-5f,
//                                            flashck::BiasEnumInfo{flashck::BiasEnum::NO_BIAS, 0},
//                                            flashck::GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT,
//                                            {-1, 0}},
//                                           true,
//                                           "test_memory_efficient_attn_s1024");
//     this->RunTestMemoryEfficientAttention({1,
//                                            2048,
//                                            2048,
//                                            32,
//                                            32,
//                                            128,
//                                            128,
//                                            1,
//                                            1e-5f,
//                                            flashck::BiasEnumInfo{flashck::BiasEnum::NO_BIAS, 0},
//                                            flashck::GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT,
//                                            {-1, 0}},
//                                           true,
//                                           "test_memory_efficient_attn_s2048");
// }

TYPED_TEST(MemoryEfficientAttentionTest, test_memory_efficient_attn_element_nomask)
{
    this->RunTestMemoryEfficientAttention({1,
                                           32,
                                           32,
                                           8,
                                           8,
                                           32,
                                           32,
                                           1,
                                           0.15f,
                                           flashck::BiasEnum::ALIBI,
                                           0,
                                           flashck::GenericAttentionMaskEnum::MASK_FROM_BOTTOM_RIGHT,
                                           {4, 8}},
                                          false,
                                          "test_memory_efficient_attn_element_nomask");
}
