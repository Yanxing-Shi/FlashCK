#pragma once

#include <string>

static const std::string g_topk_softmax_create_args_tpl = R"(
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("t", "32", "number of input num_tokens")
        .insert(
            "local_t",
            "-1",
            "Number of local input tokens for curent rank.\n"
            "This value must be within range \"[0, t)\", or \"-1\"(no such feature)\n"
            "This feature is to simulate EP case where where each rank has different tokens.\n"
            "Besides, this value will be stored in a GPU buffer, which is friendly for CUDA graph.")
        .insert("e", "8", "number of num_experts")
        .insert("k", "2", "topk")
        .insert("h", "8192", "hidden_size of this model")
        .insert("i", "8192", "intermediate_size between 2 gemms of FFN")
        .insert("stride", "-1", "stride per row, if -1 then equal to hidden_size")
        .insert("mb", "32", "blocking factor for sorted tokens")
        .insert("tp", "8", "tensor parallel size")

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
)";

static const std::string g_topk_softmax_args_parser_tpl = R"(
    ck_tile::index_t tokens            = arg_parser.get_int("t");
    ck_tile::index_t local_tokens      = arg_parser.get_int("local_t");
    ck_tile::index_t experts           = arg_parser.get_int("e");
    ck_tile::index_t topk              = arg_parser.get_int("k");
    ck_tile::index_t hidden_size       = arg_parser.get_int("h");
    ck_tile::index_t intermediate_size = arg_parser.get_int("i");
    ck_tile::index_t stride            = arg_parser.get_int("stride");
    ck_tile::index_t m_block           = arg_parser.get_int("mb");
    if(stride < 0)
        stride = hidden_size;
    
    // w0 (Gate+Up or Gate only, N size)
    ck_tile::index_t shared_intermediate_size_0 = intermediate_size * (gate_only ? 1 : 2) / tp;
    // w1 (Down, N size)
    ck_tile::index_t shared_intermediate_size_1 = intermediate_size / tp;

    bool is_local_token = local_tokens >= 0 && local_tokens < tokens;

    if(local_tokens > tokens)
    {
        printf("local_tokens:%d larger than tokens:%d, invalid\n", local_tokens, tokens);
        return false;
    }

)";

static const std::string g_topk_softmax_args_decl_tpl = R"(

topk_softmax_kargs karg{x_dev_buf_ptr,
                            value_dev_buf_ptr,
                            index_dev_buf_ptr,
                            num_num_tokens,
                            num_experts,
                            topk,
                            input_stride,
                            output_stride};

)";

static const std::string g_topk_softmax_func_signature_tpl = R"(
void {{function_name}}(
        void* x_dev_buf_ptr,
        void* value_dev_buf_ptr,
        void* index_dev_buf_ptr,
        int64_t num_num_tokens,
        int64_t num_experts,
        int64_t topk,
        int64_t input_stride,
        int64_t output_stride,
        hipStream_t stream
    )
)";

static const std::string g_topk_softmax_func_call_tpl = R"(
    {{function_name}}(
        x_dev.GetDeviceBuffer(),
        value_dev.GetDeviceBuffer(),
        index_dev.GetDeviceBuffer(),
        num_num_tokens,
        num_experts,
        topk,
        input_stride,
        output_stride,
        stream
    );
)";

const static std::string g_topk_softmax_tensor_decl_tpl = R"(
    ck_tile::HostTensor<ADataType> a_host({tokens, hidden_size}, {stride, 1});
    ck_tile::HostTensor<GDataType> g_host({experts, shared_intermediate_size_0, hidden_size});
    ck_tile::HostTensor<DDataType> d_host({experts, hidden_size, shared_intermediate_size_1});
    ck_tile::HostTensor<ODataType> o_host({tokens, hidden_size}, {stride, 1});
    ck_tile::HostTensor<AScaleDataType> sa_host({tokens});
    ck_tile::HostTensor<GScaleDataType> sg_host({shared_intermediate_size_0});
    ck_tile::HostTensor<DScaleDataType> sd_host({shared_intermediate_size_1});
    ck_tile::HostTensor<YSmoothScaleDataType> sy_host({shared_intermediate_size_1}); // smooth-quant
    ck_tile::HostTensor<IndexDataType> topk_ids_host({tokens, topk});                // to be sort
    ck_tile::HostTensor<TopkWeightDataType> topk_weight_host({tokens, topk});        // to be sort
    ck_tile::HostTensor<IndexDataType> local_expert_mask_host({experts});

    int max_num_tokens_padded = topk * tokens + experts * block_m - topk;
    ck_tile::HostTensor<IndexDataType> sorted_token_ids_host({max_num_tokens_padded});
    ck_tile::HostTensor<TopkWeightDataType> sorted_weight_host({max_num_tokens_padded});
    ck_tile::HostTensor<IndexDataType> sorted_expert_ids_host(
        {(max_num_tokens_padded + block_m - 1) / block_m});
    ck_tile::HostTensor<IndexDataType> num_sorted_tiles_host({1});

    if(init == 0)
    {
        ck_tile::FillStepRange<ADataType>{-.5f, .5f, 0.01f}(a_host);
        ck_tile::FillStepRange<GDataType>{-.5f, .5f, 0.01f}(g_host);
        ck_tile::FillStepRange<DDataType, false>{.5f, -.5f, -0.01f}(d_host);
        ck_tile::FillStepRange<AScaleDataType>{0.f, 1.f, 0.01f}(sa_host);
        ck_tile::FillStepRange<GScaleDataType>{0.f, 1.f, 0.01f}(sg_host);
        ck_tile::FillStepRange<DScaleDataType>{0.f, 1.f, 0.01f}(sd_host);
        ck_tile::FillStepRange<YSmoothScaleDataType>{0.f, 1.f, 0.01f}(sy_host);
        ck_tile::FillStepRange<TopkWeightDataType>{-.5f, .5f, 0.01f}(topk_weight_host);
    }
    else if(init == 1)
    {
        ck_tile::FillUniformDistribution<ADataType>{-.5f, .5f, seed, true}(a_host);
        ck_tile::FillUniformDistribution<GDataType>{-.5f, .5f, seed, true}(g_host);
        ck_tile::FillUniformDistribution<DDataType>{-.5f, .5f, seed, true}(d_host);
        ck_tile::FillUniformDistribution<AScaleDataType>{-.5f, .5f, seed, true}(sa_host);
        ck_tile::FillUniformDistribution<GScaleDataType>{-.5f, .5f, seed, true}(sg_host);
        ck_tile::FillUniformDistribution<DScaleDataType>{-.5f, .5f, seed, true}(sd_host);
        ck_tile::FillUniformDistribution<YSmoothScaleDataType>{-.5f, .5f, seed, true}(sy_host);
        ck_tile::FillUniformDistribution<TopkWeightDataType>{-.5f, .5f, seed, true}(
            topk_weight_host);
    }
    else if(init == 2)
    {
        ck_tile::FillNormalDistribution<ADataType>{0.f, 1.f, seed, true}(a_host);
        ck_tile::FillNormalDistribution<GDataType>{0.f, 1.f, seed, true}(g_host);
        ck_tile::FillNormalDistribution<DDataType>{0.f, 1.f, seed, true}(d_host);
        ck_tile::FillNormalDistribution<AScaleDataType>{0.f, 1.f, seed, true}(sa_host);
        ck_tile::FillNormalDistribution<GScaleDataType>{0.f, 1.f, seed, true}(sg_host);
        ck_tile::FillNormalDistribution<DScaleDataType>{0.f, 1.f, seed, true}(sd_host);
        ck_tile::FillNormalDistribution<YSmoothScaleDataType>{0.f, 1.f, seed, true}(sy_host);
        ck_tile::FillNormalDistribution<TopkWeightDataType>{0.f, 1.f, seed, true}(topk_weight_host);
    }

    // permute weight
    ck_tile::HostTensor<GDataType> g_perm_host = shuffle_moe_weight(g_host, prec_w, 1);
    ck_tile::HostTensor<DDataType> d_perm_host = shuffle_moe_weight(d_host, prec_w, 1);

)";