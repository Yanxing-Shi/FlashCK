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
        .insert("unit", "32", "unit_size")
        .insert("st_i", "-1", "row stride of input, -1 means same as num_experts")
        .insert("st_o", "-1", "row stride of output/indices, -1 means same as topk")

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
)";

static const std::string g_topk_softmax_args_parser_tpl = R"(
    int num_num_tokens              = args.get_int("t");
    int num_experts             = args.get_int("e");
    int topk                = args.get_int("k");
    int stride_input        = args.get_int("st_i");
    int stride_output       = args.get_int("st_o");
    if(stride_input < 0)
    {
        stride_input = num_experts;
    }
    if(stride_output < 0)
    {
        stride_output = topk;
    }
    assert(stride_input >= num_experts);
    assert(stride_output >= topk);
    assert(topk < num_experts);

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
    // num_tokens already considered batch size
    ck_tile::HostTensor<InputType> x_host({num_tokens, num_experts}, {stride_input, 1});
    ck_tile::HostTensor<WeightType> value_host({num_tokens, topk}, {stride_output, 1});
    ck_tile::HostTensor<IndexType> index_host({num_tokens, topk}, {stride_output, 1});

    {
        // random require per-row unique
        auto rand_gen = ck_tile::FillUniformDistribution_Unique<InputType>{
            -5.f, 5.f, {{seed}})};

        for(int i_t = 0; i_t < num_tokens; i_t++)
        {
            ck_tile::HostTensor<InputType> x_row({num_experts});
            rand_gen(x_row);
            std::copy(x_row.begin(), x_row.end(), x_host.begin() + i_t * stride_input);
            rand_gen.clear();
        }
    }

    ck_tile::DeviceMem x_dev(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem value_dev(value_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem index_dev(index_host.get_element_space_size_in_bytes());

    x_dev.ToDevice(x_host.data());

)";