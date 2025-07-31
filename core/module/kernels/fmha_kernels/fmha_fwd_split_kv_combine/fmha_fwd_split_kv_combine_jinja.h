#pragma once

#include <string>

static const std::string g_fmha_fwd_splitkv_combine_create_args_tpl = R"(
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("b", "2", "batch size")
        .insert("h", "8", "num of head, for q")
        .insert(
            "s",
            "3328",
            "q_seq_len. if group-mode, means the average value of q_seq_len\n"
            "total_seqlen_q = q_seq_len * batch, and q_seq_len per batch may vary\n"
            "also with \"-s=s0,s1,s2...\" comma seperated int to set per batch seqlen group-mode")
        .insert("d_v", "-1", "head dim for v, -1 means equal to d")
        .insert("num_splits",
                "1",
                "# of splits for key/value. 0 to determine actual number by heuristic");
                
    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
)";

static const std::string g_fmha_fwd_splitkv_combine_args_parser_tpl = R"(
    ck_tile::index_t batch   = arg_parser.get_int("b");
    ck_tile::index_t q_seq_len = arg_parser.get_int("s");
    ck_tile::index_t q_num_heads = arg_parser.get_int("h");
    ck_tile::index_t v_head_dim = arg_parser.get_int("d_v");

    ck_tile::index_t num_splits = arg_parser.get_int("num_splits");

)";

static const std::string g_fmha_fwd_splitkv_combine_args_decl_tpl = R"(
struct FmhaFwdSplitKVCombineArgs
{
    void* lse_acc_ptr;
    void* o_acc_ptr;
    void* o_ptr;

    // the real q_seq_len & seqlen_k are decided by following:
    // batch mode: q_seq_len = kargs.q_seq_len
    //             seqlen_k = kargs.seqlen_k
    // group mode: q_seq_len = kargs.q_seq_start_ptr[b + 1] - kargs.q_seq_start_ptr[b]
    //             seqlen_k = kargs.seqstart_k_ptr[b + 1] - kargs.seqstart_k_ptr[b]
    // kvcache mode (use same kernel as batch mode):
    //             q_seq_len = kargs.q_seq_len
    //             seqlen_k = kargs.seqstart_k_ptr[b + 1] - kargs.seqstart_k_ptr[b]
    const void* q_seq_start_ptr;

    ck_tile::index_t q_seq_len;
    ck_tile::index_t batch;
    ck_tile::index_t q_max_seq_len;
    ck_tile::index_t v_head_dim;
    ck_tile::index_t q_num_heads;
    ck_tile::index_t num_splits;

    float scale_o;

    ck_tile::index_t o_acc_stride;
    ck_tile::index_t o_stride;
    ck_tile::index_t lse_acc_num_heads_stride;
    ck_tile::index_t o_acc_num_heads_stride;
    ck_tile::index_t o_num_heads_stride;
    ck_tile::index_t lse_acc_batch_stride;
    ck_tile::index_t o_acc_batch_stride;
    ck_tile::index_t batch_stride_o;
    ck_tile::index_t lse_acc_split_stride;
    ck_tile::index_t o_acc_split_stride;

};

)";

static const std::string g_fmha_fwd_splitkv_combine_func_signature_tpl = R"(
    {% if is_execute %} {{c_flag}} ATER_EXPORT {% endif %} void {{function_name}}(
        void* lse_acc_buf_ptr,
        void* o_acc_buf_ptr,
        void* o_buf_ptr,
        int64_t* q_seq_start_ptr,
        int64_t batch,
        int64_t q_seq_len,
        int64_t q_num_heads,
        int64_t v_head_dim,
        int64_t q_max_seq_len,
        int num_splits,
        hipStream_t stream
    )
)";

static const std::string g_fmha_fwd_splitkv_combine_func_call_tpl = R"(
    {{function_name}}(
        lse_acc_buf.GetDeviceBuffer(),
        o_acc_buf.GetDeviceBuffer(),
        o_buf.GetDeviceBuffer(),
{% if mode == "group" %}
        seqstart_q.GetDeviceBuffer(),
{% else %}
        nullptr,
{% endif %}       
        batch,
        q_shape_seq_len,
        q_num_heads,
        v_head_dim,
        q_max_seq_len,
        num_splits,
        stream
    );
)";

static const std::string g_fmha_fwd_splitkv_combine_prepare_args_tpl = R"(

    const auto init_args = [&](auto& args){  
    /// NOTE: we broadcast bias from [1, 1, q_seq_len, seqlen_k] to [batch, nhead, q_seq_len,
    ///       seqlen_k] in this example, hence both the 'batch_stride_bias' &
    ///       'nhead_stride_bias' are 0.

    const ck_tile::index_t batch_shape = {% if mode == "batch" %} batch; {% else %} 1; {% endif %}
    const ck_tile::index_t q_shape_seq_len = {% if mode == "batch" %} q_seq_len; {% else %} q_seq_start_ptr[sizeof(q_seq_start_ptr)/sizeof(q_seq_start_ptr[0]) - 1]; {% endif %}

    // setup stride_* arguments
    const ck_tile::index_t o_acc_stride   = v_head_dim;
    const ck_tile::index_t o_stride       = q_num_heads * v_head_dim;

    // setup nhead_stride_* arguments
    const ck_tile::index_t lse_acc_num_heads_stride = (num_splits * q_shape_seq_len);
    const ck_tile::index_t o_acc_num_heads_stride   = num_splits * q_shape_seq_len * v_head_dim;
    const ck_tile::index_t o_num_heads_stride       = v_head_dim;

    // setup batch_stride_* arguments
    const ck_tile::index_t lse_acc_batch_stride = (q_num_heads * num_splits * q_shape_seq_len);
    const ck_tile::index_t o_acc_batch_stride = q_num_heads * num_splits * q_shape_seq_len * v_head_dim;
    const ck_tile::index_t batch_stride_o     = q_num_heads * q_shape_seq_len * v_head_dim;
    
    // setup split_stride_* arguments (only used in split-kv kernel)
    const ck_tile::index_t lse_acc_split_stride = q_shape_seq_len;
    const ck_tile::index_t o_acc_split_stride   = q_shape_seq_len * v_head_dim;

    args.lse_acc_ptr = lse_acc_buf_ptr;
    args.o_acc_ptr = o_acc_buf_ptr;
    args.o_ptr = o_buf_ptr;

    args.batch    = batch;
    args.q_seq_len = q_shape_seq_len; // unused in group mode
    args.v_head_dim   = v_head_dim;
    args.q_num_heads  = q_num_heads;

    args.q_seq_start_ptr = q_seq_start_ptr;

    args.q_max_seq_len = q_max_seq_len;

    args.o_stride          = o_stride;
    args.o_acc_stride         = o_acc_stride;

    args.o_num_heads_stride    = o_num_heads_stride;
    args.lse_acc_num_heads_stride = lse_acc_num_heads_stride;
    args.o_acc_num_heads_stride   = o_acc_num_heads_stride;

    args.lse_acc_batch_stride = lse_acc_batch_stride;
    args.o_acc_batch_stride   = o_acc_batch_stride;

    args.lse_acc_split_stride = lse_acc_split_stride;
    args.o_acc_split_stride   = o_acc_split_stride;

    };
)";

static const std::string g_fmha_fwd_splitkv_combine_make_args_tpl = R"(
    FmhaFwdSplitKVCombineArgs fmha_fwd_splitkv_combine_args;
    init_args(fmha_fwd_splitkv_combine_args);
    
{% if mode == "group" %}
    auto kargs = {{kernel_name}}::MakeKargs(fmha_fwd_splitkv_combine_args.lse_acc_ptr,
                                            fmha_fwd_splitkv_combine_args.o_acc_ptr,
                                            nullptr, // lse_ptr
                                            fmha_fwd_splitkv_combine_args.o_ptr,
                                            fmha_fwd_splitkv_combine_args.batch,
                                            fmha_fwd_splitkv_combine_args.q_seq_start_ptr,
                                            fmha_fwd_splitkv_combine_args.v_head_dim,
                                            fmha_fwd_splitkv_combine_args.num_splits,
                                            fmha_fwd_splitkv_combine_args.scale_o,
                                            fmha_fwd_splitkv_combine_args.o_acc_stride,
                                            fmha_fwd_splitkv_combine_args.o_stride,
                                            fmha_fwd_splitkv_combine_args.lse_acc_num_heads_stride,
                                            fmha_fwd_splitkv_combine_args.o_acc_num_heads_stride,
                                            0, // nhead_stride_lse
                                            fmha_fwd_splitkv_combine_args.o_num_heads_stride,
                                            fmha_fwd_splitkv_combine_args.lse_acc_split_stride,
                                            fmha_fwd_splitkv_combine_args.o_acc_split_stride
                                            );
{% else %}
    auto kargs = {{kernel_name}}::MakeKargs(fmha_fwd_splitkv_combine_args.lse_acc_ptr,
                                            fmha_fwd_splitkv_combine_args.o_acc_ptr,
                                            nullptr, // lse_ptr
                                            fmha_fwd_splitkv_combine_args.o_ptr,
                                            fmha_fwd_splitkv_combine_args.batch,
                                            fmha_fwd_splitkv_combine_args.q_seq_len,
                                            fmha_fwd_splitkv_combine_args.v_head_dim,
                                            fmha_fwd_splitkv_combine_args.num_splits,
                                            fmha_fwd_splitkv_combine_args.scale_o,
                                            fmha_fwd_splitkv_combine_args.o_acc_stride,
                                            fmha_fwd_splitkv_combine_args.o_stride,
                                            fmha_fwd_splitkv_combine_args.lse_acc_num_heads_stride,
                                            fmha_fwd_splitkv_combine_args.o_acc_num_heads_stride,
                                            0, // head_stride_lse
                                            fmha_fwd_splitkv_combine_args.o_num_heads_stride,
                                            fmha_fwd_splitkv_combine_args.lse_acc_batch_stride,
                                            fmha_fwd_splitkv_combine_args.o_acc_batch_stride,
                                            0, // batch_stride_lse
                                            fmha_fwd_splitkv_combine_args.batch_stride_o,
                                            fmha_fwd_splitkv_combine_args.lse_acc_split_stride,
                                            fmha_fwd_splitkv_combine_args.o_acc_split_stride);
{% endif %}
    dim3 grids = {{kernel_name}}::GridSize(fmha_fwd_splitkv_combine_args.batch, fmha_fwd_splitkv_combine_args.q_num_heads, fmha_fwd_splitkv_combine_args.q_max_seq_len, fmha_fwd_splitkv_combine_args.v_head_dim);
)";

static const std::string g_fmha_fwd_splitkv_combine_tensor_decl_tpl = R"(
{% if num_splits > 1 %}
    ck_tile::HostTensor<LSEDataType> lse_acc_host(std::array<ck_tile::index_t, 4>{batch_shape, q_num_heads, num_splits, q_shape_seq_len});
{% endif %}
     ck_tile::HostTensor<OaccDataType> o_acc_host(
        std::array<ck_tile::index_t, 5>{batch_shape, q_num_heads, num_splits, q_shape_seq_len, v_head_dim});
    ck_tile::HostTensor<ODataType> o_host(
        {batch_shape, q_shape_seq_len, q_num_heads, v_head_dim});
)";

const static std::string g_fmha_fwd_splitkv_combine_tensor_generate_tpl = R"(
{% if init_method == "uri" %}
{% if num_splits > 1 %}
    ck_tile::FillUniformDistributionIntegerValue<LSEDataType>{-3.f, 3.f, {{seed}}}(lse_acc_host);
{% endif %}
    ck_tile::FillUniformDistributionIntegerValue<OaccDataType>{-3.f, 3.f, {{seed}}}(o_acc_host);

{% elif init_method == "nri" %}
{% if num_splits > 1 %}
    ck_tile::FillNormalDistributionIntegerValue<LSEDataType>{-3.f, 3.f, {{seed}}}(lse_acc_host);
{% endif %}
    ck_tile::FillUniformDistributionIntegerValue<OaccDataType>{-3.f, 3.f, {{seed}}}(o_acc_host);

{% elif init_method == "uf" %}
{% if num_splits > 1 %}
    ck_tile::FillUniformDistribution<LSEDataType>{0.f, 1.f, {{seed}}}(lse_acc_host);
{% endif %}
    ck_tile::FillUniformDistribution<OaccDataType>{0.f, 1.f, {{seed}}}(o_acc_host);

{% elif init_method == "nf" %}
{% if num_splits > 1 %}
    ck_tile::FillNormalDistribution<LSEDataType>{0.f, 3.f, {{seed}}}(lse_acc_host);
{% endif %}
    ck_tile::FillNormalDistribution<OaccDataType>{0.f, 1.f, {{seed}}}(o_acc_host);

{% elif init_method == "tf" %}
{% if num_splits > 1 %}
    ck_tile::FillTrigValue<LSEDataType>{}(lse_acc_host);
{% endif %}
    ck_tile::FillTrigValue<OaccDataType>{}(o_acc_host);

{% elif init_method == "uf8q" %}
{% if num_splits > 1 %}
    ck_tile::FillUniformDistribution<LSEDataType>{-dtype_max, dtype_max, {{seed}}}(lse_acc_host);
{% endif %}
    ck_tile::FillUniformDistribution<OaccDataType>{-dtype_max, dtype_max, {{seed}}}(o_acc_host);

{% endif %}


{% if num_splits > 1 %}
    ck_tile::DeviceMem lse_acc_buf(lse_acc_host.get_element_space_size_in_bytes());
{% endif %}
    ck_tile::DeviceMem o_acc_buf(o_acc_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_buf(o_host.get_element_space_size_in_bytes());

)";
