#pragma once

#include <string>

static const std::string g_fmha_fwd_create_args_source = R"(
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("b", "2", "batch size")
        .insert("h", "8", "num of head, for q")
        .insert("h_k",
                "-1",
                "num of head, for k/v, -1 means equal to h\n"
                "if not equal to h, then this is GQA/MQA case")
        .insert(
            "s",
            "3328",
            "seqlen_q. if group-mode, means the average value of seqlen_q\n"
            "total_seqlen_q = seqlen_q * batch, and seqlen_q per batch may vary\n"
            "also with \"-s=s0,s1,s2...\" comma seperated int to set per batch seqlen group-mode")
        .insert("s_k", "-1", "seqlen_k (not including new key/value), -1 means equal to s")
        .insert("d", "128", "head dim for q, k")
        .insert("d_v", "-1", "head dim for v, -1 means equal to d")
        .insert("scale",
                "0",
                "scale factor of S. 0 means equal to 1/sqrt(hdim).\n"
                "note when squant=1, this value will be modified by range_q/k")
        .insert("mask", "0", "mask type, 0: no mask, 1: left mask, 2: right mask")
        .insert("window_left_size", "-1", "the size of sliding window")
        .insert("window_right_size", "-1", "the size of sliding window");


    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
)";

static const std::string g_fmha_fwd_args_parser_source = R"(
    ck_tile::index_t batch   = arg_parser.get_int("b");
    ck_tile::index_t seqlen_q = arg_parser.get_int("s");
    ck_tile::index_t seqlen_k = arg_parser.get_int("s_k");
    ck_tile::index_t nhead_q = arg_parser.get_int("h");
    ck_tile::index_t nhead_k = arg_parser.get_int("h_k");
    if(nhead_k < 0)
        nhead_k = nhead_q;

    if(nhead_q % nhead_k != 0)
    {
        std::cerr << "nhead_q:" << nhead_q << " must be multiple of nhead_k:" << nhead_k << std::endl;
    }

    ck_tile::index_t hdim_q = arg_parser.get_int("d");
    ck_tile::index_t hdim_v = arg_parser.get_int("d_v");
    if(hdim_v < 0)
        hdim_v = hdim_q;

    float scale = arg_parser.get_float("scale");
    if(scale == .0f)
        scale = 1.0 / ck_tile::sqrt(static_cast<float>(hdim_q));
    
    uint32_t mask_type = arg_parser.get_uint32("mask");
    ck_tile::index_t window_left_size = arg_parser.get_int("window_left_size");
    ck_tile::index_t window_right_size = arg_parser.get_int("window_right_size");

)";

static const std::string g_fmha_fwd_args_decl_source = R"(

// runtime args, some will passed to karg, some will used to compute grids/blocks
struct FmhaFwdArgs
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    void* o_ptr;

    const void* seqstart_q_ptr;
    const void* seqstart_k_ptr;
    const void*
        seqlen_k_ptr; // only used if both 'seqstart_q_ptr' & 'seqstart_k_ptr' are not nullptr

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    float scale_s;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_randval;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_randval;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_randval;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_o;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;

    float p_drop;

    std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
        drop_seed_offset;

};

)";

static const std::string g_fmha_fwd_func_signature_source = R"(
void {{function_name}}(
    void* q_buf_ptr,
    void* k_buf_ptr,
    void* v_buf_ptr,
    void* bias_buf_ptr,
    void* o_buf_ptr,
    int64_t* seqstart_q_ptr,
    int64_t* seqstart_k_ptr,
    int64_t* seqlen_k_ptr,
    int64_t batch,
    int64_t seqlen_q,
    int64_t seqlen_k,
    int64_t nhead_q,
    int64_t nhead_k,
    int64_t hdim_q,
    int64_t hdim_v,
    int64_t max_seqlen_q,
    float scale,
    std::array<int64_t,2> window_size,
    uint32_t mask_type,
    hipStream_t stream
)
)";

static const std::string g_fmha_fwd_func_call_source = R"(
    {{function_name}}(
        q_buf.GetDeviceBuffer(),
        k_buf.GetDeviceBuffer(),
        v_buf.GetDeviceBuffer(),
{% if bias_str == "alibi" %}
        alibi_slope_buf.GetDeviceBuffer(),
{% elif bias_str == "elementwise" %}
        bias_buf.GetDeviceBuffer(),
{% else %}
        nullptr,
{% endif %}
        o_buf.GetDeviceBuffer(),
{% if mode_str == "group" %}
        seqstart_q.GetDeviceBuffer(),
        seqstart_k.GetDeviceBuffer(),
        seqlen_k_buf.GetDeviceBuffer(),
{% else %}
        nullptr,
        nullptr,
        nullptr,
{% endif %}
        batch,
        shape_seqlen_q,
        shape_seqlen_k,
        nhead_q,
        nhead_k,
        hdim_q,
        hdim_v,
        max_seqlen_q,
        scale,
        {window_left_size,window_right_size},
        mask_type,
        stream
    );
)";

static const std::string g_fmha_fwd_prepare_args_source = R"(
   const auto init_args = [&](auto& args){  

    const ck_tile::index_t shape_batch = {% if mode_str == "batch" %} batch; {% else %} 1; {% endif %}
    const ck_tile::index_t shape_seqlen_q = {% if mode_str == "batch" %} seqlen_q; {% else %} seqstart_q_ptr[sizeof(seqstart_q_ptr)/sizeof(seqstart_q_ptr[0]) - 1]; {% endif %}
    const ck_tile::index_t shape_seqlen_k =
{% if mode_str == "batch" %} seqlen_k; {% else %} seqstart_k_ptr[sizeof(seqstart_q_ptr)/sizeof(seqstart_q_ptr[0]) - 1]; {% endif %}
    
    // setup stride_* arguments
    const ck_tile::index_t stride_q = nhead_q * hdim_q;
    const ck_tile::index_t stride_k = nhead_k * hdim_q;
    const ck_tile::index_t stride_v = nhead_k * hdim_v;

{% if bias_rank_info == 0 %}
    const ck_tile::index_t stride_bias = shape_seqlen_k;
{% elif bias_rank_info == 1 %}
    const ck_tile::index_t stride_bias = shape_seqlen_k;
{% elif bias_rank_info == 2 %}
    const ck_tile::index_t stride_bias = shape_seqlen_k;
{% endif %}

    const ck_tile::index_t stride_o       = nhead_q * hdim_v;

    // setup nhead_stride_* arguments
    const ck_tile::index_t nhead_stride_q = hdim_q;
    const ck_tile::index_t nhead_stride_k = hdim_q;
    const ck_tile::index_t nhead_stride_v = hdim_v;
{% if bias_rank_info == 0 %}
    const ck_tile::index_t nhead_stride_bias = 0 * shape_seqlen_q * shape_seqlen_k;
{% elif bias_rank_info == 1 %}
    const ck_tile::index_t nhead_stride_bias = shape_seqlen_q * shape_seqlen_k;
{% elif bias_rank_info == 2 %}
    const ck_tile::index_t nhead_stride_bias = shape_seqlen_q * shape_seqlen_k;
{% endif %}


    const ck_tile::index_t nhead_stride_o       = hdim_v;

    // setup batch_stride_* arguments
    const ck_tile::index_t batch_stride_q = nhead_q * shape_seqlen_q * hdim_q;
    const ck_tile::index_t batch_stride_k = nhead_k * shape_seqlen_k * hdim_q;
    const ck_tile::index_t batch_stride_v = nhead_k * hdim_v * shape_seqlen_k;
{% if bias_rank_info == 0 %}
    const ck_tile::index_t batch_stride_bias    = 0 * nhead_q * shape_seqlen_q * shape_seqlen_k;
{% elif bias_rank_info == 1 %}
    const ck_tile::index_t batch_stride_bias    = 0 * nhead_q * shape_seqlen_q * shape_seqlen_k;
{% elif bias_rank_info == 2 %}
    const ck_tile::index_t batch_stride_bias    = nhead_q * shape_seqlen_q * shape_seqlen_k;
{% endif %}
    const ck_tile::index_t batch_stride_o     = nhead_q * shape_seqlen_q * hdim_v;

    args.q_ptr = q_buf_ptr;
    args.k_ptr = k_buf_ptr;
    args.v_ptr = v_buf_ptr;

    args.batch    = batch;
    args.seqlen_q = shape_seqlen_q; // unused in group mode
    args.hdim_q   = hdim_q;
    args.hdim_v   = hdim_v;
    args.nhead_q  = nhead_q;
    args.nhead_k  = nhead_k;

    args.stride_q       = stride_q;
    args.stride_k       = stride_k;
    args.stride_v       = stride_v;
    args.nhead_stride_q = nhead_stride_q;
    args.nhead_stride_k = nhead_stride_k;
    args.nhead_stride_v = nhead_stride_v;
    args.batch_stride_q = batch_stride_q;
    args.batch_stride_k = batch_stride_k;
    args.batch_stride_v = batch_stride_v;

    args.bias_ptr = bias_buf_ptr;
    args.o_ptr    = o_buf_ptr;    

    args.seqstart_q_ptr = seqstart_q_ptr;
    args.seqstart_k_ptr = seqstart_k_ptr;
    args.seqlen_k_ptr = seqlen_k_ptr;
    
    args.seqlen_k     = seqlen_k; // unused in group mode (or kvcache enabled)
    args.max_seqlen_q = max_seqlen_q;

    args.scale_s = scale;

    args.stride_bias = 
{% if bias_str == "alibi" %}
{% if bias_rank_info == 0 %} 0 * nhead_q; {% else %} nhead_q; {% endif %}
{% elif bias_str == "elementwise" %}
    stride_bias;
{% else %}
    0;
{% endif %}

    args.stride_o          = stride_o;
    args.nhead_stride_bias = 
{% if bias_str == "alibi" %}
    0;
{% elif bias_str == "elementwise" %}
    nhead_stride_bias;
{% else %}
    0;
{% endif %}
    args.nhead_stride_o    = nhead_stride_o;
    args.batch_stride_bias = 
{% if bias_str == "alibi" %}
    0;
{% elif bias_str == "elementwise" %}
    batch_stride_bias;
{% else %}
    0;
{% endif %}
    args.batch_stride_o    = batch_stride_o;

    args.window_size_left  = window_size[0];
    args.window_size_right = window_size[1];
    args.mask_type         = mask_type;
    };
)";

static const std::string g_fmha_fwd_make_args_source = R"(
    FmhaFwdArgs fmha_fwd_args;
    init_args(fmha_fwd_args);
    
{% if mode_str == "group" %}
    auto kargs = {{kernel_name}}::MakeKargs(fmha_fwd_args.q_ptr,
                                            fmha_fwd_args.k_ptr,
                                            fmha_fwd_args.v_ptr,
                                            fmha_fwd_args.bias_ptr,
                                            nullptr, // rand_val_ptr
                                            nullptr, // lse_ptr
                                            fmha_fwd_args.o_ptr,
                                            fmha_fwd_args.seqstart_q_ptr,
                                            fmha_fwd_args.seqstart_k_ptr,
                                            fmha_fwd_args.seqlen_k_ptr,
                                            fmha_fwd_args.hdim_q,
                                            fmha_fwd_args.hdim_v,
                                            fmha_fwd_args.nhead_q,
                                            fmha_fwd_args.nhead_q / fmha_fwd_args.nhead_k, // nhead_ratio_qk
                                            fmha_fwd_args.scale_s,
                                            1.0f, // scale_p
                                            1.0f, // scale_o
                                            fmha_fwd_args.stride_q,
                                            fmha_fwd_args.stride_k,
                                            fmha_fwd_args.stride_v,
                                            fmha_fwd_args.stride_bias,
                                            0, // stride_rand_val
                                            fmha_fwd_args.stride_o,
                                            fmha_fwd_args.nhead_stride_q,
                                            fmha_fwd_args.nhead_stride_k,
                                            fmha_fwd_args.nhead_stride_v,
                                            fmha_fwd_args.nhead_stride_bias,
                                            0, // nhead_stride_rand_val
                                            0, // nhead_stride_lse
                                            fmha_fwd_args.batch_stride_o,
                                            fmha_fwd_args.window_size_left,
                                            fmha_fwd_args.window_size_right,
                                            fmha_fwd_args.mask_type,
                                            0, // p_dropout
                                            false, // is_store_randval
                                            std::make_pair(static_cast<uint64_t>(0), static_cast<uint64_t>(0)));
     dim3 grids = FmhaKernel::GridSize(
            fmha_fwd_args.batch, fmha_fwd_args.nhead_q, fmha_fwd_args.max_seqlen_q, fmha_fwd_args.hdim_v, fmha_fwd_args.seqlen_k_ptr != nullptr);
{% else %}
    auto kargs = {{kernel_name}}::MakeKargs(fmha_fwd_args.q_ptr,
                                            fmha_fwd_args.k_ptr,
                                            fmha_fwd_args.v_ptr,
                                            fmha_fwd_args.bias_ptr,
                                            nullptr, // rand_val_ptr
                                            nullptr, // lse_ptr
                                            fmha_fwd_args.o_ptr,
                                            fmha_fwd_args.seqlen_q,
                                            fmha_fwd_args.seqlen_k,
                                            fmha_fwd_args.hdim_q,
                                            fmha_fwd_args.hdim_v,
                                            fmha_fwd_args.nhead_q,
                                            fmha_fwd_args.nhead_q / fmha_fwd_args.nhead_k, // nhead_ratio_qk
                                            fmha_fwd_args.scale_s,
                                            1.0f, // scale_p
                                            1.0f, // scale_o
                                            fmha_fwd_args.stride_q,
                                            fmha_fwd_args.stride_k,
                                            fmha_fwd_args.stride_v,
                                            fmha_fwd_args.stride_bias,
                                            0, // stride_rand_val
                                            fmha_fwd_args.stride_o,
                                            fmha_fwd_args.nhead_stride_q,
                                            fmha_fwd_args.nhead_stride_k,
                                            fmha_fwd_args.nhead_stride_v,
                                            fmha_fwd_args.nhead_stride_bias,
                                            0, // nhead_stride_rand_val
                                            0, // nhead_stride_lse
                                            fmha_fwd_args.nhead_stride_o,
                                            fmha_fwd_args.batch_stride_q,
                                            fmha_fwd_args.batch_stride_k,
                                            fmha_fwd_args.batch_stride_v,
                                            fmha_fwd_args.batch_stride_bias,
                                            0, // batch_stride_rand_val
                                            0, // batch_stride_lse
                                            fmha_fwd_args.batch_stride_o,
                                            fmha_fwd_args.window_size_left,
                                            fmha_fwd_args.window_size_right,
                                            fmha_fwd_args.mask_type,
                                            0, // p_dropout
                                            false, // is_store_randval
                                            std::make_pair(static_cast<uint64_t>(0), static_cast<uint64_t>(0)));
    dim3 grids =
            {{kernel_name}}::GridSize(fmha_fwd_args.batch, fmha_fwd_args.nhead_q, fmha_fwd_args.max_seqlen_q, fmha_fwd_args.hdim_v, false);
{% endif %}
)";

static const std::string g_fmha_fwd_tensor_decl_source = R"(
    ck_tile::HostTensor<QDataType> q_host(
        {shape_batch, shape_seqlen_q, nhead_q, hdim_q});
    ck_tile::HostTensor<KDataType> k_host(
        {shape_batch, shape_seqlen_k, nhead_k, hdim_q});
    ck_tile::HostTensor<VDataType> v_host(
        {shape_batch, shape_seqlen_k, nhead_k, hdim_v});

    ck_tile::HostTensor<ODataType> o_host(
        {shape_batch, shape_seqlen_q, nhead_q, hdim_v});
    
{% if bias_str == "elementwise" %}
    ck_tile::HostTensor<BiasDataType> bias_host({1, 1, shape_seqlen_q, shape_seqlen_k});
{% else %}
    ck_tile::HostTensor<BiasDataType> bias_host({1, 1, 1, 1}); // dummy shape for simplifying code
{% endif %}

{% if bias_str == "alibi" %}
{% if bias_rank_info == 0 %}
    // alibi in 1*h
    ck_tile::HostTensor<SaccDataType> alibi_slope_host({1, nhead_q});
{% else %}
    // alibi in b*h
    ck_tile::HostTensor<SaccDataType> alibi_slope_host({batch, nhead_q});
{% endif %}
{% else %}
    // alibi in 1*1
    ck_tile::HostTensor<SaccDataType> alibi_slope_host({1, 1}); // dummy shape for simplifying code
{% endif %}

)";

const static std::string g_fmha_fwd_tensor_generate_source = R"(
{% if init_method_str == "uri" %}
    ck_tile::FillUniformDistributionIntegerValue<QDataType>{-3.f, 3.f, {{seed}}}(q_host);
    ck_tile::FillUniformDistributionIntegerValue<KDataType>{-3.f, 3.f, {{seed}}}(k_host);
    ck_tile::FillUniformDistributionIntegerValue<VDataType>{-3.f, 3.f, {{seed}}}(v_host);
{% if bias_str == "elementwise" %}
    ck_tile::FillUniformDistributionIntegerValue<BiasDataType>{-3.f, 3.f, {{seed}}}(bias_host);
{% endif %}

{% elif init_method_str == "nri" %}
    ck_tile::FillNormalDistributionIntegerValue<QDataType>{-3.f, 3.f, {{seed}}}(q_host);
    ck_tile::FillNormalDistributionIntegerValue<KDataType>{-3.f, 3.f, {{seed}}}(k_host);
    ck_tile::FillNormalDistributionIntegerValue<VDataType>{-3.f, 3.f, {{seed}}}(v_host);
{% if bias_str == "elementwise" %}    
    ck_tile::FillNormalDistributionIntegerValue<BiasDataType>{-3.f, 3.f, {{seed}}}(bias_host);
{% endif %}

{% elif init_method_str == "uf" %}
    ck_tile::FillUniformDistribution<QDataType>{0.f, 1.f, {{seed}}}(q_host);
    ck_tile::FillUniformDistribution<KDataType>{0.f, 1.f, {{seed}}}(k_host);
    ck_tile::FillUniformDistribution<VDataType>{0.f, 1.f, {{seed}}}(v_host);
{% if bias_str == "elementwise" %}
    ck_tile::FillUniformDistribution<BiasDataType>{0.f, 1.f, {{seed}}}(bias_host);
{% endif %}

{% elif init_method_str == "nf" %}
    ck_tile::FillNormalDistribution<QDataType>{0.f, 3.f, {{seed}}}(q_host);
    ck_tile::FillNormalDistribution<KDataType>{0.f, 3.f, {{seed}}}(k_host);
    ck_tile::FillNormalDistribution<VDataType>{0.f, 3.f, {{seed}}}(v_host);
{% if bias_str == "elementwise" %}
    ck_tile::FillNormalDistribution<BiasDataType>{0.f, 3.f, {{seed}}}(bias_host);
{% endif %}

{% elif init_method_str == "tf" %}
    ck_tile::FillTrigValue<QDataType>{}(q_host);
    ck_tile::FillTrigValue<KDataType>{}(k_host);
    ck_tile::FillTrigValue<VDataType>{}(v_host);
{% if bias_str == "elementwise" %}
    ck_tile::FillTrigValue<BiasDataType>{}(bias_host);
{% endif %}

{% elif init_method_str == "uf8q" %}
    ck_tile::FillUniformDistribution<QDataType>{-dtype_max, dtype_max, {{seed}}}(q_host);
    ck_tile::FillUniformDistribution<KDataType>{-dtype_max, dtype_max, {{seed}}}(k_host);
    ck_tile::FillUniformDistribution<VDataType>{-dtype_max, dtype_max, {{seed}}}(v_host);
{% if bias_str == "elementwise" %}
    // bias_fp8 = qscale_bias * bias_fp32
    float qscale_bias = (dtype_max / range_q) * (dtype_max / range_k);
    // Assume bias is in [-1.f, 1.f] in original fp32
    ck_tile::FillUniformDistribution<BiasDataType>{-qscale_bias, qscale_bias, {{seed}}}(bias_host);
{% endif %}

{% endif %}

{% if bias_str == "alibi" %}
    auto slopes = ck_tile::get_alibi_slopes<SaccDataType>(nhead_q);
    assert(slopes.size() == static_cast<std::size_t>(nhead_q));
{% if bias_rank_info == 0 %}
    // alibi in 1*h
    std::copy(slopes.begin(), slopes.end(), alibi_slope_host.begin());
{% else %}
    // alibi in b*h
    for(auto i_b = 0; i_b < batch; i_b++)
    {
        std::copy(slopes.begin(), slopes.end(), alibi_slope_host.begin() + i_b * nhead_q);
    }
{% endif %}
{% endif %}

    ck_tile::DeviceMem q_buf(q_host.get_element_space_size_in_bytes());
    q_buf.ToDevice(q_host.data());
    ck_tile::DeviceMem k_buf(k_host.get_element_space_size_in_bytes());
    k_buf.ToDevice(k_host.data());
    ck_tile::DeviceMem v_buf(v_host.get_element_space_size_in_bytes());
    v_buf.ToDevice(v_host.data());
{% if bias_str == "elementwise" %}
    ck_tile::DeviceMem bias_buf(bias_host.get_element_space_size_in_bytes());
    bias_buf.ToDevice(bias_host.data());
{% endif %}
{% if bias_str == "alibi" %}
    ck_tile::DeviceMem alibi_slope_buf(alibi_slope_host.get_element_space_size_in_bytes());
    alibi_slope_buf.ToDevice(alibi_slope_host.data());
{% endif %}
    ck_tile::DeviceMem o_buf(o_host.get_element_space_size_in_bytes());

{% if mode_str == "group" %}
    ck_tile::DeviceMem seqstart_q(seqstart_q_host.size() * sizeof(int64_t));
    seqstart_q.ToDevice(seqstart_q_host.data());
    ck_tile::DeviceMem seqstart_k(seqstart_k_host.size() * sizeof(int64_t));
    seqstart_k.ToDevice(seqstart_k_host.data());
{% endif %}

{% if mode_str == "group" %}
    ck_tile::DeviceMem seqlen_k_buf(0 <= seqlen_kpads[0] ? seqlen_ks.size() * sizeof(int64_t) : 0);
    seqlen_k_buf.ToDevice(seqlen_ks.data());
{% endif %}

)";