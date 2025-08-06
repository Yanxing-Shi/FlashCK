#pragma once

#include <string>

static const std::string g_fmha_fwd_append_kv_create_args_tpl = R"(
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
            "q_seq_len. if group-mode, means the average value of q_seq_len\n"
            "total_q_seq_len = q_seq_len * batch, and q_seq_len per batch may vary\n"
            "also with \"-s=s0,s1,s2...\" comma seperated int to set per batch seqlen group-mode")
        .insert("s_k", "-1", "cache_seqlen_k, -1 means equal to s")
        .insert("s_knew",
                "0",
                "kv_seq_len for new key/value, 0 means not to use this at all; "
                "-1 to choose s_knew in [1, s] randomly.")
        .insert("d", "128", "head dim for q, k")
        .insert("d_v", "-1", "head dim for v, -1 means equal to d")
        .insert("has_mask", "0", "has mask, 0: false, 1: true")
        .insert("rotary_dim", "0", "RoPE rotary dimension. rotary_dim <= 0 means not apply RoPE at all")
        .insert("paged_block_size", "0", "paged-kvcache block size. 0 means not use paged-kvcahe")
        .insert("use_cache_batch_idx", "0", "whether to use index map to the kvcache");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
)";

static const std::string g_fmha_fwd_append_kv_args_parser_tpl = R"(
    ck_tile::index_t batch   = arg_parser.get_int("b");
    ck_tile::index_t q_seq_len = arg_parser.get_int("s");
    ck_tile::index_t kv_seq_len = arg_parser.get_int("s_k");
    ck_tile::index_t q_num_heads = arg_parser.get_int("h");
    ck_tile::index_t kv_num_heads = arg_parser.get_int("h_k");
    if(kv_num_heads < 0)
        kv_num_heads = q_num_heads;

    if(q_num_heads % kv_num_heads != 0)
    {
        std::cerr << "q_num_heads:" << q_num_heads << " must be multiple of kv_num_heads:" << kv_num_heads << std::endl;
    }

    ck_tile::index_t qk_head_dim = arg_parser.get_int("d");
    ck_tile::index_t v_head_dim = arg_parser.get_int("d_v");
    if(v_head_dim < 0)
        v_head_dim = qk_head_dim;
    
    ck_tile::index_t new_kv_seq_len = arg_parser.get_int("s_knew");
    if(new_kv_seq_len < 0)
    {
        new_kv_seq_len = randint<ck_tile::index_t>(1, arg_parser.get_int("s"), 1234);
    }

    bool has_mask = arg_parser.get_bool("has_mask");

    ck_tile::index_t rotary_dim = arg_parser.get_int("rotary_dim");

    ck_tile::index_t paged_block_size = arg_parser.get_int("paged_block_size");

    bool use_cache_batch_idx = arg_parser.get_bool("use_cache_batch_idx");

)";

static const std::string g_fmha_fwd_append_kv_args_decl_tpl = R"(

struct FmhaFwdAppendKVArgs
{
    void* q_ptr;
    void* k_ptr;
    const void* new_k_ptr;
    void* v_ptr;
    const void* new_v_ptr;

    const void* q_seq_len_ptr;

    ck_tile::index_t q_seq_len;
    ck_tile::index_t new_kv_seq_len;
    ck_tile::index_t batch;
    ck_tile::index_t qk_head_dim;
    ck_tile::index_t v_head_dim;
    ck_tile::index_t q_num_heads;
    ck_tile::index_t kv_num_heads;

    const void* rotary_cos_ptr; // only used if 'rotary_dim' > 0
    const void* rotary_sin_ptr; // only used if 'rotary_dim' > 0
    ck_tile::index_t rotary_dim;
    bool has_mask;

    void* block_table_ptr;
    ck_tile::index_t batch_stride_block_table; // only used if 'block_table_ptr' is not nullptr
    ck_tile::index_t paged_block_size;          // only used if 'block_table_ptr' is not nullptr

    const void* cache_batch_idx;

    ck_tile::index_t q_stride;
    ck_tile::index_t k_stride;
    ck_tile::index_t new_k_stride;
    ck_tile::index_t v_stride;
    ck_tile::index_t new_v_stride;

    ck_tile::index_t q_num_heads_stride;
    ck_tile::index_t k_num_heads_stride;
    ck_tile::index_t new_k_num_heads_stride;
    ck_tile::index_t v_num_heads_stride;
    ck_tile::index_t new_v_num_heads_stride;

    ck_tile::index_t q_batch_stride;
    ck_tile::index_t k_batch_stride;
    ck_tile::index_t new_k_batch_stride;
    ck_tile::index_t v_batch_stride;
    ck_tile::index_t new_v_batch_stride;
};

)";

static const std::string g_fmha_fwd_append_kv_func_signature_tpl = R"(
void {{function_name}}(
        void* q_buf_ptr,
        void* k_buf_ptr,
        void* v_buf_ptr,
        void* knew_buf_ptr,
        void* vnew_buf_ptr,
        void* cache_seqlen_k_buf_ptr,
        void* rotary_cos_buf_ptr,
        void* rotary_sin_buf_ptr,
        void* block_table_buf_ptr,
        int64_t* cache_batch_idx_buf_ptr,
        int64_t batch,
        int64_t q_seq_len,
        int64_t kv_seq_len,
        int64_t q_num_heads,
        int64_t kv_num_heads,
        int64_t qk_head_dim,
        int64_t v_head_dim,
        int64_t new_kv_seq_len,
        int64_t max_num_page_blocks,
        int64_t paged_block_size,
        int64_t rotary_dim,
        bool has_mask,
        hipStream_t stream
    )
)";

static const std::string g_fmha_fwd_append_kv_func_call_tpl = R"(
    {{function_name}}(
        q_buf.GetDeviceBuffer(),
        k_buf.GetDeviceBuffer(),
        v_buf.GetDeviceBuffer(),
        knew_buf.GetDeviceBuffer(),
        vnew_buf.GetDeviceBuffer(),
        k_cache_seq_len_buf.GetDeviceBuffer(),
        rotary_cos_buf.GetDeviceBuffer(),
        rotary_sin_buf.GetDeviceBuffer(),
{% if paged_block_size > 0 %}
        block_table_buf.GetDeviceBuffer(),
{% else %}
        nullptr,
{% endif %}
{% if use_cache_batch_idx %}
        cache_batch_idx_buf.GetDeviceBuffer(),
{% else %}
        nullptr,
{% endif %}
        batch,
        q_shape_seq_len,
        kv_shape_seq_len,
        q_num_heads,
        kv_num_heads,
        qk_head_dim,
        v_head_dim,
        new_kv_seq_len,
        max_num_page_blocks,
        paged_block_size,
        rotary_dim,
{% if mask != no %}
        true,
{% else %}
        false,
{% endif %}
        stream
    );
)";

static const std::string g_fmha_fwd_append_kv_prepare_args_tpl = R"(
   const auto init_args = [&](auto& args){  
    /// NOTE: we broadcast bias from [1, 1, q_seq_len, kv_seq_len] to [batch, nhead, q_seq_len,
    ///       kv_seq_len] in this example, hence both the 'batch_stride_bias' &
    ///       'nhead_stride_bias' are 0.

    const ck_tile::index_t shape_batch = {% if mode == "batch" %} batch; {% else %} 1; {% endif %}
    const ck_tile::index_t q_shape_seq_len = {% if mode == "batch" %} q_seq_len; {% else %} seqstart_q_ptr[sizeof(seqstart_q_ptr)/sizeof(seqstart_q_ptr[0]) - 1]; {% endif %}
    const ck_tile::index_t kv_shape_seq_len =
{% if mode == "batch" %} kv_seq_len; {% else %} seqstart_k_ptr[sizeof(seqstart_q_ptr)/sizeof(seqstart_q_ptr[0]) - 1]; {% endif %}
    
    // setup stride_* arguments
    const ck_tile::index_t q_stride = q_num_heads * qk_head_dim;
    const ck_tile::index_t k_stride = kv_num_heads * qk_head_dim;
    const ck_tile::index_t v_stride = kv_num_heads * v_head_dim;
    const ck_tile::index_t new_k_stride = kv_num_heads * qk_head_dim;
    const ck_tile::index_t new_v_stride = kv_num_heads * v_head_dim;

    // setup nhead_stride_* arguments
    const ck_tile::index_t q_num_heads_stride = qk_head_dim;
    const ck_tile::index_t k_num_heads_stride = qk_head_dim;
    const ck_tile::index_t v_num_heads_stride = v_head_dim;
    const ck_tile::index_t new_k_num_heads_stride = qk_head_dim;
    const ck_tile::index_t new_v_num_heads_stride = v_head_dim;

    // setup batch_stride_* arguments
    const ck_tile::index_t q_batch_stride = q_num_heads * q_shape_seq_len * qk_head_dim;
    const ck_tile::index_t k_batch_stride = kv_num_heads * kv_shape_seq_len * qk_head_dim;
    const ck_tile::index_t v_batch_stride = kv_num_heads * v_head_dim * kv_seq_len;
    const ck_tile::index_t new_k_batch_stride = kv_num_heads * new_kv_seq_len * qk_head_dim;
    const ck_tile::index_t new_v_batch_stride    = kv_num_heads * v_head_dim * new_kv_seq_len;
    const ck_tile::index_t batch_stride_block_table = max_num_page_blocks / batch;

    args.q_ptr = q_buf_ptr;
    args.k_ptr = k_buf_ptr;
    args.v_ptr = v_buf_ptr;

    args.batch    = batch;
    args.q_seq_len = q_shape_seq_len; // unused in group mode
    args.qk_head_dim   = qk_head_dim;
    args.v_head_dim   = v_head_dim;
    args.q_num_heads  = q_num_heads;
    args.kv_num_heads  = kv_num_heads;

    args.q_stride       = q_stride;
    args.k_stride       = k_stride;
    args.v_stride       = v_stride;
    args.q_num_heads_stride = q_num_heads_stride;
    args.k_num_heads_stride = k_num_heads_stride;
    args.v_num_heads_stride = v_num_heads_stride;
    args.q_batch_stride = q_batch_stride;
    args.k_batch_stride = k_batch_stride;
    args.v_batch_stride = v_batch_stride;

    args.new_k_ptr    = knew_buf_ptr;
    args.new_v_ptr    = vnew_buf_ptr;
    args.new_kv_seq_len = new_kv_seq_len;

    args.q_seq_len_ptr = cache_seqlen_k_buf_ptr;

    args.rotary_cos_ptr = rotary_cos_buf_ptr;
    args.rotary_sin_ptr = rotary_sin_buf_ptr;
    args.rotary_dim     = rotary_dim;
    args.has_mask       = has_mask;

    args.block_table_ptr = block_table_buf_ptr;
    args.batch_stride_block_table = batch_stride_block_table;
    args.paged_block_size          = paged_block_size;

    args.cache_batch_idx = cache_batch_idx_buf_ptr;

    args.new_k_stride       = new_k_stride;
    args.new_v_stride       = new_v_stride;
    args.new_k_num_heads_stride = new_k_num_heads_stride;
    args.new_v_num_heads_stride = new_v_num_heads_stride;
    args.new_k_batch_stride = new_k_batch_stride;
    args.new_v_batch_stride = new_v_batch_stride;
    };
)";

static const std::string g_fmha_fwd_append_kv_make_args_tpl = R"(
    FmhaFwdAppendKVArgs fmha_fwd_append_kv_args;
    init_args(fmha_fwd_append_kv_args);
    
    auto kargs = {{kernel_name}}::MakeKargs(fmha_fwd_append_kv_args.q_ptr,
                                            fmha_fwd_append_kv_args.k_ptr,
                                            fmha_fwd_append_kv_args.new_k_ptr,
                                            fmha_fwd_append_kv_args.v_ptr,
                                            fmha_fwd_append_kv_args.new_v_ptr,
                                            fmha_fwd_append_kv_args.q_seq_len,
                                            fmha_fwd_append_kv_args.q_seq_len_ptr,
                                            fmha_fwd_append_kv_args.new_kv_seq_len,
                                            fmha_fwd_append_kv_args.qk_head_dim,
                                            fmha_fwd_append_kv_args.v_head_dim,
                                            fmha_fwd_append_kv_args.q_num_heads,
                                            fmha_fwd_append_kv_args.q_num_heads / fmha_fwd_append_kv_args.kv_num_heads,
                                            fmha_fwd_append_kv_args.rotary_cos_ptr,
                                            fmha_fwd_append_kv_args.rotary_sin_ptr,
                                            fmha_fwd_append_kv_args.rotary_dim,
                                            fmha_fwd_append_kv_args.has_mask,
                                            fmha_fwd_append_kv_args.block_table_ptr,
                                            fmha_fwd_append_kv_args.batch_stride_block_table,
                                            fmha_fwd_append_kv_args.paged_block_size,
                                            fmha_fwd_append_kv_args.cache_batch_idx,
                                            fmha_fwd_append_kv_args.q_stride,
                                            fmha_fwd_append_kv_args.k_stride,
                                            fmha_fwd_append_kv_args.new_k_stride,
                                            fmha_fwd_append_kv_args.v_stride,
                                            fmha_fwd_append_kv_args.new_v_stride,
                                            fmha_fwd_append_kv_args.q_num_heads_stride,
                                            fmha_fwd_append_kv_args.k_num_heads_stride,
                                            fmha_fwd_append_kv_args.new_k_num_heads_stride,
                                            fmha_fwd_append_kv_args.v_num_heads_stride,
                                            fmha_fwd_append_kv_args.new_v_num_heads_stride,
                                            fmha_fwd_append_kv_args.q_batch_stride,
                                            fmha_fwd_append_kv_args.k_batch_stride,
                                            fmha_fwd_append_kv_args.new_k_batch_stride,
                                            fmha_fwd_append_kv_args.v_batch_stride,
                                            fmha_fwd_append_kv_args.new_v_batch_stride);
    dim3 grids = {{kernel_name}}::GridSize(fmha_fwd_append_kv_args.batch, fmha_fwd_append_kv_args.q_num_heads, fmha_fwd_append_kv_args.q_seq_len, fmha_fwd_append_kv_args.new_kv_seq_len);

)";

static const std::string g_fmha_fwd_append_kv_tensor_decl_tpl = R"(
    auto kv_cache_seq_len = seqlen_ks;
    std::transform(kv_cache_seq_len.begin(),
                   kv_cache_seq_len.end(),
                   kv_cache_seq_len.begin(),
                   [&](auto kv_seq_len) { return kv_seq_len - new_kv_seq_len; });
    
    const ck_tile::index_t max_num_page_blocks = paged_block_size > 0 ? batch * std::max(1, ck_tile::integer_divide_ceil(max_seqlen_k, paged_block_size)) : 0;
    
    ck_tile::HostTensor<QDataType> q_host(
        {shape_batch, q_shape_seq_len, q_num_heads, qk_head_dim});
    ck_tile::HostTensor<KDataType> k_host(
{% if paged_block_size > 0 %}
        {max_num_page_blocks, paged_block_size, kv_num_heads, qk_head_dim});
{% else %}
        {shape_batch, kv_shape_seq_len, kv_num_heads, qk_head_dim});
{% endif %}
    ck_tile::HostTensor<VDataType> v_host(
{% if paged_block_size > 0 %}
        {max_num_page_blocks, paged_block_size, kv_num_heads, v_head_dim});
{% else %}
        {shape_batch, kv_shape_seq_len, kv_num_heads, v_head_dim});
{% endif %}
    ck_tile::HostTensor<ODataType> o_host(
        {shape_batch, q_shape_seq_len, q_num_heads, v_head_dim});
    
    /// NOTICE: always use same shape for knew_host & vnew_host in batch/group mode
    ck_tile::HostTensor<KDataType> knew_host({batch, new_kv_seq_len, kv_num_heads, qk_head_dim});
    ck_tile::HostTensor<VDataType> vnew_host({batch, new_kv_seq_len, kv_num_heads, v_head_dim});

{% if paged_block_size > 0 %}
    ck_tile::HostTensor<int32_t> block_table_host(std::array<ck_tile::index_t, 2>{batch, max_num_page_blocks / batch});
    iota_shuffle(block_table_host.begin(), block_table_host.end(), 0);
{% endif %}

{% if use_cache_batch_idx %}
    ck_tile::HostTensor<int32_t> cache_batch_idx_host(use_cache_batch_idx, std::array<ck_tile::index_t, 1>{batch});
    iota_shuffle(cache_batch_idx_host.begin(), cache_batch_idx_host.end(), 0);
{% endif %}

)";

const static std::string g_fmha_fwd_append_kv_tensor_generate_tpl = R"(
    auto [rotary_cos_host, rotary_sin_host] = generate_rotary_cos_sin<KDataType>(
        std::max(q_shape_seq_len, kv_shape_seq_len), {{rotary_dim}}, {{seed}});

{% if init_method == "uri" %}
    ck_tile::FillUniformDistributionIntegerValue<QDataType>{-3.f, 3.f, {{seed}}}(q_host);
    ck_tile::FillUniformDistributionIntegerValue<KDataType>{-3.f, 3.f, {{seed}}}(k_host);
    ck_tile::FillUniformDistributionIntegerValue<VDataType>{-3.f, 3.f, {{seed}}}(v_host);
    ck_tile::FillUniformDistributionIntegerValue<KDataType>{-3.f, 3.f, {{seed}}}(knew_host);
    ck_tile::FillUniformDistributionIntegerValue<VDataType>{-3.f, 3.f, {{seed}}}(vnew_host);

{% elif init_method == "nri" %}
    ck_tile::FillNormalDistributionIntegerValue<QDataType>{-3.f, 3.f, {{seed}}}(q_host);
    ck_tile::FillNormalDistributionIntegerValue<KDataType>{-3.f, 3.f, {{seed}}}(k_host);
    ck_tile::FillNormalDistributionIntegerValue<VDataType>{-3.f, 3.f, {{seed}}}(v_host);
    ck_tile::FillNormalDistributionIntegerValue<KDataType>{-3.f, 3.f, {{seed}}}(knew_host);
    ck_tile::FillNormalDistributionIntegerValue<VDataType>{-3.f, 3.f, {{seed}}}(vnew_host);

{% elif init_method == "uf" %}
    ck_tile::FillUniformDistribution<QDataType>{0.f, 1.f, {{seed}}}(q_host);
    ck_tile::FillUniformDistribution<KDataType>{0.f, 1.f, {{seed}}}(k_host);
    ck_tile::FillUniformDistribution<VDataType>{0.f, 1.f, {{seed}}}(v_host);
    ck_tile::FillUniformDistribution<KDataType>{0.f, 1.f, {{seed}}}(knew_host);
    ck_tile::FillUniformDistribution<VDataType>{0.f, 1.f, {{seed}}}(vnew_host);

{% elif init_method == "nf" %}
    ck_tile::FillNormalDistribution<QDataType>{0.f, 3.f, {{seed}}}(q_host);
    ck_tile::FillNormalDistribution<KDataType>{0.f, 3.f, {{seed}}}(k_host);
    ck_tile::FillNormalDistribution<VDataType>{0.f, 3.f, {{seed}}}(v_host); 
    ck_tile::FillNormalDistribution<KDataType>{0.f, 3.f, {{seed}}}(knew_host);
    ck_tile::FillNormalDistribution<VDataType>{0.f, 3.f, {{seed}}}(vnew_host);

{% elif init_method == "tf" %}
    ck_tile::FillTrigValue<QDataType>{}(q_host);
    ck_tile::FillTrigValue<KDataType>{}(k_host);
    ck_tile::FillTrigValue<VDataType>{}(v_host);
    ck_tile::FillTrigValue<KDataType>{}(knew_host);
    ck_tile::FillTrigValue<VDataType>{}(vnew_host);

{% elif init_method == "uf8q" %}
    ck_tile::FillUniformDistribution<QDataType>{-dtype_max, dtype_max, {{seed}}}(q_host);
    ck_tile::FillUniformDistribution<KDataType>{-dtype_max, dtype_max, {{seed}}}(k_host);
    ck_tile::FillUniformDistribution<VDataType>{-dtype_max, dtype_max, {{seed}}}(v_host);

    ck_tile::FillUniformDistribution<KDataType>{-dtype_max, dtype_max, {{seed}}}(knew_host);
    ck_tile::FillUniformDistribution<VDataType>{-dtype_max, dtype_max, {{seed}}}(vnew_host);

{% endif %}

    ck_tile::DeviceMem q_buf(q_host.get_element_space_size_in_bytes());
    q_buf.ToDevice(q_host.data());
    ck_tile::DeviceMem k_buf(k_host.get_element_space_size_in_bytes());
    k_buf.ToDevice(k_host.data());
    ck_tile::DeviceMem v_buf(v_host.get_element_space_size_in_bytes());
    v_buf.ToDevice(v_host.data());
    ck_tile::DeviceMem o_buf(o_host.get_element_space_size_in_bytes());

    ck_tile::DeviceMem knew_buf(knew_host.get_element_space_size_in_bytes());
    knew_buf.ToDevice(knew_host.data());
    ck_tile::DeviceMem vnew_buf(vnew_host.get_element_space_size_in_bytes());
    vnew_buf.ToDevice(vnew_host.data());
    ck_tile::DeviceMem rotary_cos_buf(rotary_cos_host.get_element_space_size_in_bytes());
    rotary_cos_buf.ToDevice(rotary_cos_host.data());
    ck_tile::DeviceMem rotary_sin_buf(rotary_sin_host.get_element_space_size_in_bytes());
    rotary_sin_buf.ToDevice(rotary_sin_host.data());

    ck_tile::DeviceMem k_cache_seq_len_buf(kv_cache_seq_len.size() * sizeof(int64_t));
    k_cache_seq_len_buf.ToDevice(kv_cache_seq_len.data());
    
{% if paged_block_size > 0 %}
    ck_tile::DeviceMem block_table_buf(block_table_host.get_element_space_size_in_bytes());
    block_table_buf.ToDevice(block_table_host.data());
{% endif %}

{% if use_cache_batch_idx %}
    ck_tile::DeviceMem cache_batch_idx_buf(cache_batch_idx_host.get_element_space_size_in_bytes());
    cache_batch_idx_buf.ToDevice(cache_batch_idx_host.data());
{% endif %}

)";