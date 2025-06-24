#pragma once

#include "flashck/core/module/kernels/fmha_kernels/fmha_common_kernel.h"

#include "flashck/core/module/kernels/kernel_registry.h"

static const std::string g_fmha_fwd_appendkv_create_args_source = R"(
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
        .insert("s_k", "-1", "cache_seqlen_k, -1 means equal to s")
        .insert("s_knew",
                "0",
                "seqlen_k for new key/value, 0 means not to use this at all; "
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

static const std::string g_fmha_fwd_appendkv_args_parser_source = R"(
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
    
    ck_tile::index_t seqlen_knew = arg_parser.get_int("s_knew");
    if(seqlen_knew < 0)
    {
        seqlen_knew = randint<ck_tile::index_t>(1, arg_parser.get_int("s"), 1234);
    }

    bool has_mask = arg_parser.get_bool("has_mask");

    ck_tile::index_t rotary_dim = arg_parser.get_int("rotary_dim");

    ck_tile::index_t paged_block_size = arg_parser.get_int("paged_block_size");

    bool use_cache_batch_idx = arg_parser.get_bool("use_cache_batch_idx");

)";

static const std::string g_fmha_fwd_appendkv_args_decl_source = R"(

struct FmhaFwdAppendKVArgs
{
    void* q_ptr;
    void* k_ptr;
    const void* knew_ptr;
    void* v_ptr;
    const void* vnew_ptr;

    const void* seqlen_k_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_knew;
    ck_tile::index_t batch;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    const void* rotary_cos_ptr; // only used if 'rotary_dim' > 0
    const void* rotary_sin_ptr; // only used if 'rotary_dim' > 0
    ck_tile::index_t rotary_dim;
    bool has_mask;

    void* block_table_ptr;
    ck_tile::index_t batch_stride_block_table; // only used if 'block_table_ptr' is not nullptr
    ck_tile::index_t paged_block_size;          // only used if 'block_table_ptr' is not nullptr

    const void* cache_batch_idx;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_knew;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_vnew;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_knew;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_vnew;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_knew;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_vnew;
};

)";

static const std::string g_fmha_fwd_appendkv_func_signature_source = R"(
    {% if is_execute %} {{c_flag}} FC_EXPORT {% endif %} void {{function_name}}(
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
        int64_t seqlen_q,
        int64_t seqlen_k,
        int64_t nhead_q,
        int64_t nhead_k,
        int64_t hdim_q,
        int64_t hdim_v,
        int64_t seqlen_knew,
        int64_t max_num_page_blocks,
        int64_t paged_block_size,
        int64_t rotary_dim,
        bool has_mask,
        hipStream_t stream
    )
)";

static const std::string g_fmha_fwd_appendkv_func_call_source = R"(
    {{function_name}}(
        q_buf.GetDeviceBuffer(),
        k_buf.GetDeviceBuffer(),
        v_buf.GetDeviceBuffer(),
        knew_buf.GetDeviceBuffer(),
        vnew_buf.GetDeviceBuffer(),
        cache_seqlen_k_buf.GetDeviceBuffer(),
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
        shape_seqlen_q,
        shape_seqlen_k,
        nhead_q,
        nhead_k,
        hdim_q,
        hdim_v,
        seqlen_knew,
        max_num_page_blocks,
        paged_block_size,
        rotary_dim,
{% if mask_str != no %}
        true,
{% else %}
        false,
{% endif %}
        stream
    );
)";

static const std::string g_fmha_fwd_appendkv_prepare_args_source = R"(
   const auto init_args = [&](auto& args){  
    /// NOTE: we broadcast bias from [1, 1, seqlen_q, seqlen_k] to [batch, nhead, seqlen_q,
    ///       seqlen_k] in this example, hence both the 'batch_stride_bias' &
    ///       'nhead_stride_bias' are 0.

    const ck_tile::index_t shape_batch = {% if mode_str == "batch" %} batch; {% else %} 1; {% endif %}
    const ck_tile::index_t shape_seqlen_q = {% if mode_str == "batch" %} seqlen_q; {% else %} seqstart_q_ptr[sizeof(seqstart_q_ptr)/sizeof(seqstart_q_ptr[0]) - 1]; {% endif %}
    const ck_tile::index_t shape_seqlen_k =
{% if mode_str == "batch" %} seqlen_k; {% else %} seqstart_k_ptr[sizeof(seqstart_q_ptr)/sizeof(seqstart_q_ptr[0]) - 1]; {% endif %}
    
    // setup stride_* arguments
    const ck_tile::index_t stride_q = nhead_q * hdim_q;
    const ck_tile::index_t stride_k = nhead_k * hdim_q;
    const ck_tile::index_t stride_v = nhead_k * hdim_v;
    const ck_tile::index_t stride_knew = nhead_k * hdim_q;
    const ck_tile::index_t stride_vnew = nhead_k * hdim_v;
    const ck_tile::index_t stride_o       = nhead_q * hdim_v;

    // setup nhead_stride_* arguments
    const ck_tile::index_t nhead_stride_q = hdim_q;
    const ck_tile::index_t nhead_stride_k = hdim_q;
    const ck_tile::index_t nhead_stride_v = hdim_v;
    const ck_tile::index_t nhead_stride_knew = hdim_q;
    const ck_tile::index_t nhead_stride_vnew = hdim_v;
    const ck_tile::index_t nhead_stride_o    = hdim_v;

    // setup batch_stride_* arguments
    const ck_tile::index_t batch_stride_q = nhead_q * shape_seqlen_q * hdim_q;
    const ck_tile::index_t batch_stride_k = nhead_k * shape_seqlen_k * hdim_q;
    const ck_tile::index_t batch_stride_v = nhead_k * hdim_v * seqlen_k;
    const ck_tile::index_t batch_stride_knew = nhead_k * seqlen_knew * hdim_q;
    const ck_tile::index_t batch_stride_vnew    = nhead_k * hdim_v * seqlen_knew;
    const ck_tile::index_t batch_stride_block_table = max_num_page_blocks / batch;
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

    args.knew_ptr    = knew_buf_ptr;
    args.vnew_ptr    = vnew_buf_ptr;
    args.seqlen_knew = seqlen_knew;

    args.seqlen_k_ptr = cache_seqlen_k_buf_ptr;

    args.rotary_cos_ptr = rotary_cos_buf_ptr;
    args.rotary_sin_ptr = rotary_sin_buf_ptr;
    args.rotary_dim     = rotary_dim;
    args.has_mask       = has_mask;

    args.block_table_ptr = block_table_buf_ptr;
    args.batch_stride_block_table = batch_stride_block_table;
    args.paged_block_size          = paged_block_size;

    args.cache_batch_idx = cache_batch_idx_buf_ptr;

    args.stride_knew       = stride_knew;
    args.stride_vnew       = stride_vnew;
    args.nhead_stride_knew = nhead_stride_knew;
    args.nhead_stride_vnew = nhead_stride_vnew;
    args.batch_stride_knew = batch_stride_knew;
    args.batch_stride_vnew = batch_stride_vnew;
    };
)";

static const std::string g_fmha_fwd_appendkv_make_args_source = R"(
    FmhaFwdAppendKVArgs fmha_fwd_appendkv_args;
    init_args(fmha_fwd_appendkv_args);
    
    auto kargs = {{kernel_name}}::MakeKargs(fmha_fwd_appendkv_args.q_ptr,
                                            fmha_fwd_appendkv_args.k_ptr,
                                            fmha_fwd_appendkv_args.knew_ptr,
                                            fmha_fwd_appendkv_args.v_ptr,
                                            fmha_fwd_appendkv_args.vnew_ptr,
                                            fmha_fwd_appendkv_args.seqlen_q,
                                            fmha_fwd_appendkv_args.seqlen_k_ptr,
                                            fmha_fwd_appendkv_args.seqlen_knew,
                                            fmha_fwd_appendkv_args.hdim_q,
                                            fmha_fwd_appendkv_args.hdim_v,
                                            fmha_fwd_appendkv_args.nhead_q,
                                            fmha_fwd_appendkv_args.nhead_q / fmha_fwd_appendkv_args.nhead_k,
                                            fmha_fwd_appendkv_args.rotary_cos_ptr,
                                            fmha_fwd_appendkv_args.rotary_sin_ptr,
                                            fmha_fwd_appendkv_args.rotary_dim,
                                            fmha_fwd_appendkv_args.has_mask,
                                            fmha_fwd_appendkv_args.block_table_ptr,
                                            fmha_fwd_appendkv_args.batch_stride_block_table,
                                            fmha_fwd_appendkv_args.paged_block_size,
                                            fmha_fwd_appendkv_args.cache_batch_idx,
                                            fmha_fwd_appendkv_args.stride_q,
                                            fmha_fwd_appendkv_args.stride_k,
                                            fmha_fwd_appendkv_args.stride_knew,
                                            fmha_fwd_appendkv_args.stride_v,
                                            fmha_fwd_appendkv_args.stride_vnew,
                                            fmha_fwd_appendkv_args.nhead_stride_q,
                                            fmha_fwd_appendkv_args.nhead_stride_k,
                                            fmha_fwd_appendkv_args.nhead_stride_knew,
                                            fmha_fwd_appendkv_args.nhead_stride_v,
                                            fmha_fwd_appendkv_args.nhead_stride_vnew,
                                            fmha_fwd_appendkv_args.batch_stride_q,
                                            fmha_fwd_appendkv_args.batch_stride_k,
                                            fmha_fwd_appendkv_args.batch_stride_knew,
                                            fmha_fwd_appendkv_args.batch_stride_v,
                                            fmha_fwd_appendkv_args.batch_stride_vnew);
    dim3 grids = {{kernel_name}}::GridSize(fmha_fwd_appendkv_args.batch, fmha_fwd_appendkv_args.nhead_q, fmha_fwd_appendkv_args.seqlen_q, fmha_fwd_appendkv_args.seqlen_knew);

)";

static const std::string g_fmha_fwd_appendkv_tensor_decl_source = R"(
    auto cache_seqlen_ks = seqlen_ks;
    std::transform(cache_seqlen_ks.begin(),
                   cache_seqlen_ks.end(),
                   cache_seqlen_ks.begin(),
                   [&](auto seqlen_k) { return seqlen_k - seqlen_knew; });
    
    const ck_tile::index_t max_num_page_blocks = paged_block_size > 0 ? batch * std::max(1, ck_tile::integer_divide_ceil(max_seqlen_k, paged_block_size)) : 0;
    
    ck_tile::HostTensor<QDataType> q_host(
        {shape_batch, shape_seqlen_q, nhead_q, hdim_q});
    ck_tile::HostTensor<KDataType> k_host(
{% if paged_block_size > 0 %}
        {max_num_page_blocks, paged_block_size, nhead_k, hdim_q});
{% else %}
        {shape_batch, shape_seqlen_k, nhead_k, hdim_q});
{% endif %}
    ck_tile::HostTensor<VDataType> v_host(
{% if paged_block_size > 0 %}
        {max_num_page_blocks, paged_block_size, nhead_k, hdim_v});
{% else %}
        {shape_batch, shape_seqlen_k, nhead_k, hdim_v});
{% endif %}
    ck_tile::HostTensor<ODataType> o_host(
        {shape_batch, shape_seqlen_q, nhead_q, hdim_v});
    
    /// NOTICE: always use same shape for knew_host & vnew_host in batch/group mode
    ck_tile::HostTensor<KDataType> knew_host({batch, seqlen_knew, nhead_k, hdim_q});
    ck_tile::HostTensor<VDataType> vnew_host({batch, seqlen_knew, nhead_k, hdim_v});

{% if paged_block_size > 0 %}
    ck_tile::HostTensor<int32_t> block_table_host(std::array<ck_tile::index_t, 2>{batch, max_num_page_blocks / batch});
    iota_shuffle(block_table_host.begin(), block_table_host.end(), 0);
{% endif %}

{% if use_cache_batch_idx %}
    ck_tile::HostTensor<int32_t> cache_batch_idx_host(use_cache_batch_idx, std::array<ck_tile::index_t, 1>{batch});
    iota_shuffle(cache_batch_idx_host.begin(), cache_batch_idx_host.end(), 0);
{% endif %}

)";

const static std::string g_fmha_fwd_appendkv_tensor_generate_source = R"(
    auto [rotary_cos_host, rotary_sin_host] = generate_rotary_cos_sin<KDataType>(
        std::max(shape_seqlen_q, shape_seqlen_k), {{rotary_dim}}, {{seed}});

{% if init_method_str == "uri" %}
    ck_tile::FillUniformDistributionIntegerValue<QDataType>{-3.f, 3.f, {{seed}}}(q_host);
    ck_tile::FillUniformDistributionIntegerValue<KDataType>{-3.f, 3.f, {{seed}}}(k_host);
    ck_tile::FillUniformDistributionIntegerValue<VDataType>{-3.f, 3.f, {{seed}}}(v_host);
    ck_tile::FillUniformDistributionIntegerValue<KDataType>{-3.f, 3.f, {{seed}}}(knew_host);
    ck_tile::FillUniformDistributionIntegerValue<VDataType>{-3.f, 3.f, {{seed}}}(vnew_host);

{% elif init_method_str == "nri" %}
    ck_tile::FillNormalDistributionIntegerValue<QDataType>{-3.f, 3.f, {{seed}}}(q_host);
    ck_tile::FillNormalDistributionIntegerValue<KDataType>{-3.f, 3.f, {{seed}}}(k_host);
    ck_tile::FillNormalDistributionIntegerValue<VDataType>{-3.f, 3.f, {{seed}}}(v_host);
    ck_tile::FillNormalDistributionIntegerValue<KDataType>{-3.f, 3.f, {{seed}}}(knew_host);
    ck_tile::FillNormalDistributionIntegerValue<VDataType>{-3.f, 3.f, {{seed}}}(vnew_host);

{% elif init_method_str == "uf" %}
    ck_tile::FillUniformDistribution<QDataType>{0.f, 1.f, {{seed}}}(q_host);
    ck_tile::FillUniformDistribution<KDataType>{0.f, 1.f, {{seed}}}(k_host);
    ck_tile::FillUniformDistribution<VDataType>{0.f, 1.f, {{seed}}}(v_host);
    ck_tile::FillUniformDistribution<KDataType>{0.f, 1.f, {{seed}}}(knew_host);
    ck_tile::FillUniformDistribution<VDataType>{0.f, 1.f, {{seed}}}(vnew_host);

{% elif init_method_str == "nf" %}
    ck_tile::FillNormalDistribution<QDataType>{0.f, 3.f, {{seed}}}(q_host);
    ck_tile::FillNormalDistribution<KDataType>{0.f, 3.f, {{seed}}}(k_host);
    ck_tile::FillNormalDistribution<VDataType>{0.f, 3.f, {{seed}}}(v_host); 
    ck_tile::FillNormalDistribution<KDataType>{0.f, 3.f, {{seed}}}(knew_host);
    ck_tile::FillNormalDistribution<VDataType>{0.f, 3.f, {{seed}}}(vnew_host);

{% elif init_method_str == "tf" %}
    ck_tile::FillTrigValue<QDataType>{}(q_host);
    ck_tile::FillTrigValue<KDataType>{}(k_host);
    ck_tile::FillTrigValue<VDataType>{}(v_host);
    ck_tile::FillTrigValue<KDataType>{}(knew_host);
    ck_tile::FillTrigValue<VDataType>{}(vnew_host);

{% elif init_method_str == "uf8q" %}
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

    ck_tile::DeviceMem cache_seqlen_k_buf(cache_seqlen_ks.size() * sizeof(int64_t));
    cache_seqlen_k_buf.ToDevice(cache_seqlen_ks.data());
    
{% if paged_block_size > 0 %}
    ck_tile::DeviceMem block_table_buf(block_table_host.get_element_space_size_in_bytes());
    block_table_buf.ToDevice(block_table_host.data());
{% endif %}

{% if use_cache_batch_idx %}
    ck_tile::DeviceMem cache_batch_idx_buf(cache_batch_idx_host.get_element_space_size_in_bytes());
    cache_batch_idx_buf.ToDevice(cache_batch_idx_host.data());
{% endif %}

)";

namespace flashck {
class FmhaFwdAppendKVKernel: public FmhaCommonKernel {
public:
    FmhaFwdAppendKVKernel()  = default;
    ~FmhaFwdAppendKVKernel() = default;

    std::map<std::string, std::shared_ptr<void>> Init(const OperationKind&   op_kind,
                                                      const TensorOperation& extra_kind) override;

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenKernelProfiler(const std::string&                               model_name,
                      const std::unordered_map<std::string, std::any>& kernel_func_map,
                      const std::string&                               folder_name = "kernel_profile") override;

    std::string GenKernelFunction(const std::string&                               func_name,
                                  const std::string&                               model_name,
                                  const std::unordered_map<std::string, std::any>& kernel_func_map) override;

    void KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args) override;
};

}  // namespace flashck

flashck_REGISTER_KERNEL(CK_TILE, fmha_fwd_appendkv, flashck::FmhaFwdAppendKVKernel, ALL_LAYOUT, _Float16, ushort);
