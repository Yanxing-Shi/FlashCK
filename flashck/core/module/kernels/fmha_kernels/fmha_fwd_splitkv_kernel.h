#pragma once

#include "flashck/core/module/kernels/fmha_kernels/fmha_common_kernel.h"

#include "flashck/core/module/kernels/kernel_registry.h"

static const std::string g_fmha_fwd_splitkv_create_args_source = R"(
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
        .insert("num_splits",
                "1",
                "# of splits for key/value. 0 to determine actual number by heuristic")
        .insert("paged_block_size", "0", "paged-kvcache block size. 0 means not use paged-kvcahe")
        .insert("mask", "0", "mask type, 0: no mask, 1: left mask, 2: right mask")
        .insert("window_left_size", "-1", "the size of sliding window")
        .insert("window_right_size", "-1", "the size of sliding window");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
)";

static const std::string g_fmha_fwd_splitkv_args_parser_source = R"(
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
    
    ck_tile::index_t num_splits = arg_parser.get_int("num_splits");

    ck_tile::index_t paged_block_size = arg_parser.get_int("paged_block_size");

    uint32_t mask_type = arg_parser.get_uint32("mask");
    ck_tile::index_t window_left_size = arg_parser.get_int("window_left_size");
    ck_tile::index_t window_right_size = arg_parser.get_int("window_right_size");

)";

static const std::string g_fmha_fwd_splitkv_args_decl_source = R"(
struct FmhaFwdSplitKVArgs
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    void* lse_acc_ptr;
    void* o_acc_ptr;
    void* o_ptr;

    void* block_table_ptr;
    ck_tile::index_t batch_stride_block_table; // only used if 'block_table_ptr' is not nullptr
    ck_tile::index_t paged_block_size;          // only used if 'block_table_ptr' is not nullptr
    bool is_gappy; // differentiate seqstart_k_ptr usage. only used if 'block_table_ptr' is not
                   // nullptr.

    const void* cache_batch_idx;

    // the real seqlen_q & seqlen_k are decided by following:
    // batch mode: seqlen_q = kargs.seqlen_q
    //             seqlen_k = kargs.seqlen_k
    // group mode: seqlen_q = kargs.seqstart_q_ptr[b + 1] - kargs.seqstart_q_ptr[b]
    //             seqlen_k = kargs.seqstart_k_ptr[b + 1] - kargs.seqstart_k_ptr[b]
    // kvcache mode (use same kernel as batch mode):
    //             seqlen_q = kargs.seqlen_q
    //             seqlen_k = kargs.seqstart_k_ptr[b + 1] - kargs.seqstart_k_ptr[b]
    const void* seqstart_q_ptr;
    const void* seqstart_k_ptr;
    const void* seqlen_k_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;
    ck_tile::index_t num_splits;

    float scale_s;
    float scale_p;
    float scale_o;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_o_acc;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_lse_acc;
    ck_tile::index_t nhead_stride_o_acc;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_lse_acc;
    ck_tile::index_t batch_stride_o_acc;
    ck_tile::index_t batch_stride_o;
    ck_tile::index_t split_stride_lse_acc;
    ck_tile::index_t split_stride_o_acc;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;
};

)";

static const std::string g_fmha_fwd_splitkv_func_signature_source = R"(
    {% if is_execute %} {{c_flag}} FC_EXPORT {% endif %} void {{function_name}}(
        void* q_buf_ptr,
        void* k_buf_ptr,
        void* v_buf_ptr,
        void* bias_buf_ptr,
        void* lse_acc_buf_ptr,
        void* o_acc_buf_ptr,
        int64_t* seqstart_q_ptr,
        int64_t* seqstart_k_ptr,
        void* seqlen_k_ptr,
        void* block_table_buf_ptr,
        int64_t* cache_batch_idx_buf_ptr,
        int64_t batch,
        int64_t seqlen_q,
        int64_t seqlen_k,
        int64_t nhead_q,
        int64_t nhead_k,
        int64_t hdim_q,
        int64_t hdim_v,
        int64_t max_seqlen_q,
    {% if not is_execute %}
        int64_t num_splits,
    {% endif %}
        int64_t max_num_page_blocks,
        int64_t paged_block_size,
        float scale,
        std::array<int64_t,2> window_size,
        uint32_t mask_type,
        hipStream_t stream
    )
)";

static const std::string g_fmha_fwd_splitkv_func_call_source = R"(
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
{% if num_splits > 1 %}
        lse_acc_buf.GetDeviceBuffer(),
        o_acc_buf.GetDeviceBuffer(),
{% else %}
        nullptr,
        o_acc_buf.GetDeviceBuffer(),
{% endif %}
{% if mode_str == "group" %}
        seqstart_q.GetDeviceBuffer(),
        seqstart_k.GetDeviceBuffer(),
{% else %}
        nullptr,
        nullptr,
{% endif %}
        seqlen_k_buf.GetDeviceBuffer(),
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
        max_seqlen_q,
        num_splits,
        max_num_page_blocks,
        paged_block_size,
        scale,
        {window_left_size,window_right_size},
        mask_type,
        stream
    );
)";

static const std::string g_fmha_fwd_splitkv_prepare_args_source = R"(
   const auto init_args = [&](auto& args){  
 
    auto max_seqlen_k = std::numeric_limits<int32_t>::min();
    for(ck_tile::index_t wb = 0; wb < batch; ++wb){
        const int32_t real_seqlen_k = seqstart_k_ptr[wb + 1] - seqstart_k_ptr[wb];
        if(max_seqlen_k < real_seqlen_k)
            {
                max_seqlen_k = real_seqlen_k;
            }
    }
    
    const ck_tile::index_t shape_batch = {% if mode_str == "batch" %} batch; {% else %} 1; {% endif %}
    const ck_tile::index_t shape_seqlen_q = {% if mode_str == "batch" %} seqlen_q; {% else %} seqstart_q_ptr[sizeof(seqstart_q_ptr)/sizeof(seqstart_q_ptr[0]) - 1]; {% endif %}
    const ck_tile::index_t shape_seqlen_k =
{% if mode_str == "batch" %} seqlen_k; {% else %} seqstart_k_ptr[sizeof(seqstart_q_ptr)/sizeof(seqstart_q_ptr[0]) - 1]; {% endif %}
    
    args.num_splits = {% if not is_execute %} num_splits; {% else %} {{num_splits}}; {% endif %}

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

    const ck_tile::index_t stride_o_acc   = hdim_v;
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

    const ck_tile::index_t nhead_stride_o_acc   = args.num_splits * shape_seqlen_q * hdim_v;
    const ck_tile::index_t nhead_stride_o    = hdim_v;

    // setup batch_stride_* arguments
    // setup batch_stride_* arguments
    const ck_tile::index_t batch_stride_q = nhead_q * shape_seqlen_q * hdim_q;
    const ck_tile::index_t batch_stride_k = 
{% if paged_block_size > 0 %}
    nhead_k * paged_block_size * hdim_q;
{% else %}
    nhead_k * shape_seqlen_k * hdim_q;
{% endif %}
    const ck_tile::index_t batch_stride_v = 
{% if paged_block_size > 0 %}
    nhead_k * hdim_v * paged_block_size;
{% else %}
    nhead_k * hdim_v * seqlen_k;
{% endif %}

    const ck_tile::index_t batch_stride_o_acc = nhead_q * args.num_splits * shape_seqlen_q * hdim_v;

{% if bias_rank_info == 0 %}
    const ck_tile::index_t batch_stride_bias    = 0 * nhead_q * shape_seqlen_q * shape_seqlen_k;
{% elif bias_rank_info == 1 %}
    const ck_tile::index_t batch_stride_bias    = 0 * nhead_q * shape_seqlen_q * shape_seqlen_k;
{% elif bias_rank_info == 2 %}
    const ck_tile::index_t batch_stride_bias    = nhead_q * shape_seqlen_q * shape_seqlen_k;
{% endif %}

    const ck_tile::index_t batch_stride_block_table = max_num_page_blocks / batch;
    const ck_tile::index_t batch_stride_o     = nhead_q * shape_seqlen_q * hdim_v;

    // setup split_stride_* arguments (only used in split-kv kernel)
    const ck_tile::index_t split_stride_lse_acc = shape_seqlen_q;
    const ck_tile::index_t split_stride_o_acc   = shape_seqlen_q * hdim_v;

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
    args.lse_acc_ptr = lse_acc_buf_ptr;
    args.o_acc_ptr   = o_acc_buf_ptr;

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
    batch_stride_bias;
{% else %}
    0;
{% endif %}
    args.nhead_stride_o    = nhead_stride_o;
    args.batch_stride_bias = 
{% if bias_str == "alibi" %}
    0;
{% elif bias_str == "elementwise" %}
    nhead_stride_bias;
{% else %}
    0;
{% endif %}
    args.batch_stride_o    = batch_stride_o;

    args.window_size_left  = window_size[0];
    args.window_size_right = window_size[1];
    args.mask_type         = mask_type;

    args.lse_acc_ptr = lse_acc_buf_ptr;
    args.o_acc_ptr   = o_acc_buf_ptr;

    args.block_table_ptr = block_table_buf_ptr;
    args.batch_stride_block_table = batch_stride_block_table;
    args.paged_block_size          = paged_block_size;
    args.cache_batch_idx = cache_batch_idx_buf_ptr;

    args.stride_o_acc         = stride_o_acc;
    args.nhead_stride_o_acc   = nhead_stride_o_acc;
    args.batch_stride_o_acc   = batch_stride_o_acc;

    args.split_stride_lse_acc = split_stride_lse_acc;
    args.split_stride_o_acc   = split_stride_o_acc;

    args.is_gappy = false; // use 'false' for flash-attention integration
    };
)";

static const std::string g_fmha_fwd_splitkv_make_args_source = R"(
    FmhaFwdSplitKVArgs fmha_fwd_splitkv_args;
    init_args(fmha_fwd_splitkv_args);

{% if mode_str == "group" %}
    auto kargs = {{kernel_name}}::MakeKargs(fmha_fwd_splitkv_args.q_ptr,
                                            fmha_fwd_splitkv_args.k_ptr,
                                            fmha_fwd_splitkv_args.v_ptr,
                                            fmha_fwd_splitkv_args.bias_ptr,
                                            fmha_fwd_splitkv_args.lse_acc_ptr,
                                            fmha_fwd_splitkv_args.o_acc_ptr,
                                            fmha_fwd_splitkv_args.batch,
                                            fmha_fwd_splitkv_args.seqstart_q_ptr,
                                            fmha_fwd_splitkv_args.seqstart_k_ptr,
                                            fmha_fwd_splitkv_args.seqlen_k_ptr,
                                            fmha_fwd_splitkv_args.hdim_q,
                                            fmha_fwd_splitkv_args.hdim_v,
                                            fmha_fwd_splitkv_args.nhead_q,
                                            fmha_fwd_splitkv_args.nhead_q / fmha_fwd_splitkv_args.nhead_k,
                                            fmha_fwd_splitkv_args.num_splits,
                                            fmha_fwd_splitkv_args.block_table_ptr,
                                            fmha_fwd_splitkv_args.batch_stride_block_table,
                                            fmha_fwd_splitkv_args.paged_block_size,
                                            fmha_fwd_splitkv_args.is_gappy,
                                            fmha_fwd_splitkv_args.scale_s,
                                            1.0f, // scale_p
                                            fmha_fwd_splitkv_args.stride_q,
                                            fmha_fwd_splitkv_args.stride_k,
                                            fmha_fwd_splitkv_args.stride_v,
                                            fmha_fwd_splitkv_args.stride_bias,
                                            fmha_fwd_splitkv_args.stride_o_acc,
                                            fmha_fwd_splitkv_args.nhead_stride_q,
                                            fmha_fwd_splitkv_args.nhead_stride_k,
                                            fmha_fwd_splitkv_args.nhead_stride_v,
                                            fmha_fwd_splitkv_args.nhead_stride_bias,
                                            fmha_fwd_splitkv_args.nhead_stride_lse_acc,
                                            fmha_fwd_splitkv_args.nhead_stride_o_acc,
                                            fmha_fwd_splitkv_args.batch_stride_k,
                                            fmha_fwd_splitkv_args.batch_stride_v,
                                            fmha_fwd_splitkv_args.batch_stride_lse_acc,
                                            fmha_fwd_splitkv_args.batch_stride_o_acc,
                                            fmha_fwd_splitkv_args.split_stride_lse_acc,
                                            fmha_fwd_splitkv_args.split_stride_o_acc,
                                            fmha_fwd_splitkv_args.window_size_left,
                                            fmha_fwd_splitkv_args.window_size_right,
                                            fmha_fwd_splitkv_args.mask_type);
{% else %}
    auto kargs = {{kernel_name}}::MakeKargs(fmha_fwd_splitkv_args.q_ptr,
                                            fmha_fwd_splitkv_args.k_ptr,
                                            fmha_fwd_splitkv_args.v_ptr,
                                            fmha_fwd_splitkv_args.bias_ptr,
                                            fmha_fwd_splitkv_args.lse_acc_ptr,
                                            fmha_fwd_splitkv_args.o_acc_ptr,
                                            fmha_fwd_splitkv_args.batch,
                                            fmha_fwd_splitkv_args.seqlen_q,
                                            fmha_fwd_splitkv_args.seqlen_k,
                                            fmha_fwd_splitkv_args.seqlen_k_ptr, // seqlen_k_ptr
                                            fmha_fwd_splitkv_args.hdim_q,
                                            fmha_fwd_splitkv_args.hdim_v,
                                            fmha_fwd_splitkv_args.nhead_q,
                                            fmha_fwd_splitkv_args.nhead_q / fmha_fwd_splitkv_args.nhead_k,
                                            fmha_fwd_splitkv_args.num_splits,
                                            nullptr, // block_table_ptr
                                            0, // batch_stride_block_table
                                            0, // page_table_size 
                                            nullptr, // cache_batch_idx
                                            fmha_fwd_splitkv_args.scale_s,
                                            1.0f, // scale_p
                                            fmha_fwd_splitkv_args.stride_q,
                                            fmha_fwd_splitkv_args.stride_k,
                                            fmha_fwd_splitkv_args.stride_v,
                                            fmha_fwd_splitkv_args.stride_bias,
                                            fmha_fwd_splitkv_args.stride_o_acc,
                                            fmha_fwd_splitkv_args.nhead_stride_q,
                                            fmha_fwd_splitkv_args.nhead_stride_k,
                                            fmha_fwd_splitkv_args.nhead_stride_v,
                                            fmha_fwd_splitkv_args.nhead_stride_bias,
                                            fmha_fwd_splitkv_args.nhead_stride_lse_acc,
                                            fmha_fwd_splitkv_args.nhead_stride_o_acc,
                                            fmha_fwd_splitkv_args.batch_stride_q,
                                            fmha_fwd_splitkv_args.batch_stride_k,
                                            fmha_fwd_splitkv_args.batch_stride_v,
                                            fmha_fwd_splitkv_args.batch_stride_bias,
                                            fmha_fwd_splitkv_args.batch_stride_lse_acc,
                                            fmha_fwd_splitkv_args.batch_stride_o_acc,
                                            fmha_fwd_splitkv_args.split_stride_lse_acc,
                                            fmha_fwd_splitkv_args.split_stride_o_acc,
                                            fmha_fwd_splitkv_args.window_size_left,
                                            fmha_fwd_splitkv_args.window_size_right,
                                            fmha_fwd_splitkv_args.mask_type);
{% endif %}
    dim3 grids = {{kernel_name}}::GridSize(
        fmha_fwd_splitkv_args.batch, fmha_fwd_splitkv_args.nhead_q, fmha_fwd_splitkv_args.nhead_k, fmha_fwd_splitkv_args.max_seqlen_q, fmha_fwd_splitkv_args.hdim_v, fmha_fwd_splitkv_args.num_splits);
)";

static const std::string g_fmha_fwd_splitkv_tensor_decl_source = R"(
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

{% if num_splits > 1 %}
    ck_tile::HostTensor<LSEDataType> lse_acc_host(std::array<ck_tile::index_t, 4>{shape_batch, nhead_q, num_splits, shape_seqlen_q});
{% endif %}
    ck_tile::HostTensor<OaccDataType> o_acc_host(
        std::array<ck_tile::index_t, 5>{shape_batch, nhead_q, num_splits, shape_seqlen_q, hdim_v});

{% if paged_block_size > 0 %}
    ck_tile::HostTensor<int32_t> block_table_host(std::array<ck_tile::index_t, 2>{batch, max_num_page_blocks / batch});
    iota_shuffle(block_table_host.begin(), block_table_host.end(), 0);
{% endif %}

{% if use_cache_batch_idx %}
    ck_tile::HostTensor<int32_t> cache_batch_idx_host(use_cache_batch_idx, std::array<ck_tile::index_t, 1>{batch});
    iota_shuffle(cache_batch_idx_host.begin(), cache_batch_idx_host.end(), 0);
{% endif %}

)";

const static std::string g_fmha_fwd_splitkv_tensor_generate_source = R"(

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

{% if mode_str == "group" %}
    ck_tile::DeviceMem seqstart_q(seqstart_q_host.size() * sizeof(int64_t));
    seqstart_q.ToDevice(seqstart_q_host.data());
    ck_tile::DeviceMem seqstart_k(seqstart_k_host.size() * sizeof(int64_t));
    seqstart_k.ToDevice(seqstart_k_host.data());
{% endif %}

{% if mode_str == "batch" %}
    ck_tile::DeviceMem seqlen_k_buf(seqlen_ks.size() * sizeof(int64_t));
    seqlen_k_buf.ToDevice(seqlen_ks.data());
{% endif %}

{% if num_splits > 1 %}
    ck_tile::DeviceMem lse_acc_buf(lse_acc_host.get_element_space_size_in_bytes());
{% endif %}
    ck_tile::DeviceMem o_acc_buf(o_acc_host.get_element_space_size_in_bytes());

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
class FmhaFwdSplitKVKernel: public FmhaCommonKernel {
public:
    FmhaFwdSplitKVKernel()  = default;
    ~FmhaFwdSplitKVKernel() = default;

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

flashck_REGISTER_KERNEL(CK_TILE, fmha_fwd_splitkv, flashck::FmhaFwdSplitKVKernel, ALL_LAYOUT, _Float16, ushort);
