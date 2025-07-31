#pragma once

#include <string>

static const std::string g_fmha_fwd_splitkv_create_args_tpl = R"(
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
            "total_seqlen_q = q_seq_len * batch, and q_seq_len per batch may vary\n"
            "also with \"-s=s0,s1,s2...\" comma seperated int to set per batch seqlen group-mode")
        .insert("s_k", "-1", "kv_seq_len (not including new key/value), -1 means equal to s")
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

static const std::string g_fmha_fwd_splitkv_args_parser_tpl = R"(
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

    float scale = arg_parser.get_float("scale");
    if(scale == .0f)
        scale = 1.0 / ck_tile::sqrt(static_cast<float>(qk_head_dim));
    
    ck_tile::index_t num_splits = arg_parser.get_int("num_splits");

    ck_tile::index_t paged_block_size = arg_parser.get_int("paged_block_size");

    uint32_t mask_type = arg_parser.get_uint32("mask");

    ck_tile::index_t window_left_size = arg_parser.get_int("window_left_size");
    ck_tile::index_t window_right_size = arg_parser.get_int("window_right_size");

)";

static const std::string g_fmha_fwd_splitkv_args_decl_tpl = R"(
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
    bool is_gappy; // differentiate kv_seq_start_ptr usage. only used if 'block_table_ptr' is not
                   // nullptr.

    const void* cache_batch_idx;

    // the real q_seq_len & kv_seq_len are decided by following:
    // batch mode: q_seq_len = kargs.q_seq_len
    //             kv_seq_len = kargs.kv_seq_len
    // group mode: q_seq_len = kargs.q_seq_start_ptr[b + 1] - kargs.q_seq_start_ptr[b]
    //             kv_seq_len = kargs.kv_seq_start_ptr[b + 1] - kargs.kv_seq_start_ptr[b]
    // kvcache mode (use same kernel as batch mode):
    //             q_seq_len = kargs.q_seq_len
    //             kv_seq_len = kargs.kv_seq_start_ptr[b + 1] - kargs.kv_seq_start_ptr[b]
    const void* q_seq_start_ptr;
    const void* kv_seq_start_ptr;
    const void* k_seq_len_ptr;

    ck_tile::index_t batch;
    ck_tile::index_t q_seq_len;
    ck_tile::index_t kv_seq_len;
    ck_tile::index_t q_max_seq_len;
    ck_tile::index_t qk_head_dim;
    ck_tile::index_t v_head_dim;
    ck_tile::index_t q_num_heads;
    ck_tile::index_t kv_num_heads;
    ck_tile::index_t num_splits;

    float scale_s;
    float scale_p;
    float scale_o;

    ck_tile::index_t q_stride;
    ck_tile::index_t k_stride;
    ck_tile::index_t v_stride;
    ck_tile::index_t bias_stride; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t o_acc_stride;
    ck_tile::index_t o_stride;

    ck_tile::index_t q_num_heads_stride;
    ck_tile::index_t k_num_heads_stride;
    ck_tile::index_t v_num_heads_stride;
    ck_tile::index_t bias_num_heads_stride;
    ck_tile::index_t lse_num_heads_stride;
    ck_tile::index_t lse_acc_num_heads_stride;
    ck_tile::index_t o_acc_num_heads_stride;
    ck_tile::index_t o_num_heads_stride;

    ck_tile::index_t q_batch_stride;
    ck_tile::index_t k_batch_stride;
    ck_tile::index_t v_batch_stride;
    ck_tile::index_t bias_batch_stride;
    ck_tile::index_t lse_batch_stride;
    ck_tile::index_t lse_acc_batch_stride;
    ck_tile::index_t o_acc_batch_stride;
    ck_tile::index_t o_batch_stride;

    ck_tile::index_t lse_acc_split_stride;
    ck_tile::index_t o_acc_split_stride;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;
};

)";

static const std::string g_fmha_fwd_splitkv_func_signature_tpl = R"(
    void {{function_name}}(
        void* q_buf_ptr,
        void* k_buf_ptr,
        void* v_buf_ptr,
        void* bias_buf_ptr,
        void* lse_acc_buf_ptr,
        void* o_acc_buf_ptr,
        int64_t* q_seq_start_ptr,
        int64_t* kv_seq_start_ptr,
        void* k_seq_len_ptr,
        void* block_table_buf_ptr,
        int64_t* cache_batch_idx_buf_ptr,
        int64_t batch,
        int64_t q_seq_len,
        int64_t kv_seq_len,
        int64_t q_num_heads,
        int64_t kv_num_heads,
        int64_t qk_head_dim,
        int64_t v_head_dim,
        int64_t q_max_seq_len,
        int64_t num_splits,
        int64_t max_num_page_blocks,
        int64_t paged_block_size,
        float scale,
        std::array<int64_t,2> window_size,
        uint32_t mask_type,
        hipStream_t stream
    )
)";

static const std::string g_fmha_fwd_splitkv_func_call_tpl = R"(
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
{% if mode == "group" %}
        q_seq_start.GetDeviceBuffer(),
        kv_seq_start.GetDeviceBuffer(),
{% else %}
        nullptr,
        nullptr,
{% endif %}
        k_seq_len_buf.GetDeviceBuffer(),
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
        q_max_seq_len,
        ,
        max_num_page_blocks,
        paged_block_size,
        scale,
        {window_left_size,window_right_size},
        mask_type,
        stream
    );
)";

static const std::string g_fmha_fwd_splitkv_prepare_args_tpl = R"(
   const auto init_args = [&](auto& args){  
 
    auto max_seqlen_k = std::numeric_limits<int32_t>::min();
    for(ck_tile::index_t wb = 0; wb < batch; ++wb){
        const int32_t real_seqlen_k = kv_seq_start_ptr[wb + 1] - kv_seq_start_ptr[wb];
        if(max_seqlen_k < real_seqlen_k)
            {
                max_seqlen_k = real_seqlen_k;
            }
    }
    
    const ck_tile::index_t shape_batch = {% if mode == "batch" %} batch; {% else %} 1; {% endif %}
    const ck_tile::index_t q_shape_seq_len = {% if mode == "batch" %} q_seq_len; {% else %} q_seq_start_ptr[sizeof(q_seq_start_ptr)/sizeof(q_seq_start_ptr[0]) - 1]; {% endif %}
    const ck_tile::index_t kv_shape_seq_len =
{% if mode == "batch" %} kv_seq_len; {% else %} kv_seq_start_ptr[sizeof(q_seq_start_ptr)/sizeof(q_seq_start_ptr[0]) - 1]; {% endif %}
    
    args.num_splits = num_splits;

    // setup stride_* arguments
    const ck_tile::index_t q_stride = q_num_heads * qk_head_dim;
    const ck_tile::index_t k_stride = kv_num_heads * qk_head_dim;
    const ck_tile::index_t v_stride = kv_num_heads * v_head_dim;

{% if bias_rank_info == 0 %}
    const ck_tile::index_t bias_stride = kv_shape_seq_len;
{% elif bias_rank_info == 1 %}
    const ck_tile::index_t bias_stride = kv_shape_seq_len;
{% elif bias_rank_info == 2 %}
    const ck_tile::index_t bias_stride = kv_shape_seq_len;
{% endif %}

    const ck_tile::index_t o_acc_stride   = v_head_dim;
    const ck_tile::index_t o_stride       = q_num_heads * v_head_dim;

    // setup nhead_stride_* arguments
    const ck_tile::index_t q_num_heads_stride = qk_head_dim;
    const ck_tile::index_t k_num_heads_stride = qk_head_dim;
    const ck_tile::index_t v_num_heads_stride = v_head_dim;

{% if bias_rank_info == 0 %}
    const ck_tile::index_t bias_num_heads_stride = 0 * q_shape_seq_len * kv_shape_seq_len;
{% elif bias_rank_info == 1 %}
    const ck_tile::index_t bias_num_heads_stride = q_shape_seq_len * kv_shape_seq_len;
{% elif bias_rank_info == 2 %}
    const ck_tile::index_t bias_num_heads_stride = q_shape_seq_len * kv_shape_seq_len;
{% endif %}

    const ck_tile::index_t o_acc_num_heads_stride   = args.num_splits * q_shape_seq_len * v_head_dim;
    const ck_tile::index_t o_num_heads_stride    = v_head_dim;

    // setup batch_stride_* arguments
    // setup batch_stride_* arguments
    const ck_tile::index_t q_batch_stride = q_num_heads * q_shape_seq_len * qk_head_dim;
    const ck_tile::index_t k_batch_stride = 
{% if paged_block_size > 0 %}
    kv_num_heads * paged_block_size * qk_head_dim;
{% else %}
    kv_num_heads * kv_shape_seq_len * qk_head_dim;
{% endif %}
    const ck_tile::index_t v_batch_stride = 
{% if paged_block_size > 0 %}
    kv_num_heads * v_head_dim * paged_block_size;
{% else %}
    kv_num_heads * v_head_dim * kv_seq_len;
{% endif %}

    const ck_tile::index_t o_acc_batch_stride = q_num_heads * args.num_splits * q_shape_seq_len * v_head_dim;

{% if bias_rank_info == 0 %}
    const ck_tile::index_t bias_batch_stride    = 0 * q_num_heads * q_shape_seq_len * kv_shape_seq_len;
{% elif bias_rank_info == 1 %}
    const ck_tile::index_t bias_batch_stride    = 0 * q_num_heads * q_shape_seq_len * kv_shape_seq_len;
{% elif bias_rank_info == 2 %}
    const ck_tile::index_t bias_batch_stride    = q_num_heads * q_shape_seq_len * kv_shape_seq_len;
{% endif %}

    const ck_tile::index_t batch_stride_block_table = max_num_page_blocks / batch;
    const ck_tile::index_t o_batch_stride     = q_num_heads * q_shape_seq_len * v_head_dim;

    // setup split_stride_* arguments (only used in split-kv kernel)
    const ck_tile::index_t lse_acc_split_stride = q_shape_seq_len;
    const ck_tile::index_t o_acc_split_stride   = q_shape_seq_len * v_head_dim;

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

    args.bias_ptr = bias_buf_ptr;
    args.lse_acc_ptr = lse_acc_buf_ptr;
    args.o_acc_ptr   = o_acc_buf_ptr;

    args.q_seq_start_ptr = q_seq_start_ptr;
    args.kv_seq_start_ptr = kv_seq_start_ptr;
    args.k_seq_len_ptr = k_seq_len_ptr;
    
    args.kv_seq_len     = kv_seq_len; // unused in group mode (or kvcache enabled)
    args.q_max_seq_len = q_max_seq_len;

    args.scale_s = scale;

    args.bias_stride = 
{% if bias_str == "alibi" %}
{% if bias_rank_info == 0 %} 0 * q_num_heads; {% else %} q_num_heads; {% endif %}
{% elif bias_str == "elementwise" %}
    bias_stride;
{% else %}
    0;
{% endif %}

    args.o_stride          = o_stride;
    args.bias_num_heads_stride = 
{% if bias_str == "alibi" %}
    0;
{% elif bias_str == "elementwise" %}
    bias_batch_stride;
{% else %}
    0;
{% endif %}
    args.o_num_heads_stride    = o_num_heads_stride;
    args.bias_batch_stride = 
{% if bias_str == "alibi" %}
    0;
{% elif bias_str == "elementwise" %}
    bias_num_heads_stride;
{% else %}
    0;
{% endif %}
    args.o_batch_stride    = o_batch_stride;

    args.window_size_left  = window_size[0];
    args.window_size_right = window_size[1];
    args.mask_type         = mask_type;

    args.lse_acc_ptr = lse_acc_buf_ptr;
    args.o_acc_ptr   = o_acc_buf_ptr;

    args.block_table_ptr = block_table_buf_ptr;
    args.batch_stride_block_table = batch_stride_block_table;
    args.paged_block_size          = paged_block_size;
    args.cache_batch_idx = cache_batch_idx_buf_ptr;

    args.o_acc_stride         = o_acc_stride;
    args.o_acc_num_heads_stride   = o_acc_num_heads_stride;
    args.o_acc_batch_stride   = o_acc_batch_stride;

    args.lse_acc_split_stride = lse_acc_split_stride;
    args.o_acc_split_stride   = o_acc_split_stride;

    args.is_gappy = false; // use 'false' for flash-attention integration
    };
)";

static const std::string g_fmha_fwd_splitkv_make_args_tpl = R"(
    FmhaFwdSplitKVArgs fmha_fwd_splitkv_args;
    init_args(fmha_fwd_splitkv_args);

{% if mode == "group" %}
    auto kargs = {{kernel_name}}::MakeKargs(fmha_fwd_splitkv_args.q_ptr,
                                            fmha_fwd_splitkv_args.k_ptr,
                                            fmha_fwd_splitkv_args.v_ptr,
                                            fmha_fwd_splitkv_args.bias_ptr,
                                            fmha_fwd_splitkv_args.lse_acc_ptr,
                                            fmha_fwd_splitkv_args.o_acc_ptr,
                                            fmha_fwd_splitkv_args.batch,
                                            fmha_fwd_splitkv_args.q_seq_start_ptr,
                                            fmha_fwd_splitkv_args.kv_seq_start_ptr,
                                            fmha_fwd_splitkv_args.k_seq_len_ptr,
                                            fmha_fwd_splitkv_args.qk_head_dim,
                                            fmha_fwd_splitkv_args.v_head_dim,
                                            fmha_fwd_splitkv_args.q_num_heads,
                                            fmha_fwd_splitkv_args.q_num_heads / fmha_fwd_splitkv_args.kv_num_heads,
                                            fmha_fwd_splitkv_args.num_splits,
                                            fmha_fwd_splitkv_args.block_table_ptr,
                                            fmha_fwd_splitkv_args.batch_stride_block_table,
                                            fmha_fwd_splitkv_args.paged_block_size,
                                            fmha_fwd_splitkv_args.is_gappy,
                                            fmha_fwd_splitkv_args.scale_s,
                                            1.0f, // scale_p
                                            fmha_fwd_splitkv_args.q_stride,
                                            fmha_fwd_splitkv_args.k_stride,
                                            fmha_fwd_splitkv_args.v_stride,
                                            fmha_fwd_splitkv_args.bias_stride,
                                            fmha_fwd_splitkv_args.o_acc_stride,
                                            fmha_fwd_splitkv_args.q_num_heads_stride,
                                            fmha_fwd_splitkv_args.k_num_heads_stride,
                                            fmha_fwd_splitkv_args.v_num_heads_stride,
                                            fmha_fwd_splitkv_args.bias_num_heads_stride,
                                            fmha_fwd_splitkv_args.lse_acc_num_heads_stride,
                                            fmha_fwd_splitkv_args.o_acc_num_heads_stride,
                                            fmha_fwd_splitkv_args.k_batch_stride,
                                            fmha_fwd_splitkv_args.v_batch_stride,
                                            fmha_fwd_splitkv_args.lse_acc_batch_stride,
                                            fmha_fwd_splitkv_args.o_acc_batch_stride,
                                            fmha_fwd_splitkv_args.lse_acc_split_stride,
                                            fmha_fwd_splitkv_args.o_acc_split_stride,
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
                                            fmha_fwd_splitkv_args.q_seq_len,
                                            fmha_fwd_splitkv_args.kv_seq_len,
                                            fmha_fwd_splitkv_args.k_seq_len_ptr, // k_seq_len_ptr
                                            fmha_fwd_splitkv_args.qk_head_dim,
                                            fmha_fwd_splitkv_args.v_head_dim,
                                            fmha_fwd_splitkv_args.q_num_heads,
                                            fmha_fwd_splitkv_args.q_num_heads / fmha_fwd_splitkv_args.kv_num_heads,
                                            fmha_fwd_splitkv_args.num_splits,
                                            nullptr, // block_table_ptr
                                            0, // batch_stride_block_table
                                            0, // page_table_size 
                                            nullptr, // cache_batch_idx
                                            fmha_fwd_splitkv_args.scale_s,
                                            1.0f, // scale_p
                                            fmha_fwd_splitkv_args.q_stride,
                                            fmha_fwd_splitkv_args.k_stride,
                                            fmha_fwd_splitkv_args.v_stride,
                                            fmha_fwd_splitkv_args.bias_stride,
                                            fmha_fwd_splitkv_args.o_acc_stride,
                                            fmha_fwd_splitkv_args.q_num_heads_stride,
                                            fmha_fwd_splitkv_args.k_num_heads_stride,
                                            fmha_fwd_splitkv_args.v_num_heads_stride,
                                            fmha_fwd_splitkv_args.bias_num_heads_stride,
                                            fmha_fwd_splitkv_args.lse_acc_num_heads_stride,
                                            fmha_fwd_splitkv_args.o_acc_num_heads_stride,
                                            fmha_fwd_splitkv_args.q_batch_stride,
                                            fmha_fwd_splitkv_args.k_batch_stride,
                                            fmha_fwd_splitkv_args.v_batch_stride,
                                            fmha_fwd_splitkv_args.bias_batch_stride,
                                            fmha_fwd_splitkv_args.lse_acc_batch_stride,
                                            fmha_fwd_splitkv_args.o_acc_batch_stride,
                                            fmha_fwd_splitkv_args.lse_acc_split_stride,
                                            fmha_fwd_splitkv_args.o_acc_split_stride,
                                            fmha_fwd_splitkv_args.window_size_left,
                                            fmha_fwd_splitkv_args.window_size_right,
                                            fmha_fwd_splitkv_args.mask_type);
{% endif %}
    dim3 grids = {{kernel_name}}::GridSize(
        fmha_fwd_splitkv_args.batch, fmha_fwd_splitkv_args.q_num_heads, fmha_fwd_splitkv_args.kv_num_heads, fmha_fwd_splitkv_args.q_max_seq_len, fmha_fwd_splitkv_args.v_head_dim, fmha_fwd_splitkv_args.num_splits);
)";

static const std::string g_fmha_fwd_splitkv_tensor_decl_tpl = R"(
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

{% if bias_str == "elementwise" %}
    ck_tile::HostTensor<BiasDataType> bias_host({1, 1, q_shape_seq_len, kv_shape_seq_len});
{% else %}
    ck_tile::HostTensor<BiasDataType> bias_host({1, 1, 1, 1}); // dummy shape for simplifying code
{% endif %}

{% if bias_str == "alibi" %}
{% if bias_rank_info == 0 %}
    // alibi in 1*h
    ck_tile::HostTensor<SaccDataType> alibi_slope_host({1, q_num_heads});
{% else %}
    // alibi in b*h
    ck_tile::HostTensor<SaccDataType> alibi_slope_host({batch, q_num_heads});
{% endif %}
{% else %}
    // alibi in 1*1
    ck_tile::HostTensor<SaccDataType> alibi_slope_host({1, 1}); // dummy shape for simplifying code
{% endif %}

{% if num_splits > 1 %}
    ck_tile::HostTensor<LSEDataType> lse_acc_host(std::array<ck_tile::index_t, 4>{shape_batch, q_num_heads, num_splits, q_shape_seq_len});
{% endif %}
    ck_tile::HostTensor<OaccDataType> o_acc_host(
        std::array<ck_tile::index_t, 5>{shape_batch, q_num_heads, num_splits, q_shape_seq_len, v_head_dim});

{% if paged_block_size > 0 %}
    ck_tile::HostTensor<int32_t> block_table_host(std::array<ck_tile::index_t, 2>{batch, max_num_page_blocks / batch});
    iota_shuffle(block_table_host.begin(), block_table_host.end(), 0);
{% endif %}

{% if use_cache_batch_idx %}
    ck_tile::HostTensor<int32_t> cache_batch_idx_host(use_cache_batch_idx, std::array<ck_tile::index_t, 1>{batch});
    iota_shuffle(cache_batch_idx_host.begin(), cache_batch_idx_host.end(), 0);
{% endif %}

)";

const static std::string g_fmha_fwd_splitkv_tensor_generate_tpl = R"(

{% if init_method == "uri" %}
    ck_tile::FillUniformDistributionIntegerValue<QDataType>{-3.f, 3.f, {{seed}}}(q_host);
    ck_tile::FillUniformDistributionIntegerValue<KDataType>{-3.f, 3.f, {{seed}}}(k_host);
    ck_tile::FillUniformDistributionIntegerValue<VDataType>{-3.f, 3.f, {{seed}}}(v_host);
{% if bias_str == "elementwise" %}
    ck_tile::FillUniformDistributionIntegerValue<BiasDataType>{-3.f, 3.f, {{seed}}}(bias_host);
{% endif %}

{% elif init_method == "nri" %}
    ck_tile::FillNormalDistributionIntegerValue<QDataType>{-3.f, 3.f, {{seed}}}(q_host);
    ck_tile::FillNormalDistributionIntegerValue<KDataType>{-3.f, 3.f, {{seed}}}(k_host);
    ck_tile::FillNormalDistributionIntegerValue<VDataType>{-3.f, 3.f, {{seed}}}(v_host);
{% if bias_str == "elementwise" %}
    ck_tile::FillNormalDistributionIntegerValue<BiasDataType>{-3.f, 3.f, {{seed}}}(bias_host);
{% endif %}

{% elif init_method == "uf" %}
    ck_tile::FillUniformDistribution<QDataType>{0.f, 1.f, {{seed}}}(q_host);
    ck_tile::FillUniformDistribution<KDataType>{0.f, 1.f, {{seed}}}(k_host);
    ck_tile::FillUniformDistribution<VDataType>{0.f, 1.f, {{seed}}}(v_host);
{% if bias_str == "elementwise" %}
    ck_tile::FillUniformDistribution<BiasDataType>{0.f, 1.f, {{seed}}}(bias_host);
{% endif %}

{% elif init_method == "nf" %}
    ck_tile::FillNormalDistribution<QDataType>{0.f, 3.f, {{seed}}}(q_host);
    ck_tile::FillNormalDistribution<KDataType>{0.f, 3.f, {{seed}}}(k_host);
    ck_tile::FillNormalDistribution<VDataType>{0.f, 3.f, {{seed}}}(v_host);
{% if bias_str == "elementwise" %}
    ck_tile::FillNormalDistribution<BiasDataType>{0.f, 3.f, {{seed}}}(bias_host);
{% endif %}

{% elif init_method == "tf" %}
    ck_tile::FillTrigValue<QDataType>{}(q_host);
    ck_tile::FillTrigValue<KDataType>{}(k_host);
    ck_tile::FillTrigValue<VDataType>{}(v_host);
{% if bias_str == "elementwise" %}
    ck_tile::FillTrigValue<BiasDataType>{}(bias_host);
{% endif %}
    
{% elif init_method == "uf8q" %}
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
    auto slopes = ck_tile::get_alibi_slopes<SaccDataType>(q_num_heads);
    assert(slopes.size() == static_cast<std::size_t>(q_num_heads));
{% if bias_rank_info == 0 %}
    // alibi in 1*h
    std::copy(slopes.begin(), slopes.end(), alibi_slope_host.begin());
{% else %}
    // alibi in b*h
    for(auto i_b = 0; i_b < batch; i_b++)
    {
        std::copy(slopes.begin(), slopes.end(), alibi_slope_host.begin() + i_b * q_num_heads);
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

{% if mode == "group" %}
    ck_tile::DeviceMem q_seq_start(q_seq_start_host.size() * sizeof(int64_t));
    q_seq_start.ToDevice(q_seq_start_host.data());
    ck_tile::DeviceMem kv_seq_start(kv_seq_start_host.size() * sizeof(int64_t));
    kv_seq_start.ToDevice(kv_seq_start_host.data());
{% endif %}

{% if mode == "batch" %}
    ck_tile::DeviceMem k_seq_len_buf(kv_seq_len.size() * sizeof(int64_t));
    k_seq_len_buf.ToDevice(kv_seq_len.data());
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
