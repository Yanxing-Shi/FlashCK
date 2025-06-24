#pragma once

#include "flashck/core/module/kernels/fmha_kernels/fmha_common_kernel.h"

#include "flashck/core/module/kernels/kernel_registry.h"

static const std::string g_fmha_fwd_splitkv_combine_create_args_source = R"(
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("b", "2", "batch size")
        .insert("h", "8", "num of head, for q")
        .insert(
            "s",
            "3328",
            "seqlen_q. if group-mode, means the average value of seqlen_q\n"
            "total_seqlen_q = seqlen_q * batch, and seqlen_q per batch may vary\n"
            "also with \"-s=s0,s1,s2...\" comma seperated int to set per batch seqlen group-mode")
        .insert("d_v", "-1", "head dim for v, -1 means equal to d")
        .insert("num_splits",
                "1",
                "# of splits for key/value. 0 to determine actual number by heuristic");
                
    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
)";

static const std::string g_fmha_fwd_splitkv_combine_args_parser_source = R"(
    ck_tile::index_t batch   = arg_parser.get_int("b");
    ck_tile::index_t seqlen_q = arg_parser.get_int("s");
    ck_tile::index_t nhead_q = arg_parser.get_int("h");
    ck_tile::index_t hdim_v = arg_parser.get_int("d_v");

    ck_tile::index_t num_splits = arg_parser.get_int("num_splits");

)";

static const std::string g_fmha_fwd_splitkv_combine_args_decl_source = R"(
struct FmhaFwdSplitKVCombineArgs
{
    void* lse_acc_ptr;
    void* o_acc_ptr;
    void* o_ptr;

    // the real seqlen_q & seqlen_k are decided by following:
    // batch mode: seqlen_q = kargs.seqlen_q
    //             seqlen_k = kargs.seqlen_k
    // group mode: seqlen_q = kargs.seqstart_q_ptr[b + 1] - kargs.seqstart_q_ptr[b]
    //             seqlen_k = kargs.seqstart_k_ptr[b + 1] - kargs.seqstart_k_ptr[b]
    // kvcache mode (use same kernel as batch mode):
    //             seqlen_q = kargs.seqlen_q
    //             seqlen_k = kargs.seqstart_k_ptr[b + 1] - kargs.seqstart_k_ptr[b]
    const void* seqstart_q_ptr;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t num_splits;

    float scale_o;

    ck_tile::index_t stride_o_acc;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_lse_acc;
    ck_tile::index_t nhead_stride_o_acc;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_lse_acc;
    ck_tile::index_t batch_stride_o_acc;
    ck_tile::index_t batch_stride_o;
    ck_tile::index_t split_stride_lse_acc;
    ck_tile::index_t split_stride_o_acc;

};

)";

static const std::string g_fmha_fwd_splitkv_combine_func_signature_source = R"(
    {% if is_execute %} {{c_flag}} FC_EXPORT {% endif %} void {{function_name}}(
        void* lse_acc_buf_ptr,
        void* o_acc_buf_ptr,
        void* o_buf_ptr,
        int64_t* seqstart_q_ptr,
        int64_t batch,
        int64_t seqlen_q,
        int64_t nhead_q,
        int64_t hdim_v,
        int64_t max_seqlen_q,
        int num_splits,
        hipStream_t stream
    )
)";

static const std::string g_fmha_fwd_splitkv_combine_func_call_source = R"(
    {{function_name}}(
        lse_acc_buf.GetDeviceBuffer(),
        o_acc_buf.GetDeviceBuffer(),
        o_buf.GetDeviceBuffer(),
{% if mode_str == "group" %}
        seqstart_q.GetDeviceBuffer(),
{% else %}
        nullptr,
{% endif %}       
        batch,
        shape_seqlen_q,
        nhead_q,
        hdim_v,
        max_seqlen_q,
        num_splits,
        stream
    );
)";

static const std::string g_fmha_fwd_splitkv_combine_prepare_args_source = R"(

    const auto init_args = [&](auto& args){  
    /// NOTE: we broadcast bias from [1, 1, seqlen_q, seqlen_k] to [batch, nhead, seqlen_q,
    ///       seqlen_k] in this example, hence both the 'batch_stride_bias' &
    ///       'nhead_stride_bias' are 0.

    const ck_tile::index_t shape_batch = {% if mode_str == "batch" %} batch; {% else %} 1; {% endif %}
    const ck_tile::index_t shape_seqlen_q = {% if mode_str == "batch" %} seqlen_q; {% else %} seqstart_q_ptr[sizeof(seqstart_q_ptr)/sizeof(seqstart_q_ptr[0]) - 1]; {% endif %}

    // setup stride_* arguments
    const ck_tile::index_t stride_o_acc   = hdim_v;
    const ck_tile::index_t stride_o       = nhead_q * hdim_v;

    // setup nhead_stride_* arguments
    const ck_tile::index_t nhead_stride_lse_acc = (num_splits * shape_seqlen_q);
    const ck_tile::index_t nhead_stride_o_acc   = num_splits * shape_seqlen_q * hdim_v;
    const ck_tile::index_t nhead_stride_o       = hdim_v;

    // setup batch_stride_* arguments
    const ck_tile::index_t batch_stride_lse_acc = (nhead_q * num_splits * shape_seqlen_q);
    const ck_tile::index_t batch_stride_o_acc = nhead_q * num_splits * shape_seqlen_q * hdim_v;
    const ck_tile::index_t batch_stride_o     = nhead_q * shape_seqlen_q * hdim_v;
    
    // setup split_stride_* arguments (only used in split-kv kernel)
    const ck_tile::index_t split_stride_lse_acc = shape_seqlen_q;
    const ck_tile::index_t split_stride_o_acc   = shape_seqlen_q * hdim_v;

    args.lse_acc_ptr = lse_acc_buf_ptr;
    args.o_acc_ptr = o_acc_buf_ptr;
    args.o_ptr = o_buf_ptr;

    args.batch    = batch;
    args.seqlen_q = shape_seqlen_q; // unused in group mode
    args.hdim_v   = hdim_v;
    args.nhead_q  = nhead_q;

    args.seqstart_q_ptr = seqstart_q_ptr;

    args.max_seqlen_q = max_seqlen_q;

    args.stride_o          = stride_o;
    args.nhead_stride_o    = nhead_stride_o;

    args.stride_o_acc         = stride_o_acc;
    args.nhead_stride_lse_acc = nhead_stride_lse_acc;
    args.nhead_stride_o_acc   = nhead_stride_o_acc;
    args.batch_stride_lse_acc = batch_stride_lse_acc;
    args.batch_stride_o_acc   = batch_stride_o_acc;
    args.split_stride_lse_acc = split_stride_lse_acc;
    args.split_stride_o_acc   = split_stride_o_acc;

    };
)";

static const std::string g_fmha_fwd_splitkv_combine_make_args_source = R"(
    FmhaFwdSplitKVCombineArgs fmha_fwd_splitkv_combine_args;
    init_args(fmha_fwd_splitkv_combine_args);
    
{% if mode_str == "group" %}
    auto kargs = {{kernel_name}}::MakeKargs(fmha_fwd_splitkv_combine_args.lse_acc_ptr,
                                            fmha_fwd_splitkv_combine_args.o_acc_ptr,
                                            nullptr, // lse_ptr
                                            fmha_fwd_splitkv_combine_args.o_ptr,
                                            fmha_fwd_splitkv_combine_args.batch,
                                            fmha_fwd_splitkv_combine_args.seqstart_q_ptr,
                                            fmha_fwd_splitkv_combine_args.hdim_v,
                                            fmha_fwd_splitkv_combine_args.num_splits,
                                            fmha_fwd_splitkv_combine_args.scale_o,
                                            fmha_fwd_splitkv_combine_args.stride_o_acc,
                                            fmha_fwd_splitkv_combine_args.stride_o,
                                            fmha_fwd_splitkv_combine_args.nhead_stride_lse_acc,
                                            fmha_fwd_splitkv_combine_args.nhead_stride_o_acc,
                                            0, // nhead_stride_lse
                                            fmha_fwd_splitkv_combine_args.nhead_stride_o,
                                            fmha_fwd_splitkv_combine_args.split_stride_lse_acc,
                                            fmha_fwd_splitkv_combine_args.split_stride_o_acc
                                            );
{% else %}
    auto kargs = {{kernel_name}}::MakeKargs(fmha_fwd_splitkv_combine_args.lse_acc_ptr,
                                            fmha_fwd_splitkv_combine_args.o_acc_ptr,
                                            nullptr, // lse_ptr
                                            fmha_fwd_splitkv_combine_args.o_ptr,
                                            fmha_fwd_splitkv_combine_args.batch,
                                            fmha_fwd_splitkv_combine_args.seqlen_q,
                                            fmha_fwd_splitkv_combine_args.hdim_v,
                                            fmha_fwd_splitkv_combine_args.num_splits,
                                            fmha_fwd_splitkv_combine_args.scale_o,
                                            fmha_fwd_splitkv_combine_args.stride_o_acc,
                                            fmha_fwd_splitkv_combine_args.stride_o,
                                            fmha_fwd_splitkv_combine_args.nhead_stride_lse_acc,
                                            fmha_fwd_splitkv_combine_args.nhead_stride_o_acc,
                                            0, // head_stride_lse
                                            fmha_fwd_splitkv_combine_args.nhead_stride_o,
                                            fmha_fwd_splitkv_combine_args.batch_stride_lse_acc,
                                            fmha_fwd_splitkv_combine_args.batch_stride_o_acc,
                                            0, // batch_stride_lse
                                            fmha_fwd_splitkv_combine_args.batch_stride_o,
                                            fmha_fwd_splitkv_combine_args.split_stride_lse_acc,
                                            fmha_fwd_splitkv_combine_args.split_stride_o_acc);
{% endif %}
    dim3 grids = {{kernel_name}}::GridSize(fmha_fwd_splitkv_combine_args.batch, fmha_fwd_splitkv_combine_args.nhead_q, fmha_fwd_splitkv_combine_args.max_seqlen_q, fmha_fwd_splitkv_combine_args.hdim_v);
)";

static const std::string g_fmha_fwd_splitkv_combine_tensor_decl_source = R"(
{% if num_splits > 1 %}
    ck_tile::HostTensor<LSEDataType> lse_acc_host(std::array<ck_tile::index_t, 4>{shape_batch, nhead_q, num_splits, shape_seqlen_q});
{% endif %}
     ck_tile::HostTensor<OaccDataType> o_acc_host(
        std::array<ck_tile::index_t, 5>{shape_batch, nhead_q, num_splits, shape_seqlen_q, hdim_v});
    ck_tile::HostTensor<ODataType> o_host(
        {shape_batch, shape_seqlen_q, nhead_q, hdim_v});
)";

const static std::string g_fmha_fwd_splitkv_combine_tensor_generate_source = R"(
{% if init_method_str == "uri" %}
{% if num_splits > 1 %}
    ck_tile::FillUniformDistributionIntegerValue<LSEDataType>{-3.f, 3.f, {{seed}}}(lse_acc_host);
{% endif %}
    ck_tile::FillUniformDistributionIntegerValue<OaccDataType>{-3.f, 3.f, {{seed}}}(o_acc_host);

{% elif init_method_str == "nri" %}
{% if num_splits > 1 %}
    ck_tile::FillNormalDistributionIntegerValue<LSEDataType>{-3.f, 3.f, {{seed}}}(lse_acc_host);
{% endif %}
    ck_tile::FillUniformDistributionIntegerValue<OaccDataType>{-3.f, 3.f, {{seed}}}(o_acc_host);

{% elif init_method_str == "uf" %}
{% if num_splits > 1 %}
    ck_tile::FillUniformDistribution<LSEDataType>{0.f, 1.f, {{seed}}}(lse_acc_host);
{% endif %}
    ck_tile::FillUniformDistribution<OaccDataType>{0.f, 1.f, {{seed}}}(o_acc_host);

{% elif init_method_str == "nf" %}
{% if num_splits > 1 %}
    ck_tile::FillNormalDistribution<LSEDataType>{0.f, 3.f, {{seed}}}(lse_acc_host);
{% endif %}
    ck_tile::FillNormalDistribution<OaccDataType>{0.f, 1.f, {{seed}}}(o_acc_host);

{% elif init_method_str == "tf" %}
{% if num_splits > 1 %}
    ck_tile::FillTrigValue<LSEDataType>{}(lse_acc_host);
{% endif %}
    ck_tile::FillTrigValue<OaccDataType>{}(o_acc_host);

{% elif init_method_str == "uf8q" %}
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

namespace flashck {
class FmhaFwdSplitKVCombineKernel: public FmhaCommonKernel {
public:
    FmhaFwdSplitKVCombineKernel()  = default;
    ~FmhaFwdSplitKVCombineKernel() = default;

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

flashck_REGISTER_KERNEL(
    CK_TILE, fmha_fwd_splitkv_combine, flashck::FmhaFwdSplitKVCombineKernel, ALL_LAYOUT, _Float16, ushort);
