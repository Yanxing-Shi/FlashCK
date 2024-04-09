#pragma once

#include <any>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <tuple>
#include <vector>

#include "ater/core/utils/jinjia2_utils.h"

#include "ater/core/module/kernels/kernel.h"
#include "ater/core/profiler/base.h"
#include "ater/core/profiler/library.h"

namespace ater {

static const std::string g_exec_cond_source = R"(
{{indent}}if ({{cond}}) {
{{indent}}  {{program}}
{{indent}}}
)";

static const std::string g_extra_shape_source = R"(
{{indent}}ck::index_t stride_a = *a_dim1;
{{indent}}ck::index_t stride_b = *b_dim1;
{{indent}}ck::index_t stride_c = *c_dim1;
)";

static const std::string g_instance_source = R"(
{{config}}
using {{name}} = {{config_name}};
)";

static const std::string g_exec_source = R"(
{{indent}}auto op =  {{instance}}{};
{{indent}}auto invoker  = op.MakeInvoker();
{{indent}}auto argument = op.MakeArgument(
{{problem_args}}
{{indent}});
{{indent}}if(!op.IsSupportedArgument(argument)) {
{{indent}}  LOG(ERROR) << "wrong! " << op.GetTypeString() << " with the specified compilation parameters does not support this Gemm problem.";
{{indent}}}
{% if is_profiler %}
{{indent}}auto workspace_size = op.GetWorkSpaceSize(&argument);
{{indent}}GLOBAL_WORKSPACE_SIZE = workspace_size;
{% endif %}
{{indent}}invoker.Run(argument, StreamConfig{stream, false});
{{indent}}return;
)";

static const std::string g_extra_header_source = R"(
{% if gemm_flag == "" %}
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"
{% elif gemm_flag == "permute_m2n3" %}
#include "ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_xdl_cshuffle.hpp"
{% elif "bias" in gemm_flag or has_d0 %}
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp"
    {% if gemm_flag == "bias_permute" %}
#include "ck/tensor_operation/gpu/device/impl/device_gemm_bias_e_permute_xdl.hpp"
#include "ck/tensor_operation/gpu/device/impl/gemm_specialization.hpp"
    {% elif gemm_flag in ["bias_permute_m2n3", "bias_permute_m3n2"]  %}
#include "ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_xdl_cshuffle.hpp"
    {% endif %}
{% endif %}
)";

static const std::string g_problem_args_source = R"(
{{indent}}                                static_cast<ck::half_t *>(in_ptr),
{{indent}}                                static_cast<ck::half_t *>(weight_ptr),
{{indent}}                                std::array<const void*, 0>{},
{% if gemm_flag == "bias_permute" %}
{{indent}}                                static_cast<ck::half_t *>(bias_ptr),
{% elif gemm_flag == "permute" %}
{{indent}}                                nullptr,
{% elif gemm_flag == "bias_permute_m2n3" %}
{{indent}}                                std::array<const void*, 1>{static_cast<ck::half_t *>(bias_ptr)},
{% elif gemm_flag == "permute_m2n3" %}
{{indent}}                                {},
{% else %}
{% if "bias" in gemm_flag and not has_d0 %}
{{indent}}                                std::array<const void*, 1>{static_cast<ck::half_t *>(bias_ptr)},
{% elif has_d0 and not has_d1 %}
{{indent}}                                std::array<const void*, 2>{static_cast<ck::half_t *>(bias_ptr),
                                                                    static_cast<ck::half_t *>(d0_ptr)},
{% elif has_d1 %}
{{indent}}                                std::array<const void*, 3>{static_cast<ck::half_t *>(bias_ptr),
                                                                    static_cast<ck::half_t *>(d0_ptr),
                                                                    static_cast<ck::half_t *>(d1_ptr)},
{% endif %}
{% endif %}
{{indent}}                                static_cast<ck::half_t *>(out_ptr),
{% if gemm_flag in ["permute_m2n3", "bias_permute_m2n3", "bias_permute_m3n2"] %}
{% else %}
{{indent}}                                M,
{{indent}}                                N,
{{indent}}                                K,
{{indent}}                                stride_a,
{{indent}}                                stride_b,
{{indent}}                                std::array<ck::index_t, 0>{},
{% endif %}
{% if gemm_flag == "bias_permute" %}
{{indent}}                                {M0, M1, M2, N0, N1, stride_D_M0, stride_D_M1, stride_D_M2, stride_D_N0, stride_D_N1},
{{indent}}                                {M0, M1, M2, N0, N1, stride_E_M0, stride_E_M1, stride_E_M2, stride_E_N0, stride_E_N1},
{% elif gemm_flag == "permute" %}
{{indent}}                                {},
{{indent}}                                {M0, M1, M2, N0, N1, stride_E_M0, stride_E_M1, stride_E_M2, stride_E_N0, stride_E_N1},
{% elif gemm_flag in ["permute_m2n3", "bias_permute_m2n3", "bias_permute_m3n2"]  %}
{{indent}}                                a_ms_ks_lengths,
{{indent}}                                a_ms_ks_strides,
{{indent}}                                b_ns_ks_lengths,
{{indent}}                                b_ns_ks_strides,
    {% if gemm_flag == "permute_m2n3"  %}
{{indent}}                                {},
{{indent}}                                {},
    {% else %}
{{indent}}                                std::array<std::vector<ck::index_t>, 1>{d_ms_ns_lengths},
{{indent}}                                std::array<std::vector<ck::index_t>, 1>{d_ms_ns_strides},
    {% endif %}
{{indent}}                                e_ms_ns_lengths,
{{indent}}                                e_ms_ns_strides,
{% else %}
{% if "bias" in gemm_flag and not has_d0 %}
{{indent}}                                std::array<ck::index_t, 1>{0},
{% elif has_d0 and not has_d1 %}
{{indent}}                                std::array<ck::index_t, 2>{0, static_cast<int>(stride_c)},
{% elif has_d1 %}
{{indent}}                                std::array<ck::index_t, 3>{0, static_cast<int>(stride_c), static_cast<int>(stride_c)},
{% endif %}
{{indent}}                                stride_c,
{% endif %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{},
{% if gemm_flag == "" %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{}
{% elif gemm_flag in ["permute", "permute_m2n3"] %}
{{indent}}                                ck::tensor_operation::element_wise::PassThrough{}
{% elif gemm_flag == "bias" or "bias_permute" in gemm_flag %}
{{indent}}                                ck::tensor_operation::element_wise::Add{}
{% elif gemm_flag == "bias_relu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddRelu{}
{% elif gemm_flag == "bias_fast_gelu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddFastGelu{}
{% elif gemm_flag == "bias_swish" %}
{{indent}}                                ck::tensor_operation::element_wise::AddSwish{}
{% elif gemm_flag == "bias_hardswish" %}
{{indent}}                                ck::tensor_operation::element_wise::AddHardswish{}
{% elif gemm_flag == "bias_tanh" %}
{{indent}}                                ck::tensor_operation::element_wise::AddTanh{}
{% elif gemm_flag == "bias_sigmoid" %}
{{indent}}                                ck::tensor_operation::element_wise::AddSigmoid{}
{% elif gemm_flag == "bias_add" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAdd{}
{% elif gemm_flag == "bias_mul" %}
{{indent}}                                ck::tensor_operation::element_wise::AddMul{}
{% elif gemm_flag == "bias_mul_tanh" %}
{{indent}}                                ck::tensor_operation::element_wise::AddMulTanh{}
{% elif gemm_flag == "bias_add_relu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAddRelu{}
{% elif gemm_flag == "bias_add_fast_gelu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAddFastGelu{}
{% elif gemm_flag == "bias_sigmoid_mul" %}
{{indent}}                                ck::tensor_operation::element_wise::AddSigmoidMul{}
{% elif gemm_flag == "bias_sigmoid_mul_tanh" %}
{{indent}}                                ck::tensor_operation::element_wise::AddSigmoidMulTanh{}
{% elif gemm_flag == "bias_mul_add" %}
{{indent}}                                ck::tensor_operation::element_wise::AddMulAdd{}
{% elif gemm_flag == "bias_add_add" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAddAdd{}
{% elif gemm_flag == "bias_add_add_relu" %}
{{indent}}                                ck::tensor_operation::element_wise::AddAddAddRelu{}
{% endif %}
)";

static const std::string g_src_source = R"(
#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
// #include <half.hpp>
#include <random>
#include <rocrand/rocrand.h>

// #include "logging.h"
#include "glog/logging.h"

#include "library/include/ck/library/utility/device_memory.hpp"
#include "library/include/ck/library/utility/host_tensor.hpp"
#include "library/include/ck/library/utility/host_tensor_generator.hpp"
#include "include/ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "include/ck/tensor_operation/gpu/element/element_wise_operation.hpp"

{{extra_header}}


{{extra_code}}

{{instances}}


void {{function_name}}(
    void * in_ptr,
    void * weight_ptr,
    void * out_ptr,
{% if "bias" in gemm_flag or gemm_flag == "add" %}
    void * bias_ptr,
{% endif %}
{% if has_d0 %}
    void * d0_ptr,
{% endif %}
{% if has_d1 %}
    void * d1_ptr,
{% endif %}
{% for idx in range(ndims) %}
    int64_t* a_dim{{idx}},
{% endfor %}
{% for idx in range(ndims) %}
    int64_t* b_dim{{idx}},
{% endfor %}
{% for idx in range(ndims) %}
    int64_t* c_dim{{idx}},
{% endfor %}
{% for idx in range(pdims) %}
    const int p_dim{{idx}},
{% endfor %}
    hipStream_t stream
    ) {
    {{shape_func}}
    {{extra_shape}}
    {{input_addr_calculator}}
    {{output_addr_calculator}}
    {{exec_paths}}

    LOG(ERROR) << "Unsupported workload for this gemm specialization.";
}
)";

static const std::string g_function_source = R"(
{{extra_header}}

{{extra_code}}

{{instances}}

extern "C" __global__ void {{function_name}}(
     ck::half_t* in_ptr, 
     ck::half_t* weight_ptr, 
     ck::half_t* out_ptr, 
{% if "bias" in gemm_flag or gemm_flag == "add" %}
    void * bias_ptr,
{% endif %}
    int M,
    int N,
    int K,
    hipStream_t stream=NULL    
    )
{   
    {{shape_func}}
    {{input_addr_calculator}}
    {{output_addr_calculator}}
    {{exec_paths}}   
}
)";

static const std::string g_function_exec_source = R"(
    auto desc = {{instance}}::make_descriptor(ck::make_naive_tensor_descriptor_packed(ck::make_tuple(M, K)),
                                             ck::make_naive_tensor_descriptor(ck::make_tuple(N, K), ck::make_tuple(1, N)),
                                             ck::make_tuple(),
                                             ck::make_naive_tensor_descriptor_packed(ck::make_tuple(M, N)));

    if(desc.IsValid())
    {
        {{instance}}::Run(desc,
               in_ptr,
               weight_ptr,
               ck::make_tuple(),
               out_ptr);
    } else{
        printf("desc is unvalid");
    }
)";

static const std::string g_tensor_decl_source = R"(
    int64_t a_ptr_sz = M*K;
    int64_t b_ptr_sz = N*K;
    int64_t c_ptr_sz = M*N;
    int64_t ptr_max_sz = std::max({a_ptr_sz, b_ptr_sz, c_ptr_sz});
    // TODO: special pool size for 8M L2 cache
    // need to tune it for other devices
    int64_t mem_pool_sz = std::max(2,  std::min(64, int((1 << 23) / ptr_max_sz)));

    memory_pool->AllocateHalfTensor(a_ptr_sz, mem_pool_sz);  // x: index 0
    memory_pool->AllocateHalfTensor(b_ptr_sz, mem_pool_sz);  // w: index 1
    memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // y: index 2
{% if "bias" in gemm_flag %}
    memory_pool->AllocateHalfTensor(N, mem_pool_sz);  // b: index 3
{% endif %}
{% if has_d0 %}
    memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // d0 ptr: index 4
{% endif %}
{% if has_d1 %}
    memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // d1 ptr: index 5
{% endif %}
)";

static const std::string g_structs_source = R"(
struct ProfilerMemoryPool {
    ProfilerMemoryPool() {
        std::random_device rd;
        gen = std::mt19937(rd());
        uniform_dist = std::uniform_int_distribution<int64_t>(1, 48964896);
        offsets.reserve(512);
        strides.reserve(512);
        copies.reserve(512);
        ptrs.reserve(512);
    }
    ~ProfilerMemoryPool() {
        for(int i = 0; i < ptrs.size(); i++){
            hipFree(ptrs[i]);
        }
    }

    template <typename DType>
    DType* AllocateGaussianTensor(int64_t size) {
        size_t length = size * sizeof(DType);
        DType *d_x;
        hipMalloc(&d_x, length);

        float mean = 0.0f;
        float stddev = 1.0f;
        uint64_t seed = uniform_dist(gen);
        rocrand_set_seed(generator, seed);
        rocrand_generate_normal(generator, reinterpret_cast<float*>(d_x), size, mean, stddev);
        return d_x;
    }

    ck::half_t* AllocateHalfGaussianTensor(int64_t size) {
        return reinterpret_cast<ck::half_t*>(
            AllocateGaussianTensor<ck::half_t>(size));
    }

    int AllocateHalfTensor(int64_t size, int64_t copy) {
        offsets.push_back(0);
        strides.push_back(size);
        copies.push_back(copy);
        auto ptr = AllocateHalfGaussianTensor(size * copy);
        ptrs.push_back(reinterpret_cast<void*>(ptr));
        return ptrs.size() - 1;
    }

    ck::half_t* RequestHalfTensorByIdx(int idx) {
        auto copy = copies.at(idx);
        auto offset = offsets.at(idx);
        auto stride = strides.at(idx);
        ck::half_t* ptr = reinterpret_cast<ck::half_t*>(ptrs.at(idx));
        ptr += offset;
        offset += stride;
        if (offset == copy * stride) {
            offset = 0;
        }
        offsets[idx] = offset;
        return ptr;
    }
    std::vector<int64_t> offsets;
    std::vector<int64_t> strides;
    std::vector<int64_t> copies;
    std::vector<void*> ptrs;
    std::mt19937 gen;
    std::uniform_int_distribution<int64_t> uniform_dist;
    rocrand_generator generator;
};

// hack for DeviceMem linking error
// TODO fix this by making CK a header-only lib
// <<< hack begin
DeviceMem::DeviceMem(std::size_t mem_size) : mMemSize(mem_size)
{
    hipGetErrorString(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
}
void* DeviceMem::GetDeviceBuffer() const { return mpDeviceBuf; }
void DeviceMem::ToDevice(const void* p) const
{
    hipGetErrorString(
        hipMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, hipMemcpyHostToDevice));
}
void DeviceMem::FromDevice(void* p) const
{
    hipGetErrorString(hipMemcpy(p, mpDeviceBuf, mMemSize, hipMemcpyDeviceToHost));
}
DeviceMem::~DeviceMem() { hipGetErrorString(hipFree(mpDeviceBuf)); }
struct KernelTimerImpl
{
    KernelTimerImpl() {
        hipGetErrorString(hipEventCreate(&mStart));
        hipGetErrorString(hipEventCreate(&mEnd));
    }
    ~KernelTimerImpl() {
        hipGetErrorString(hipEventDestroy(mStart));
        hipGetErrorString(hipEventDestroy(mEnd));
    }
    void Start() {
        hipGetErrorString(hipDeviceSynchronize());
        hipGetErrorString(hipEventRecord(mStart, nullptr));
    }
    void End() {
        hipGetErrorString(hipEventRecord(mEnd, nullptr));
        hipGetErrorString(hipEventSynchronize(mEnd));
    }
    float GetElapsedTime() const {
        float time;
        hipGetErrorString(hipEventElapsedTime(&time, mStart, mEnd));
        return time;
    }
    hipEvent_t mStart, mEnd;
};
// >>> hack end

)";

static const std::string g_profiler_source = R"(
size_t GLOBAL_WORKSPACE_SIZE = 0;
{{op_func}}

{{structs_def}}

int main(int argc, char** argv) {
    if (argc < 4) {
        throw std::runtime_error("wrong params");
    }
    {{args_parse}}
    auto memory_pool = std::make_unique<ProfilerMemoryPool>();
    hipStream_t stream = nullptr;
    {{tensor_decl}}
    // TODO: random init
    // warmup
    for(int i = 0; i < 3; ++i) {
        {{func_call}}
    }
    // run
    auto timer = new KernelTimerImpl();
    timer->Start();
    for(int i = 0; i < 5; ++i) {
        {{func_call}}
    }
    timer->End();
    std::cout << "KERNEL:" << "{{kernel_config_name}}" << ",";
    std::cout << "TIME:" << timer->GetElapsedTime() << ",";
    std::cout << "WS:" << GLOBAL_WORKSPACE_SIZE << std::endl;
    delete(timer);

    return 0;
}
)";

static const std::string g_func_decl_source = R"(
void {{func_name}}(
  void *,
  void *,
  void *,
{% if "bias" in gemm_flag or gemm_flag == "add" %}
  void *,
{% endif %}
{% if has_d0 %}
  void *,
{% endif %}
{% if has_d1 %}
  void *,
{% endif %}
{% for idx in range(ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(pdims) %}
  const int,
{% endfor %}
  hipStream_t
);
)";

const static std::string g_func_call_source = R"(
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{{indent}}    {{weight_ptr}},
{{indent}}    {{out_ptr}},
{% if "bias" in gemm_flag or gemm_flag == "add" %}
{{indent}}    {{bias_ptr}},
{% endif %}
{% if d0_ptr != "" %}
{{indent}}    {{d0_ptr}},
{% endif %}
{% if d1_ptr != "" %}
{{indent}}    {{d1_ptr}},
{% endif %}
{% for dim in adims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in bdims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in cdims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in pdims %}
{{indent}}    {{dim}},
{% endfor %}
{{indent}}    stream
{{indent}});
)";

const static std::string g_shape_eval_source = R"(
{{indent}}{{dtype}} {{name}} = {{dim_calculator}};
)";

class GemmKernelCommon: public Kernel {
public:
    GemmKernelCommon() {}

    ~GemmKernelCommon() {}

    /*
    Extract (operation name, operation instance) pair
     from all operation candidates.

     Parameters
     ----------
     op_kind : ck_lib.library.OperationKind
         Operation kind.
     extra_kind : ck_lib.library.[AnyKind]
         Used to as extra flag to distinguish kernels.
         E.g. bias_add_relu vs. add_relu_bias
     f_prop_op: function
         Used to filter operation.

     Returns
     -------
     Dict
         Extracted (operation name, operation instance) pair.
    */
    std::map<std::string, std::shared_ptr<void>> ExtractConfig(const OperationKind&   op_kind,
                                                               const TensorOperation& extra_kind);

    // Exract name from the statement, e.g. 'model' for 'using model = xxx'.
    std::string ExetractConfigName(const std::string& config);

    std::string GenDimCalculator(const std::shared_ptr<DimInfo>& dim_info, bool is_ptr);

    std::string GenShapeEvalCode(const std::string&                                                  dtype,
                                 const std::map<std::string, std::vector<std::shared_ptr<DimInfo>>>& dim_info_map,
                                 bool                                                                is_ptr);

    /*
    Generates standalone executables for profiler.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    workdir : str
        Directory to store the generated outputs.
    dim_info_dict: Dict[str, DimInfo]
        Generated from gemm._extract_dims().
        Used to store mapping between dim_names to input / output tensor dims.
    args_parse: str
        Profiler input argument parser.
    gemm_flag : str
        Flag telling which backend should be generated. options are
    '','bias','bias_relu','bias_sigmoid','bias_add_relu'. extra_code : str Extra code for self-defined operators.
    ndims : int Number of dims for each parameter, 2 for gemm, 3 for bmm extra_shape_template: jinja2.Template Shape
    evaluation template. problem_args_template: jinja2.Template Problem args template for profiler.
    extra_header_template: jinja2.Template
        Extra header template as we have different headers for gemm and bmm.
    tensor_decl_template: jinja2.Template
        Tensor declaration template.
    */
    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenCommonKernelProfiler(const std::string&                                                  kernel_name,
                            const std::string&                                                  model_name,
                            const std::map<std::string, std::shared_ptr<void>>&                 kernel_instance_map,
                            const int                                                           num_sources,
                            const std::map<std::string, std::vector<std::shared_ptr<DimInfo>>>& dim_info_map,
                            const std::string&                                                  arg_parse,
                            const std::string&                                                  permute_shape = "",
                            const std::string&                                                  extra_code    = "",
                            const std::string&                                                  gemm_flag     = "",
                            const int                                                           ndims         = 2,
                            const std::string& extra_shape_template   = g_extra_shape_source,
                            const std::string& problem_args_template  = g_problem_args_source,
                            const std::string& extra_header_template  = g_extra_header_source,
                            const std::string& tensor_decl_template   = g_tensor_decl_source,
                            const std::string& input_addr_calculator  = "",
                            const std::string& output_addr_calculator = "",
                            const std::string& folder_name            = "kernel_profile");

    std::string
    GenCommonKernelFunction(const std::string&                                                  func_name,
                            const std::map<std::string, std::shared_ptr<void>>&                 kernel_instance_map,
                            const int                                                           num_sources,
                            const std::map<std::string, std::shared_ptr<ExecItem>>&             exec_path,
                            const std::string                                                   permute_shape,
                            const std::map<std::string, std::vector<std::shared_ptr<DimInfo>>>& dim_info_map,
                            const std::string&                                                  gemm_flag  = "",
                            const std::string&                                                  extra_code = "",
                            const int                                                           ndims      = 2,
                            const std::string& exec_cond_template     = g_exec_cond_source,
                            const std::string& extra_shape_template   = g_extra_shape_source,
                            const std::string& problem_args_template  = g_problem_args_source,
                            const std::string& extra_header_template  = g_extra_header_source,
                            const std::string& input_addr_calculator  = "",
                            const std::string& output_addr_calculator = "");
};
}  // namespace ater