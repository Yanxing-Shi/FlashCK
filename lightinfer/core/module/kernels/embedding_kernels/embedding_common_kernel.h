#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "lightinfer/core/module/kernels/kernel.h"

static const std::string g_exec_cond_source = R"(
{{indent}}if ({{cond}}) {
{{indent}}  {{program}}
{{indent}}}
)";

static const std::string g_extra_header_source = R"(
#include "ck/tensor_operation/gpu/device/impl/device_sparse_embeddings_forward_layernorm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
)";

static const std::string g_macro_decl_source = R"(
// We compile all models with -fvisibility=hidden. Any symbols that need to be
// exposed in the final shared library must be declared with ATER_EXPORT to make
// them visible.
#ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
#define ATER_EXPORT __attribute__((__visibility__("default")))
#else
#ifdef _WIN32
#define ATER_EXPORT __declspec(dllexport)
#else
#define ATER_EXPORT
#endif
#endif
)";

static const std::string g_profiler_header_source = R"(

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

)";

static const std::string g_dtype_decl_source = R"(
using EmbType                 = {{emb_dtype}};
using IndexType               = {{index_dtype}};
using GammaDataType           = {{gamma_dtype}};
using BetaDataType            = {{beta_dtype}};
using AccDataType             = {{acc_dtype}};
using OutType                 = {{out_dtype}};
{% if embedding_flag == "add_add_layer_norm"  %}
using EmbElementwiseOperation = ck::tensor_operation::element_wise::AddAdd;
{% else %}
using EmbElementwiseOperation = ck::tensor_operation::element_wise::PassThrough;
{% endif %}
)";

static const std::string g_instance_source = R"(
{{config}}
using {{name}} = {{config_name}};
)";

static const std::string g_exec_source = R"(
{{indent}}auto device_instance =  {{instance}}{};
{% if embedding_flag == "add_add_layer_norm"  %}
{{indent}}auto argument_ptr = device_instance.MakeArgumentPointer(
        ck::type_convert<OutType*>(out_dev_buffer),
        {ck::type_convert<EmbType*>(emb_a_dev_buffer),
         ck::type_convert<EmbType*>(emb_b_dev_buffer),
         ck::type_convert<EmbType*>(emb_c_dev_buffer)},
        {ck::type_convert<IndexType*>(index_a_dev_buffer),
         ck::type_convert<IndexType*>(index_b_dev_buffer),
         ck::type_convert<IndexType*>(index_c_dev_buffer)},
        ck::type_convert<GammaDataType*>(gamma_dev_buffer),
        ck::type_convert<BetaDataType*>(beta_dev_buffer),
        embedding_dims,
        num_indices,
        epsilon,
        EmbElementwiseOperation{}
        );
{% else %}
{{indent}}auto argument_ptr = device_instance.MakeArgumentPointer(
        ck::type_convert<OutType*>(out_dev_buffer),
        {ck::type_convert<EmbType*>(emb_a_dev_buffer)},
        {ck::type_convert<IndexType*>(index_a_dev_buffer)},
        ck::type_convert<GammaDataType*>(gamma_dev_buffer),
        ck::type_convert<BetaDataType*>(beta_dev_buffer),
        embedding_dims,
        num_indices,
        epsilon,
        EmbElementwiseOperation{}
        );
{% endif %}
{{indent}}if(!device_instance.IsSupportedArgument(argument_ptr.get())) {
{{indent}}  std::cerr << "wrong! " << device_instance.GetTypeString() << " with the specified compilation parameters does not support this embdding problem.";
{{indent}} return;} 
{{indent}}auto invoker_ptr = device_instance.MakeInvokerPointer();
{{indent}}invoker_ptr->Run(argument_ptr.get(), StreamConfig{stream, false});
)";

static const std::string g_structs_source = R"(
constexpr int error_exit_code = -1;

#define HIP_CHECK(condition)                                                                                           \
    {                                                                                                                  \
        const hipError_t error = condition;                                                                            \
        if (error != hipSuccess) {                                                                                     \
            std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " << __FILE__ << ':'         \
                      << __LINE__ << std::endl;                                                                        \
            std::exit(error_exit_code);                                                                                \
        }                                                                                                              \
    }

class HipTimer {
private:
    hipEvent_t  event_start_;
    hipEvent_t  event_stop_;
    hipStream_t stream_;

public:
    explicit HipTimer(hipStream_t stream = 0)
    {
        stream_ = stream;
    }
    void Start()
    {
        HIP_CHECK(hipEventCreate(&event_start_));
        HIP_CHECK(hipEventCreate(&event_stop_));
        HIP_CHECK(hipEventRecord(event_start_, stream_));
    }
    float Stop()
    {
        float time;
        HIP_CHECK(hipEventRecord(event_stop_, stream_));
        HIP_CHECK(hipEventSynchronize(event_stop_));
        HIP_CHECK(hipEventElapsedTime(&time, event_start_, event_stop_));
        HIP_CHECK(hipEventDestroy(event_start_));
        HIP_CHECK(hipEventDestroy(event_stop_));
        return time;
    }
    ~HipTimer() {}
};
)";

static const std::string g_func_call_source = R"(

{{indent}}  {{func_name}}(
{{indent}}            out_dev_buffer,
{{indent}}            emb_a_dev_buffer,
{% if embedding_flag == "add_add_layer_norm" %}
{{indent}}            emb_b_dev_buffer,
{{indent}}            emb_c_dev_buffer,
{% endif %}
{{indent}}            index_a_dev_buffer,
{% if embedding_flag == "add_add_layer_norm" %}
{{indent}}            index_b_dev_buffer,
{{indent}}            index_c_dev_buffer,
{% endif %}
{{indent}}            gamma_dev_buffer,
{{indent}}            beta_dev_buffer,
{{indent}}            embedding_dims,
{{indent}}            num_indices,
{{indent}}            epsilon,
{{indent}}            stream);
)";

static const std::string g_tensor_decl_source = R"(
    auto f_host_tensor_desc_1d = [](std::size_t len_) { return HostTensorDescriptor({len_}); };

    auto f_host_tensor_desc_2d = [](std::size_t rows_, std::size_t cols_) {
        return HostTensorDescriptor({rows_, cols_});
    };

{% if embedding_flag == "add_add_layer_norm" %}
    Tensor<EmbType>       emb_a(f_host_tensor_desc_2d(word_embeddings, embedding_dims));
    Tensor<EmbType>       emb_b(f_host_tensor_desc_2d(token_type_embeddings, embedding_dims));
    Tensor<EmbType>       emb_c(f_host_tensor_desc_2d(position_embeddings, embedding_dims));
{% else %}
    Tensor<EmbType>       emb_a(f_host_tensor_desc_2d(num_embeddings, embedding_dims));
{% endif %}

{% if embedding_flag == "add_add_layer_norm" %}
    Tensor<IndexType>     index_a(f_host_tensor_desc_1d(num_indices));
    Tensor<IndexType>     index_b(f_host_tensor_desc_1d(num_indices));
    Tensor<IndexType>     index_c(f_host_tensor_desc_1d(num_indices));
{% else %}
    Tensor<IndexType>     index_a(f_host_tensor_desc_1d(num_indices));
{% endif %}

    Tensor<GammaDataType> gamma(f_host_tensor_desc_1d(embedding_dims));
    Tensor<BetaDataType>  beta(f_host_tensor_desc_1d(embedding_dims));
    Tensor<OutType>       out(f_host_tensor_desc_2d(num_indices, embedding_dims));

{% if embedding_flag == "add_add_layer_norm" %}
    emb_a.GenerateTensorValue(GeneratorTensor_3<EmbType>{0.0, 1.0});
    emb_b.GenerateTensorValue(GeneratorTensor_3<EmbType>{0.0, 1.0});
    emb_c.GenerateTensorValue(GeneratorTensor_3<EmbType>{0.0, 1.0});
{% else %}
    emb_a.GenerateTensorValue(GeneratorTensor_3<EmbType>{0.0, 1.0});
{% endif %}

{% if embedding_flag == "add_add_layer_norm" %}
    index_a.GenerateTensorValue(GeneratorTensor_2<IndexType>{0, word_embeddings});
    index_b.GenerateTensorValue(GeneratorTensor_2<IndexType>{0, token_type_embeddings});
    index_c.GenerateTensorValue(GeneratorTensor_2<IndexType>{0, position_embeddings});
{% else %}
    index_a.GenerateTensorValue(GeneratorTensor_2<IndexType>{0, num_embeddings});
{% endif %}
    gamma.GenerateTensorValue(GeneratorTensor_3<GammaDataType>{0.0, 1.0});
    beta.GenerateTensorValue(GeneratorTensor_3<BetaDataType>{0.0, 1.0});

{% if embedding_flag == "add_add_layer_norm" %}
    DeviceMem emb_a_dev(sizeof(EmbType) * emb_a.mDesc.GetElementSpaceSize());
    DeviceMem emb_b_dev(sizeof(EmbType) * emb_b.mDesc.GetElementSpaceSize());
    DeviceMem emb_c_dev(sizeof(EmbType) * emb_c.mDesc.GetElementSpaceSize());
    DeviceMem index_a_dev(sizeof(IndexType) * index_a.mDesc.GetElementSpaceSize());
    DeviceMem index_b_dev(sizeof(IndexType) * index_b.mDesc.GetElementSpaceSize());
    DeviceMem index_c_dev(sizeof(IndexType) * index_c.mDesc.GetElementSpaceSize());
{% else %}
    DeviceMem emb_a_dev(sizeof(EmbType) * emb_a.mDesc.GetElementSpaceSize());
    DeviceMem index_a_dev(sizeof(IndexType) * index_a.mDesc.GetElementSpaceSize());
{% endif %}
    DeviceMem gamma_dev(sizeof(GammaDataType) * gamma.mDesc.GetElementSpaceSize());
    DeviceMem beta_dev(sizeof(BetaDataType) * beta.mDesc.GetElementSpaceSize());
    DeviceMem out_dev(sizeof(OutType) * out.mDesc.GetElementSpaceSize());

{% if embedding_flag == "add_add_layer_norm" %}
    emb_a_dev.ToDevice(emb_a.mData.data());
    emb_b_dev.ToDevice(emb_b.mData.data());
    emb_c_dev.ToDevice(emb_c.mData.data());
    index_a_dev.ToDevice(index_a.mData.data());
    index_b_dev.ToDevice(index_b.mData.data());
    index_c_dev.ToDevice(index_c.mData.data());
{% else %}
    emb_a_dev.ToDevice(emb_a.mData.data());
    index_a_dev.ToDevice(index_a.mData.data());
{% endif %}
    gamma_dev.ToDevice(gamma.mData.data());
    beta_dev.ToDevice(beta.mData.data());

    auto out_dev_buffer     = out_dev.GetDeviceBuffer();
{% if embedding_flag == "add_add_layer_norm" %}
    auto emb_a_dev_buffer   = emb_a_dev.GetDeviceBuffer();
    auto emb_b_dev_buffer   = emb_b_dev.GetDeviceBuffer();
    auto emb_c_dev_buffer   = emb_c_dev.GetDeviceBuffer();
    auto index_a_dev_buffer = index_a_dev.GetDeviceBuffer();
    auto index_b_dev_buffer = index_b_dev.GetDeviceBuffer();
    auto index_c_dev_buffer = index_c_dev.GetDeviceBuffer();
{% else %}
    auto emb_a_dev_buffer   = emb_a_dev.GetDeviceBuffer();
    auto index_a_dev_buffer = index_a_dev.GetDeviceBuffer();
{% endif %}
    auto gamma_dev_buffer   = gamma_dev.GetDeviceBuffer();
    auto beta_dev_buffer    = beta_dev.GetDeviceBuffer();
)";

static const std::string g_profiler_source = R"(
{{op_func}}

{{structs_def}}

int main(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error("wrong params");
    }
    
 
    auto num_indices    = std::stoi(argv[1]);
{% if embedding_flag == "add_add_layer_norm" %}
    auto word_embeddings = std::stoi(argv[2]);
    auto token_type_embeddings     = std::stoi(argv[3]);
    auto position_embeddings       = std::stoi(argv[4]);
    auto embedding_dims = std::stoi(argv[5]);
{% else %}
    auto num_embeddings = std::stoi(argv[2]);
    auto embedding_dims = std::stoi(argv[3]);
{% endif %}

    {{tensor_decl}}

    constexpr float epsilon = 1e-4;
    constexpr int warm_iter    = 3;
    constexpr int profile_iter = 5;
    hipStream_t stream = nullptr;

    // warmup
    for(int i = 0; i < warm_iter; ++i) {
        {{func_call}}
    }

    // run
    HipTimer hip_timer;
    hip_timer.Start();
    for(int i = 0; i < profile_iter; ++i) {
        {{func_call}}
    }

    float  ave_time  = hip_timer.Stop() / profile_iter;

    std::cout << "KERNEL:" << "{{kernel_config_name}}" << ",";
    std::cout << "TIME:" << ave_time << "ms";

    return 0;
}
)";

static const std::string g_func_source = R"(
#include "ck/ck.hpp"

{{macro_decl}}

{{extra_header}}

{{profiler_header}}

{{dtype_decl}}

{{instances_decl}}

{% if is_execute %} {{c_flag}} ATER_EXPORT {% endif %} void {{func_name}}(
                 void* out_dev_buffer,
                 void* emb_a_dev_buffer,
{% if embedding_flag == "add_add_layer_norm" %}
                 void* emb_b_dev_buffer,
                 void* emb_c_dev_buffer,
{% endif %}
                 void* index_a_dev_buffer,
{% if embedding_flag == "add_add_layer_norm" %}
                 void* index_b_dev_buffer, 
                 void* index_c_dev_buffer,
{% endif %}
                 void* gamma_dev_buffer,
                 void* beta_dev_buffer,
                int64_t embedding_dims,
                int64_t num_indices,
                float epsilon,
                hipStream_t stream){
    
    {{exec_path}}
    
}
)";

namespace lightinfer {

class EmbeddingCommonKernel: public Kernel {
public:
    EmbeddingCommonKernel()          = default;
    virtual ~EmbeddingCommonKernel() = default;
    std::map<std::string, std::shared_ptr<void>> ExtractConfig(const EmbeddingOperationKind& op_kind,
                                                               const TensorOperation&        extra_kind);

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenCommonKernelProfiler(const std::string&                               model_name,
                            const std::unordered_map<std::string, std::any>& kernel_func_map,
                            const std::string&                               embedding_flag,
                            const std::string&                               folder_name = "kernel_profile");

    std::string GenCommonKernelFunction(const std::string&                               func_name,
                                        const std::unordered_map<std::string, std::any>& kernel_func_map,
                                        const std::string&                               embedding_flag);
};

}  // namespace lightinfer
