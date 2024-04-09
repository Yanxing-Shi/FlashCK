#pragma once

#include <memory>
#include <random>
#include <vector>

#include "ater/core/graph/common.h"
#include "ater/core/graph/context.h"
#include "ater/core/graph/node.h"
#include "ater/core/graph/shape.h"
#include "ater/core/utils/memory_utils.h"

#include "ater/core/module/layers/linear_layer.h"

namespace ater {

// template<typename T>
// std::vector<T> generate_buffer(std::size_t n, std::size_t m, std::size_t seed = 0)
// {
//     std::vector<T> result(n * m);
//     // std::mt19937                           gen(seed);
//     // std::uniform_real_distribution<double> dis(-1.0);
//     // std::generate(result.begin(), result.end(), [&] { return dis(gen); });
//     for (int i = 0; i < n * m; i++) {
//         result[i] = static_cast<T>(i);
//     }
//     return result;
// }

template<typename T>
void DoubleGemmModel()
{
    Context::CreateGlobalContext("linear", Mode::Inference);
    auto context_ptr = Context::GetGlobalInstance();

    // step1. define model
    // input arguments
    int in_channels  = 8;  // K
    int out_channels = 8;  // N

    int batch_size   = 2;
    int seq_len      = 4;
    int batch_tokens = batch_size * seq_len;  // M

    // load weight and bias
    std::vector<T> in     = GenHostRandomBuffer<T>(64, 0);
    std::vector<T> weight = GenHostRandomBuffer<T>(64, 1);
    std::vector<T> out    = GenHostRandomBuffer<T>(64, 2);
    // allocate memory on gpu
    T* in_ptr;
    T* weight_ptr;
    T* out_ptr;

    hipMalloc(&in_ptr, 8 * 8 * sizeof(T));
    hipMalloc(&weight_ptr, 8 * 8 * sizeof(T));
    hipMalloc(&out_ptr, 8 * 8 * sizeof(T));

    // copy data from cpu to gpu
    hipDeviceSynchronize();
    hipMemcpy(in_ptr, in.data(), 8 * 8 * sizeof(T), hipMemcpyHostToDevice);
    hipMemcpy(weight_ptr, weight.data(), 8 * 8 * sizeof(T), hipMemcpyHostToDevice);
    hipMemcpy(out_ptr, out.data(), 8 * 8 * sizeof(T), hipMemcpyHostToDevice);

    // step2.build model graph(dynamic shape inference)
    // input node
    Variable* input_var = new Variable("input_var", CppTypeToDataType<T>::Type());
    // M>=1 && M<=batch_tokens, K>=1 && K<=in_channels
    Shape input_var_shape = Shape({DDim({1, batch_tokens}), DDim(in_channels)});
    input_var->SetShape(input_var_shape);  // infer shape, only memory allocated

    // create linear layer and set weight and bias(RCR)
    // y = xA^T + b
    auto linear_layer = std::make_shared<LinearLayer<T>>(in_channels, out_channels);
    linear_layer->LoadParam(weight_ptr);
    // // build model graph
    Variable* output_var = (*linear_layer)(input_var);
    context_ptr->BuildContext();

    // // step3. codegen profiler
    context_ptr->CodegenAndProfileKernel();

    // // step.4 malloc input tensor and output tensor
    input_var->SetValue((char*)in_ptr);
    input_var->SetShape(Shape({DDim(batch_tokens), DDim(in_channels)}));  // actual tensor shape, for model forward
    output_var->SetValue((char*)out_ptr);
    output_var->SetShape(Shape({DDim(batch_tokens), DDim(out_channels)}));  // actual tensor shape, for model forward

    // // step5. run model and inference
    linear_layer->Forward();

    // print the result
    hipDeviceSynchronize();
    hipMemcpy(out.data(), out_ptr, 8 * 8 * sizeof(T), hipMemcpyDeviceToHost);

    // // step6. check output tensor
    // printf in
    auto input_value = (T*)input_var->GetValue();
    std::cout << "====a =====\n " << std::endl;
    for (int i = 0; i < in_channels; i++) {
        for (int j = 0; j < batch_tokens; j++) {
            std::cout << static_cast<float>(input_value[i * batch_tokens + j]) << " ";
        }
        printf("\n");
    }
    std::cout << "====b =====\n " << std::endl;
    for (int i = 0; i < out_channels; i++) {
        for (int j = 0; j < in_channels; j++) {
            std::cout << static_cast<float>(weight[i * in_channels + j]) << " ";
        }
        printf("\n");
    }

    // print out
    std::cout << "====c =====\n " << std::endl;
    auto output_value = (T*)output_var->GetValue();
    for (int i = 0; i < out_channels; i++) {
        for (int j = 0; j < batch_tokens; j++) {
            std::cout << static_cast<float>(output_value[i * batch_tokens + j]) << " ";
        }
        printf("\n");
    }
}
}  // namespace ater
