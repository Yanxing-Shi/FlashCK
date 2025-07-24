#include <ATen/ATen.h>
#include <ATen/Tensor.h>

#include "flashck/norm/layer_norm.h"

namespace flashck {

namespace pytorch {

at::Tensor layer_norm_fwd(
    at::Tensor input, const std::vector<int64_t>& normalized_shape, at::Tensor weight, at::Tensor bias, float eps)
{
    // Check the input tensor is not empty
    TORCH_CHECK(input.numel() != 0, "Input tensor is empty");
    // Check the weight and bias tensors are not empty
    TORCH_CHECK(weight.numel() != 0, "weight tensor is empty");
    TORCH_CHECK(bias.numel() != 0, "bias tensor is empty");
    // Check the normalized_shape is not empty
    TORCH_CHECK(!normalized_shape.empty(), "normalized_shape is empty");

    for (size_t i = 0; i < normalized_shape.size(); ++i) {
        TORCH_CHECK(normalized_shape[i] > 0, "normalized_shape must be greater than 0");
        int input_dim = input.dim() - normalized_shape.size() + i;
        TORCH_CHECK(input.size(input_dim) == normalized_shape[i],
                    "Input tensor size at dimension ",
                    input_dim,
                    " is ",
                    input.size(input_dim),
                    " but expected ",
                    normalized_shape[i]);
    }

    // If the input is a 2D tensor or higher, reshape it to 2D according to the normalized_shape
    std::vector<int64_t> original_shape(input.sizes().begin(), input.sizes().end());
    at::Tensor           input_reshaped = input;

    if (input.dim() != 2) {
        input_reshaped = input.view({input.size(0), -1});
    }

    int m = input_reshaped.size(0);
    int n = input_reshaped.size(1);

    at::Tensor output;

    // Handle different data types
    if (input.dtype() == at::kFloat) {
        auto output_ptr = flashck::naive::layer_norm_fwd<float>(
            input_reshaped.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), m, n, eps);
        output = at::from_blob(output_ptr, {m, n}, at::kFloat).clone();
    }
    // else if (input.dtype() == at::kHalf) {
    //     auto output_ptr = flashck::naive::layer_norm_fwd<at::kHalf>(
    //         input_reshaped.data_ptr<at::kHalf>(), weight.data_ptr<at::kHalf>(), bias.data_ptr<at::kHalf>(), m, n, eps);
    //     output = at::from_blob(output_ptr, {m, n}, at::kHalf).clone();
    // }
    // else if (input.dtype() == at::kBFloat16) {
    //     auto output_ptr = flashck::naive::layer_norm_fwd<ushort>(
    //         input_reshaped.data_ptr<ushort>(), weight.data_ptr<ushort>(), bias.data_ptr<at::kBFloat16>(), m, n, eps);
    //     output = at::from_blob(output_ptr, {m, n}, at::kBFloat16).clone();
    // }
    else {
        TORCH_CHECK(false, "Unsupported data type for layer norm");
    }

    // Reshape back to original shape if needed
    if (input.dim() != 2) {
        output = output.view(original_shape);
    }

    return output;
}

}  // namespace pytorch

}  // namespace flashck