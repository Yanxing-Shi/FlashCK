#include <ATen/ATen.h>

#include "flashck/wrapper/cpp/norm/layer_norm.h"

namespace flashck {

namespace pytorch {

at::Tensor layer_norm_fwd(
    at::Tensor input, const std::vector<int64_t>& normalized_shape, at::Tensor gamma, at::Tensor beta, float eps)
{
    // Check the input tensor is not empty
    TORCH_CHECK(input.numel() != 0, "Input tensor is empty");
    // Check the gamma and beta tensors are not empty
    TORCH_CHECK(gamma.numel() != 0, "gamma tensor is empty");
    TORCH_CHECK(beta.numel() != 0, "beta tensor is empty");
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
        auto output_ptr = cpp::layer_norm_fwd<float>(
            input_reshaped.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), m, n, eps);
        output = at::from_blob(output_ptr, {m, n}, at::kFloat).clone();
    }
    else if (input.dtype() == at::kHalf) {
        auto output_ptr = cpp::layer_norm_fwd<_Float16>(
            input_reshaped.data_ptr<_Float16>(), gamma.data_ptr<_Float16>(), beta.data_ptr<_Float16>(), m, n, eps);
        output = at::from_blob(output_ptr, {m, n}, at::kHalf).clone();
    }
    else if (input.dtype() == at::kBFloat16) {
        auto output_ptr = cpp::layer_norm_fwd<ushort>(
            input_reshaped.data_ptr<ushort>(), gamma.data_ptr<ushort>(), beta.data_ptr<ushort>(), m, n, eps);
        output = at::from_blob(output_ptr, {m, n}, at::kBFloat16).clone();
    }
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