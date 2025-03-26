#pragma once

#include <algorithm>

#include "tests/utils/torch_utils.h"

/*----------------------------------------------------------------mask---------------------------------*/
// get random mask for q_mask and k_mask
template<typename T>
inline torch::Tensor
GetRandomPaddingMask(const int64_t max_seq_len, const int64_t batch_size, const std::string& mode = "random")
{
    torch::Tensor length;
    if (mode == "full") {
        length = GetFullTorchTensor<int32_t>({batch_size, 1}, max_seq_len);
    }
    else if (mode == "random") {
        length = GetRandomIntTorchTensor({batch_size, 1}, std::max((int64_t)1, max_seq_len - 20), max_seq_len + 1);
    }
    else if (mode == "third") {
        length = GetRandomIntTorchTensor({batch_size, 1}, max_seq_len / 3, max_seq_len + 1);
    }
    else {
        throw std::invalid_argument("Invalid mode");
    }

    // [batch_size, max_seq_len]
    torch::Tensor padding_mask = GetArrangeTorchTensor<T>(0, max_seq_len, 1)
                                     .expand({batch_size, max_seq_len})
                                     .lt(length.expand({batch_size, max_seq_len}));

    return padding_mask;
}

// get local attention mask for sliding window
template<typename T>
inline torch::Tensor GetLocalMaskFromSlidingWindow(
    const int64_t                       batch_size,
    const int64_t                       q_seq_len,
    const int64_t                       kv_seq_len,
    const std::array<int64_t, 2>&       window_size,
    const std::optional<torch::Tensor>& query_padding_mask = std::nullopt,  // [batch_size, q_seq_len]
    const std::optional<torch::Tensor>& key_padding_mask   = std::nullopt,
    // [batch_size, kv_seq_len]
    const std::optional<torch::Tensor>& key_leftpad = std::nullopt)  // [batch_size]
{
    if (window_size[0] == -1 && window_size[1] == -1) {
        return torch::Tensor();  // empty tensor
    }

    torch::Tensor row_idx = GetArrangeTorchTensor<int64_t>(0, q_seq_len, 1).reshape({-1, 1});  // [q_seq_len, 1]
    torch::Tensor col_idx = GetArrangeTorchTensor<int64_t>(0, kv_seq_len, 1);                  // [kv_seq_len]
    if (key_leftpad.has_value()) {
        auto key_leftpad_reshape = key_leftpad.value().reshape({-1, 1, 1});  // [batch_size, 1, 1]
        col_idx =
            col_idx.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat({batch_size, 1, 1});  // [batch_size, 1, kv_seq_len]
        col_idx = torch::where(col_idx >= key_leftpad_reshape, col_idx - key_leftpad_reshape, 2e32);
    }

    //  [batch_size, 1,1]
    auto in_kv_seq_len =
        key_padding_mask.has_value() ?
            key_padding_mask.value().sum(-1).reshape({key_padding_mask.value().sizes()[0], 1, 1}) :
            torch::tensor(kv_seq_len, torch::kInt64).reshape({1, 1, 1}).expand({batch_size, -1, -1}).to(torch::kCUDA);
    //  [batch_size, 1,1]
    auto in_q_seq_len =
        query_padding_mask.has_value() ?
            query_padding_mask.value().sum(-1).reshape({query_padding_mask.value().sizes()[0], 1, 1}) :
            torch::tensor(q_seq_len, torch::kInt64).reshape({1, 1, 1}).expand({batch_size, -1, -1}).to(torch::kCUDA);

    if (window_size[0] < 0) {
        return col_idx.gt(row_idx + in_kv_seq_len - in_q_seq_len
                          + window_size[1]);  // [batch_size, q_seq_len, kv_seq_len]
    }
    else {
        in_kv_seq_len = key_padding_mask.has_value() ? torch::full_like(col_idx, kv_seq_len) : in_kv_seq_len;
        return torch::logical_or(
            col_idx.gt(torch::minimum(row_idx + in_kv_seq_len - in_q_seq_len + window_size[1], in_kv_seq_len)),
            col_idx.lt(row_idx + in_kv_seq_len - in_q_seq_len
                       - window_size[0]));  // [batch_size, q_seq_len, kv_seq_len]
    }
}
