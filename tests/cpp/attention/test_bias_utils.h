#pragma once

#include <tuple>

#include "flashck/core/profiling/library.h"
#include "tests/utils/torch_utils.h"

template<typename T>
inline torch::Tensor GetSlopesTensor(const int64_t            num_heads,
                                     const int64_t            batch_size,
                                     const flashck::BiasEnum& bias_enum,
                                     const int64_t            bias_rank_info)
{
    int64_t       n   = std::pow(2, std::floor(std::log2(num_heads)));
    float         m_0 = std::pow(2.0, -8.0 / n);
    torch::Tensor m   = torch::pow(m_0, GetArrangeTorchTensor<T>(1, 1 + n, 1));

    if (n < num_heads) {
        float         m_hat_0 = std::pow(2.0, -4.0 / n);
        torch::Tensor m_hat   = torch::pow(m_hat_0, GetArrangeTorchTensor<T>(1, 1 + 2 * (num_heads - n), 2));
        m                     = torch::cat({m, m_hat});  // [num_heads]
    }

    // [1, num_heads] or [batch_size, num_heads]
    return bias_rank_info == 0 ? m.reshape({1, -1}) : m.reshape({1, -1}).expand({batch_size, -1});
}

inline torch::Tensor GetAttentionBiasFromAlibiSlopes(
    const int64_t                 batch_size,
    const int64_t                 q_seq_len,
    const int64_t                 kv_seq_len,
    const int64_t                 q_num_heads,
    const flashck::BiasEnum&      bias_enum,
    const int64_t                 bias_rank_info,
    const std::array<int64_t, 2>& window_size,
    const torch::Tensor&          slopes,  // rank_info 0: [1, q_num_heads] or rank_info 1: [batch_size, q_num_heads]
    const std::optional<torch::Tensor>& query_padding_mask =
        std::nullopt,  // [batch_size, q_seq_len], which is used for group mode.
    const std::optional<torch::Tensor>& key_padding_mask =
        std::nullopt,  // [batch_size, kv_seq_len], which is used for group mode.
    const std::optional<torch::Tensor>& key_leftpad = std::nullopt  // [batch_size]
)
{
    // slopes: ->[batch_size, q_num_heads, 1, 1]
    torch::Tensor reshape_slopes;
    if (bias_rank_info == 0) {
        reshape_slopes = slopes.reshape({1, slopes.sizes()[1], 1, 1}).expand({batch_size, -1, -1, -1});
    }
    else {
        reshape_slopes = slopes.reshape({slopes.sizes()[0], slopes.sizes()[1], 1, 1});
    }

    if (window_size[0] < 0 && window_size[1] == 0) {
        // casual
        return GetArrangeTorchTensor<float>(-kv_seq_len + 1, 1, 1)
               * reshape_slopes;  // [batch_size, q_num_heads, kv_seq_len, q_seq_len]
    }
    else {
        // local
        torch::Tensor row_idx = GetArrangeTorchTensor<int64_t>(0, q_seq_len, 1).reshape({-1, 1});  // [q_seq_len, 1]
        torch::Tensor col_idx = GetArrangeTorchTensor<int64_t>(0, kv_seq_len, 1);                  // [kv_seq_len]
        if (key_leftpad.has_value()) {
            auto key_leftpad_reshape = key_leftpad.value().reshape({-1, 1, 1, 1});  // [batch_size, 1, 1, 1]
            col_idx                  = col_idx.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(
                {batch_size, 1, 1, 1});  // [batch_size, 1, 1, kv_seq_len]
            col_idx = torch::where(col_idx >= key_leftpad_reshape, col_idx - key_leftpad_reshape, 2e32);
        }

        //  [batch_size, 1,1,1]
        auto in_kv_seq_len =
            key_padding_mask.has_value() ?
                key_padding_mask.value().sum(-1).reshape({key_padding_mask.value().sizes()[0], 1, 1, 1}) :
                torch::tensor(kv_seq_len, torch::kInt64)
                    .reshape({1, 1, 1, 1})
                    .expand({batch_size, -1, -1, -1})
                    .to(torch::kCUDA);
        //  [batch_size, 1,1,1]
        auto in_q_seq_len =
            query_padding_mask.has_value() ?
                query_padding_mask.value().sum(-1).reshape({query_padding_mask.value().sizes()[0], 1, 1, 1}) :
                torch::tensor(q_seq_len, torch::kInt64)
                    .reshape({1, 1, 1, 1})
                    .expand({batch_size, -1, -1, -1})
                    .to(torch::kCUDA);

        auto relative_pos =
            torch::abs(row_idx + in_kv_seq_len - in_q_seq_len - col_idx);  // [batch_size, 1, q_seq_len, kv_seq_len]

        return -reshape_slopes
               * relative_pos.to(reshape_slopes.dtype());  // [batch_size, num_heads, q_seq_len, kv_seq_len]
    }
}

template<typename T>
inline std::tuple<torch::Tensor, torch::Tensor>
CreateAttentionBiasTensor(const int64_t                       batch_size,
                          const int64_t                       q_seq_len,
                          const int64_t                       kv_seq_len,
                          const int64_t                       q_num_heads,
                          const flashck::BiasEnum&            bias_enum,
                          const int64_t                       bias_rank_info,
                          const std::array<int64_t, 2>&       window_size,
                          const std::optional<torch::Tensor>& query_padding_mask =
                              std::nullopt,  // [batch_size, q_seq_len], which is used for group mode.
                          const std::optional<torch::Tensor>& key_padding_mask =
                              std::nullopt,  // [batch_size, kv_seq_len], which is used for group mode.
                          const std::optional<torch::Tensor>& key_leftpad = std::nullopt  // [batch_size]
)
{
    torch::Tensor bias_tensor, alibi_slopes_tensor;

    if (bias_enum == flashck::BiasEnum::NO_BIAS) {
        return std::make_tuple(bias_tensor, alibi_slopes_tensor);
    }
    else if (bias_enum == flashck::BiasEnum::ELEMENTWISE_BIAS) {
        if (bias_rank_info == 0) {
            bias_tensor = GetRandomTorchTensor<T>({1, 1, q_seq_len, kv_seq_len});
        }
        else if (bias_rank_info == 1) {
            bias_tensor = GetRandomTorchTensor<T>({1, q_num_heads, q_seq_len, kv_seq_len});
        }
        else {
            bias_tensor = GetRandomTorchTensor<T>({batch_size, q_num_heads, q_seq_len, kv_seq_len});
        }
        return std::make_tuple(bias_tensor, alibi_slopes_tensor);
    }
    else if (bias_enum == flashck::BiasEnum::ALIBI) {
        if (bias_rank_info == 0) {
            // [1, q_num_heads]
            alibi_slopes_tensor = GetSlopesTensor<float>(q_num_heads, batch_size, bias_enum, bias_rank_info);
            // [batch_size, q_num_heads, q_seq_len, kv_seq_len]
            bias_tensor = GetAttentionBiasFromAlibiSlopes(batch_size,
                                                          q_seq_len,
                                                          kv_seq_len,
                                                          q_num_heads,
                                                          bias_enum,
                                                          bias_rank_info,
                                                          window_size,
                                                          alibi_slopes_tensor,
                                                          query_padding_mask,
                                                          key_padding_mask,
                                                          key_leftpad);
        }
        else {
            // [batch_size, q_num_heads]
            alibi_slopes_tensor = GetSlopesTensor<float>(q_num_heads, batch_size, bias_enum, bias_rank_info);
            // [batch_size, q_num_heads, q_seq_len, kv_seq_len]
            bias_tensor = GetAttentionBiasFromAlibiSlopes(batch_size,
                                                          q_seq_len,
                                                          kv_seq_len,
                                                          q_num_heads,
                                                          bias_enum,
                                                          bias_rank_info,
                                                          window_size,
                                                          alibi_slopes_tensor,
                                                          query_padding_mask,
                                                          key_padding_mask,
                                                          key_leftpad);
        }
        return std::make_tuple(bias_tensor, alibi_slopes_tensor);
    }
    else {
        throw std::runtime_error("Unsupported bias type");
    }
}
