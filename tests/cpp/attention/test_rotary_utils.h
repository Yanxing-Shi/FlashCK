#pragma once

#include <algorithm>

#include "flashck/core/profiling/library.h"
#include "flashck/core/utils/enforce.h"
#include "tests/utils/torch_utils.h"

template<typename T>
inline std::tuple<torch::Tensor, torch::Tensor> GenerateCosSinTensor(const int64_t seq_len, const int64_t rotary_dim)
{
    // [seq_len * 2, rotary_dim / 2]
    torch::Tensor angle = GetRandomTorchTensor<T>({seq_len * 2, rotary_dim / 2}) * 2 * M_PI;
    // [seq_len * 2, rotary_dim / 2]
    torch::Tensor cos = torch::cos(angle);
    // [seq_len * 2, rotary_dim / 2]
    torch::Tensor sin = torch::sin(angle);

    return std::make_tuple(cos, sin);
}

template<typename T>
inline std::tuple<torch::Tensor, torch::Tensor>
SliceRotaryCosSin(const torch::Tensor& cos,            // [seqlen * 2 , rotary_dim / 2]
                  const torch::Tensor& sin,            // [seqlen * 2, rotary_dim / 2]
                  const torch::Tensor& seqlen_offset,  // [batch_size,1]
                  const int64_t        seqlen)
{
    // [1, seqlen]
    torch::Tensor arrange = GetArrangeTorchTensor<int64_t>(0, seqlen, 1).reshape({1, -1});
    // [batch_size, seqlen]
    torch::Tensor idx = arrange + seqlen_offset;
    // [batch_size, seqlen * 2, rotary_dim / 2]
    torch::Tensor cos_ = cos.index_select(0, idx.flatten()).reshape({seqlen_offset.sizes()[0], -1, cos.sizes()[1]});
    // [batch_size, seqlen * 2, rotary_dim / 2]
    torch::Tensor sin_ = sin.index_select(0, idx.flatten()).reshape({seqlen_offset.sizes()[0], -1, sin.sizes()[1]});

    return std::make_tuple(cos_, sin_);
}

template<typename T>
inline torch::Tensor RotateHalf(const torch::Tensor& x,  // [batch_size, seqlen, num_heads, rotary_dim]
                                bool                 is_rotated = true)
{
    if (is_rotated) {
        auto x1 = x.chunk(2, -1)[0];       // [batch_size, seqlen, num_heads, rotary_dim / 2]
        auto x2 = x.chunk(2, -1)[1];       // [batch_size, seqlen, num_heads, rotary_dim / 2]
        return torch::cat({-x2, x1}, -1);  // [batch_size, seqlen, num_heads, rotary_dim]
    }
    else {
        auto x1 = x.index_select(
            -1, GetArrangeTorchTensor<int64_t>(0, x.sizes()[3], 2));  // [batch_size, seqlen, num_heads, rotary_dim / 2]
        auto x2 = x.index_select(
            -1, GetArrangeTorchTensor<int64_t>(1, x.sizes()[3], 2));  // [batch_size, seqlen, num_heads, rotary_dim / 2]
        return torch::stack({-x2, x1}, -1)
            .reshape(
                {-1, x.sizes()[1], x.sizes()[2] / 2, 2 * x.sizes()[3]});  // [batch_size, seqlen, num_heads, rotary_dim]
    }
}

template<typename T>
inline torch::Tensor ApplyRotaryEmbeddingTorch(const torch::Tensor& x,    // [batch_size, seqlen, num_heads, head_dim]
                                               const torch::Tensor& cos,  // [batch_size, seqlen, rotary_dim / 2]
                                               const torch::Tensor& sin,  // [batch_size, seqlen, rotary_dim / 2]
                                               bool                 is_rotated = true)
{
    auto roatry_dim = cos.sizes()[2] * 2;
    assert(roatry_dim <= x.sizes()[3]);
    // [batch_size, seqlen, 1, rotary_dim]
    torch::Tensor cos_ = is_rotated ? cos.repeat({1, 1, 2}).unsqueeze(-2) : cos.repeat_interleave(2, 0).unsqueeze(-2);

    // [batch_size, seqlen, 1, rotary_dim]
    torch::Tensor sin_ = is_rotated ? sin.repeat({1, 1, 2}).unsqueeze(-2) : sin.repeat_interleave(2, 0).unsqueeze(-2);

    // [batch_size, seqlen, num_heads, head_dim]
    return torch::cat(
        {x.index_select(-1, GetArrangeTorchTensor<int64_t>(0, roatry_dim)) * cos_
             + RotateHalf<T>(x.index_select(-1, GetArrangeTorchTensor<int64_t>(0, roatry_dim)), is_rotated) * sin_,
         x.index_select(-1, GetArrangeTorchTensor<int64_t>(roatry_dim, x.sizes()[3]))},
        -1);
}

template<typename T>
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> CreateRotaryEmbeddingTensor(
    const torch::Tensor&                q,  // [batch_size, q_seq_len, q_num_heads, qk_head_dim]
    const torch::Tensor&                k,  // [batch_size, new_kv_seq_len, kv_num_heads, qk_head_dim]
    const int64_t                       rotary_dim,
    bool                                is_rotated,
    const torch::Tensor&                cache_seqlen_k,                    // [batch_size, 1]
    const std::optional<torch::Tensor>& query_padding_mask = std::nullopt  // [batch_size, q_seq_len]
)
{
    torch::Tensor cos, sin;
    torch::Tensor q_rotated, k_rotated;
    if (rotary_dim > 0) {
        // [seq_len * 2 , rotary_dim / 2]
        std::tie(cos, sin) = GenerateCosSinTensor<T>(std::max(q.sizes()[1], k.sizes()[1]), rotary_dim);

        // slice cos and sin
        // [seq_len * 2, rotary_dim / 2]
        torch::Tensor cos_q, sin_q;
        std::tie(cos_q, sin_q) = SliceRotaryCosSin<T>(cos, sin, cache_seqlen_k, q.sizes()[1]);
        torch::Tensor cos_k, sin_k;
        std::tie(cos_k, sin_k) = SliceRotaryCosSin<T>(cos, sin, cache_seqlen_k, k.sizes()[1]);

        // [batch_size, q_seq_len, q_num_heads, qk_head_dim]
        q_rotated = ApplyRotaryEmbeddingTorch<T>(q, cos_q, sin_q, is_rotated);

        // [batch_size, new_kv_seq_len, kv_num_heads, qk_head_dim]
        k_rotated = ApplyRotaryEmbeddingTorch<T>(k, cos_k, sin_k, is_rotated);

        // [batch_size, q_seq_len, q_num_heads, qk_head_dim]
        if (query_padding_mask.has_value()) {
            q_rotated = q_rotated.masked_fill(query_padding_mask.value().reshape({-1, q.sizes()[1], 1, 1}),
                                              -std::numeric_limits<float>::infinity());
        }
    }
    else {
        q_rotated = q;
        k_rotated = k;
    }

    return std::make_tuple(sin, cos, q_rotated, k_rotated);
}
