#pragma once

#include <algorithm>

#include "flashck/core/profiling/library.h"
#include "flashck/core/utils/enforce.h"
#include "tests/utils/torch_utils.h"

// Arguments:
//         hidden_states: (batch, seqlen, ...)
//         attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
//         unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.
//     Return:
//         hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
//         indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
//         cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
//         max_seqlen_in_batch: int
//         seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
template<typename T>
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
UnpadInput(const torch::Tensor& hidden_states,               // batch tensor:[batch_size, seq_len, num_heads, head_dim]
           const torch::Tensor& attention_mask,              // [batch_size, seq_len]
           const std::optional<torch::Tensor>& unused_mask)  // [batch_size, seq_len]
{
    torch::Tensor all_masks        = unused_mask.has_value() ? attention_mask + unused_mask.value() : attention_mask;
    torch::Tensor seqlens_in_batch = all_masks.sum(-1).to(torch::kInt32);
    torch::Tensor used_seqlens_in_batch = attention_mask.sum(-1).to(torch::kInt32);
    torch::Tensor indices               = torch::nonzero(all_masks.flatten()).flatten();
    int64_t       max_seqlen_in_batch   = seqlens_in_batch.max().item<int64_t>();
    torch::Tensor cu_seqlens            = torch::cumsum(seqlens_in_batch, 0, torch::kInt32).pad({1, 0});

    torch::Tensor hidden_states_unpad =
        hidden_states.reshape({hidden_states.sizes()[0] * hidden_states.sizes()[1], -1})[indices];

    return std::make_tuple(hidden_states_unpad, indices, cu_seqlens, seqlens_in_batch, used_seqlens_in_batch);
}

// Arguments:
//         hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
//         indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
//         batch: int, batch size for the padded sequence.
//         seqlen: int, maximum sequence length for the padded sequence.
//     Return:
//         hidden_states: (batch, seqlen, ...)
template<typename T>
inline torch::Tensor PadInput(const torch::Tensor& hidden_states,
                              const torch::Tensor& indices,
                              const int64_t        batch_size,
                              const int64_t        seqlen)
{
    auto          dim    = hidden_states.sizes()[1];
    torch::Tensor output = GetZerosTorchTensor<T>({batch_size * seqlen, dim});
    output.index_put_({indices}, hidden_states);
    return output.reshape({batch_size, seqlen, dim});
}

template<typename T>
inline auto GenerateVarlenQKVTensor(
    const AttentionConfigs&             configs,
    const torch::Tensor&                q,  // [batch_size, q_seq_len, q_num_heads, qk_head_dim]
    const torch::Tensor&                k,  // [batch_size, kv_seq_len, kv_num_heads, qk_head_dim]
    const torch::Tensor&                v,  // [batch_size, kv_seq_len, kv_num_heads, v_head_dim]
    const std::optional<torch::Tensor>& query_padding_mask = std::nullopt,  // [batch_size, q_seq_len]
    const std::optional<torch::Tensor>& key_padding_mask   = std::nullopt   // [batch_size, kv_seq_len]
)
{
    torch::Tensor q_unpad, q_indices, q_cu_seqlens;
    int64_t       q_max_seq_len;

    std::function<torch::Tensor(torch::Tensor)> ouput_pad_fn;

    if (query_padding_mask.has_value()) {
        std::tie(q_unpad, q_indices, q_cu_seqlens, std::ignore) = UnpadInput(q, query_padding_mask.v());
        ouput_pad_fn                                            = [&](torch::Tensor output_unpad) {
            return PadInput(output_unpad, query_padding_mask.v(), q_indices, q_cu_seqlens);
        };
    }
    else {
        q_unpad       = BMHK2BMK(q);
        q_cu_seqlens  = GetArrangeTorchTensor<T>(0, (batch_size + 1) * q_seq_len, q_seq_len);
        q_max_seq_len = q_seq_len;
        ouput_pad_fn  = [&](torch::Tensor output_unpad) { return BMK2BMHK(output_unpad); };
    }

    torch::Tensor k_unpad, k_indices, k_cu_seqlens, v_unpad;
    int64_t       k_max_seq_len;

    if (key_padding_mask.has_value()) {
        std::tie(k_unpad, k_indices, k_cu_seqlens, std::ignore)  = UnpadInput(k, key_padding_mask.v());
        std::tie(v_unpad, std::ignore, std::ignore, std::ignore) = UnpadInput(v, key_padding_mask.v());
    }
    else {
        k_unpad       = BMHK2BMK(k);
        v_unpad       = BMHK2BMK(v);
        k_cu_seqlens  = GetArrangeTorchTensor<T>(0, (batch_size + 1) * kv_seq_len, kv_seq_len);
        k_max_seq_len = kv_seq_len;
    }

    if (configs.is_qkv_packed_) {
        if (!(query_padding_mask == key_padding_mask).all()) {
            FC_THROW(InvalidArgument("query_padding_mask and key_padding_mask should be the same"));
        }
        FC_ENFORCE_EQ(q_num_heads, kv_num_heads, Unimplemented("QKV packed Only support same num_heads"));

        torch::Tensor qkv = torch::stack({q, k, v}, 2);
        return std::make_tuple(qkv_unpad, q_cu_seqlens, q_max_seq_len, qkv, ouput_pad_fn);
    }
    else if (configs.is_kv_packed_) {
        torch::Tensor kv_unpad = torch::stack({k_unpad, v_unpad}, 1);
        torch::Tensor kv       = torch::stack({k, v}, 2);
        return std::make_tuple(
            q_unpad, kv_unpad, q_cu_seqlens, k_cu_seqlens, q_max_seq_len, k_max_seq_len, q, kv, ouput_pad_fn);
    }
    else {
        return std::make_tuple(q_unpad, k_unpad, v_unpad, q_cu_seqlens, k_cu_seqlens, q_max_seq_len, k_max_seq_len);
    }
}