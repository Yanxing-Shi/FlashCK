#pragma once

#include "tests/utils/torch_utils.h"

/*----------------------------------------q,k,v, tensor---------------------------------------------------*/
inline torch::Tensor BMHK2BMK(const torch::Tensor& tensor)
{
    return tensor.permute({0, 2, 1, 3})
        .contiguous()
        .view({tensor.sizes()[0] * tensor.sizes()[2], tensor.sizes()[1], tensor.sizes()[3]});
}

inline torch::Tensor BMK2BMHK(const torch::Tensor& tensor, const int64_t num_heads)
{
    return tensor.contiguous()
        .view({tensor.sizes()[0] / num_heads, num_heads, tensor.sizes()[1], tensor.sizes()[2]})
        .permute({0, 2, 1, 3})
        .contiguous();
}

template<typename T>
inline torch::Tensor CreateTensor(const int64_t      batch_size,
                                  const int64_t      seq_len,
                                  const int64_t      num_heads,
                                  const int64_t      head_dim,
                                  const std::string& format     = "BMHK",
                                  const int64_t      group_size = 1)
{
    torch::Tensor tensor;

    if (format == "BMK") {
        tensor = GetRandomTorchTensor<T>({batch_size * num_heads, seq_len, head_dim});
    }
    else if (format == "BMHK") {
        tensor = GetRandomTorchTensor<T>({batch_size, seq_len, num_heads, head_dim});
    }
    else if (format == "BMGHK") {
        tensor = GetRandomTorchTensor<T>({batch_size, seq_len, group_size, num_heads, head_dim});
    }
    else {
        throw std::runtime_error("Invalid format");
    }

    return tensor;
}

template<typename T>
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CreateBatchQKVTensor(const int64_t      batch_size,
                                                                                    const int64_t      q_seq_len,
                                                                                    const int64_t      kv_seq_len,
                                                                                    const int64_t      q_num_heads,
                                                                                    const int64_t      kv_num_heads,
                                                                                    const int64_t      qk_head_dim,
                                                                                    const int64_t      v_head_dim,
                                                                                    const std::string& format = "BMHK",
                                                                                    const int64_t      group_size = 1)
{
    auto q = CreateTensor<T>(batch_size, q_seq_len, q_num_heads, qk_head_dim, format, group_size);
    auto k = CreateTensor<T>(batch_size, kv_seq_len, kv_num_heads, qk_head_dim, format, group_size);
    auto v = CreateTensor<T>(batch_size, kv_seq_len, kv_num_heads, v_head_dim, format, group_size);

    return std::make_tuple(q, k, v);
}

template<typename T>
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t>
GenerateBlockKVCacheTensor(const int64_t batch_size,
                           const int64_t k_max_seqlen,
                           const int64_t kv_seq_len,
                           const int64_t kv_num_heads,
                           const int64_t qk_head_dim,
                           const int64_t v_head_dim,
                           const int64_t paged_block_size)
{
    auto ceildiv = [](int64_t a, int64_t b) { return (a + b - 1) / b; };

    const int64_t max_num_page_blocks = batch_size * std::max((int64_t)1, ceildiv(k_max_seqlen, paged_block_size));

    // [max_num_page_blocks, paged_block_size, kv_num_heads, qk_head_dim]
    torch::Tensor k_cache_paged = CreateTensor<T>(max_num_page_blocks, paged_block_size, kv_num_heads, qk_head_dim);

    // [max_num_page_blocks, paged_block_size, kv_num_heads, v_head_dim]
    torch::Tensor v_cache_paged = CreateTensor<T>(max_num_page_blocks, paged_block_size, kv_num_heads, v_head_dim);

    // [batch_size,  max_num_page_blocks / batch]
    torch::Tensor block_table = GetRandpermTorchTensor(max_num_page_blocks).reshape({batch_size, -1});

    // [batch_size, kv_seq_len, kv_num_heads, qk_head_dim]
    torch::Tensor k_cache = k_cache_paged.index_select(0, block_table.flatten())
                                .reshape({
                                    batch_size,
                                    paged_block_size * block_table.sizes()[1],
                                    kv_num_heads,
                                    qk_head_dim,
                                })
                                .index_select(1, GetArrangeTorchTensor<int64_t>(0, kv_seq_len, 1));

    // [batch_size, kv_seq_len, kv_num_heads, v_head_dim]
    torch::Tensor v_cache = v_cache_paged.index_select(0, block_table.flatten())
                                .reshape({
                                    batch_size,
                                    paged_block_size * block_table.sizes()[1],
                                    kv_num_heads,
                                    v_head_dim,
                                })
                                .index_select(1, GetArrangeTorchTensor<int64_t>(0, kv_seq_len, 1));

    return std::make_tuple(k_cache, v_cache, block_table, k_cache_paged, v_cache_paged, max_num_page_blocks);
}