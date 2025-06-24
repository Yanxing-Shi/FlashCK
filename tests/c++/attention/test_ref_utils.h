#pragma once

#include "tests/attention/test_tensor_utils.h"
#include "tests/utils/torch_utils.h"

torch::Tensor RefAttentionBMHK(const torch::Tensor&                q,
                               const torch::Tensor&                k,
                               const torch::Tensor&                v,
                               const std::optional<torch::Tensor>& attn_bias = std::nullopt,
                               const std::optional<torch::Tensor>& mask      = std::nullopt,
                               const std::optional<float>          scale     = std::nullopt,
                               bool                                is_cast   = true);

inline torch::Tensor RefAttention(
    const torch::Tensor&                q,  // [batch_size, q_num_heads, q_seq_len, q_head_dim]
    const torch::Tensor&                k,  // [batch_size, kv_num_heads, kv_seq_len, kv_head_dim]
    const torch::Tensor&                v,  // [batch_size, kv_num_heads, kv_seq_len, v_head_dim]
    const std::optional<torch::Tensor>& attn_bias = std::nullopt,  // element-wise bias: [batch_size, q_num_heads,
                                                                   // q_seq_len, kv_seq_len] alibi bias: [batch_size,
                                                                   // q_num_heads, q_seq_len, kv_seq_len]
    const std::optional<torch::Tensor>& mask    = std::nullopt,    // [batch_size, q_seq_len, kv_seq_len]
    const std::optional<float>          scale   = std::nullopt,
    bool                                is_cast = true)
{
    if (q.dim() == 5) {
        auto GetAttnBiasGroup = [&](const int group_id) {
            if (group_id == 1) {
                return attn_bias.value();
            }
            return attn_bias.value().slice(1, group_id, group_id + 1);
        };

        std::vector<torch::Tensor> outputs;
        for (int group_id = 0; group_id < q.sizes()[2]; group_id++) {
            auto q_group = q.slice(2, group_id, group_id + 1);
            auto k_group = k.slice(2, group_id, group_id + 1);
            auto v_group = v.slice(2, group_id, group_id + 1);
            outputs.push_back(RefAttentionBMHK(q_group, k_group, v_group, GetAttnBiasGroup(group_id), mask));
        }

        return torch::stack(outputs, 2);
    }

    if (q.dim() == 4) {
        return RefAttentionBMHK(q, k, v, attn_bias, mask);
    }

    torch::Tensor q_in    = is_cast ? q.to(torch::kFloat32) : q;
    torch::Tensor k_in    = is_cast ? k.to(torch::kFloat32) : k;
    torch::Tensor v_in    = is_cast ? v.to(torch::kFloat32) : v;
    torch::Tensor bias_in = is_cast && attn_bias.has_value() && attn_bias.value().dim() == 4 ?
                                attn_bias.value().to(torch::kFloat32) :
                                attn_bias.value();

    // scale q
    float scale_ = scale.has_value() ? scale.value() : 1 / std::sqrt(static_cast<float>(q_in.sizes()[2]));

    // q*k
    torch::Tensor logits = torch::bmm(q_in, k_in.transpose(-2, -1)) * scale_;

    // local mask
    if (mask.has_value() && mask.value().dim() == 3) {
        logits = logits.masked_fill(mask.value(), -std::numeric_limits<float>::infinity());
    }

    // q*k + attn_bias
    if (attn_bias.has_value() && attn_bias.value().dim() == 4) {
        bias_in = bias_in.reshape({-1, bias_in.sizes()[2], bias_in.sizes()[3]});
        logits  = logits + bias_in;
    }

    // softmax(q*k + attn_bias)
    torch::Tensor weights = torch::nn::functional::softmax(logits, /*dim*/ -1);

    // q*k*v
    return torch::bmm(weights, v_in).to(torch::kFloat32);
}

inline torch::Tensor RefAttentionBMHK(const torch::Tensor&                q,
                                      const torch::Tensor&                k,
                                      const torch::Tensor&                v,
                                      const std::optional<torch::Tensor>& attn_bias,
                                      const std::optional<torch::Tensor>& mask,
                                      const std::optional<float>          scale,
                                      bool                                is_cast)
{
    if (q.dim() != 4) {
        throw std::runtime_error("Only support BMHK format");
    }

    return BMK2BMHK(RefAttention(BMHK2BMK(q), BMHK2BMK(k), BMHK2BMK(v), attn_bias, mask, scale, is_cast), q.sizes()[2]);
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> InProjectionPacked(const torch::Tensor& x,
                                                                                  const torch::Tensor& proj_weight,
                                                                                  const torch::Tensor  proj_bias,
                                                                                  const PackedType&    packed_type)
{
    auto embedding_dims = x.sizes()[0];

    if (packed_type == PackedType::QKVPacked) {
        // self-attention
        auto proj = torch::nn::functional::linear(x, proj_weight, proj_bias);
        proj      = proj.unflatten(-1, {3, embedding_dims}).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous();
        return std::make_tuple(proj[0], proj[1], proj[2]);
    }
    else if (packed_type == PackedType::KVPacked) {
        // encoder - decoder attention
        auto          q_kv_tensor = proj_weight.split({embedding_dims, 2 * embedding_dims});
        torch::Tensor w_q         = q_kv_tensor[0];
        torch::Tensor w_kv        = q_kv_tensor[1];

        torch::Tensor b_q, b_kv;
        if (proj_bias.has_value()) {
            split_b_qkv = proj_bias.split({embedding_dims, 2 * embedding_dims});
            b_q         = split_b_qkv[0];
            b_kv        = split_b_qkv[1];
        }

        auto q_proj  = torch::nn::functional::linear(q, w_q, b_q);
        auto kv_proj = torch::nn::functional::linear(k, w_kv, b_kv);

        kv_proj = kv_proj.unflatten(-1, {2, embedding_dims}).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous();

        return std::make_tuple(q_proj, kv_proj[0], kv_proj[1]);
    }
    else if (packed_type == PackedType::GenericPacked) {
        torch::Tensor w = proj_weight.chunk(3, 0);

        torch::Tensor b_q, b_k, b_v;
        if (proj_bias.has_value()) {
            auto chunk_b = proj_bias.chunk(3, 0);
            b_q          = chunk_b[0];
            b_k          = chunk_b[1];
            b_v          = chunk_b[2];
        }

        return std::make_tuple(torch::nn::functional::linear(q, w[0], b_q),
                               torch::nn::functional::linear(k, w[1], b_k),
                               torch::nn::functional::linear(v, w[2], b_v));
    }
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> InProjection(const torch::Tensor& q,
                                                                            const torch::Tensor& k,
                                                                            const torch::Tensor& v,
                                                                            const torch::Tensor& w_q,
                                                                            const torch::Tensor& w_k,
                                                                            const torch::Tensor& w_v,
                                                                            const torch::Tensor& b_q,
                                                                            const torch::Tensor& b_k,
                                                                            const torch::Tensor& b_v)
{
    return std::make_tuple(torch::nn::functional::linear(q, w_q, b_q),
                           torch::nn::functional::linear(k, w_k, b_k),
                           torch::nn::functional::linear(v, w_v, b_v));
}

inline torch::Tensor RefFuseMultiHeadAttention(const float                         epsilon,
                                               const PackedType&                   packed_type,
                                               const torch::Tensor&                x,
                                               const torch::Tensor&                gamma,
                                               const torch::Tensor&                beta,
                                               const torch::Tensor&                out_proj_weight,
                                               const std::optional<torch::Tensor>& out_proj_bias,
                                               const std::optional<torch::Tensor>& proj_weight   = std::nullopt,
                                               const std::optional<torch::Tensor>& proj_bias     = std::nullopt,
                                               const std::optional<torch::Tensor>& q_proj_weight = std::nullopt,
                                               const std::optional<torch::Tensor>& k_proj_weight = std::nullopt,
                                               const std::optional<torch::Tensor>& v_proj_weight = std::nullopt,
                                               const std::optional<torch::Tensor>& q_proj_bias   = std::nullopt,
                                               const std::optional<torch::Tensor>& k_proj_bias   = std::nullopt,
                                               const std::optional<torch::Tensor>& v_proj_bias   = std::nullopt,
                                               const std::optional<torch::Tensor>& attn_bias     = std::nullopt,
                                               const std::optional<torch::Tensor>& mask          = std::nullopt,
                                               const std::optional<float>          scale         = std::nullopt)
{
    // compute in-projection
    torch::Tensor q, k, v;
    std::tie(q, k, v) = packed_type != PackedType::NoPacked ?
                            InProjection(q_proj_weight.value(),
                                         k_proj_weight.value(),
                                         v_proj_weight.value(),
                                         q_proj_bias.value(),
                                         k_proj_bias.value(),
                                         v_proj_bias.value()) :
                            InProjectionPacked(x, proj_weight.value(), proj_bias.value(), packed_type);

    // attention
    auto attn_out = RefAttentionBMHK(q, k, v, attn_bias, mask, scale);

    // compute out
    attn_out = torch::nn::functional::linear(BMHK2BMK(attn_out), out_proj_weight, out_proj_bias);

    // layer_norm
    return torch::nn::functional::layer_norm(
               attn_out,
               torch::nn::functional::LayerNormFuncOptions({embedding_dims}).eps(epsilon).weight(gamma).bias(beta))
        .to(torch::kFloat32);
}
