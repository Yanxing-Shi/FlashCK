#pragma once

#include "flashck/core/profiling/library.h"
#include "flashck/core/utils/enforce.h"
#include "tests/utils/torch_utils.h"

struct AttentionConfigs {
    int64_t batch_size_;
    int64_t q_seq_len_;
    int64_t kv_seq_len_;
    int64_t q_num_heads_;
    int64_t kv_num_heads_;
    int64_t qk_head_dim_;
    int64_t v_head_dim_;

    int64_t group_size_;  // gqa or mqa

    float scale_;

    flashck::BiasEnum bias_enum_;
    int64_t           bias_rank_info_;

    flashck::GenericAttentionMaskEnum mask_enum_;

    // causal: window_size_right == 0 and window_size_left < 0.
    // no mask: window_size_right == -1 and window_size_left == -1.
    // Local: window_size_right >= 0 or window_size_left >= 0.
    std::array<int64_t, 2> window_size_;
};

enum class PackedType {
    GenericPacked = 0,
    QKVPacked     = 1,
    KVPacked      = 2,
    NoPacked      = 3,
};

struct FuseMultiHeadAttentionConfigs: public AttentionConfigs {
    bool  is_pre_layer_norm_;
    float epsilon_;

    PackedType packed_type_;
};

struct DecoderAttentionConfigs: public AttentionConfigs {
    int64_t new_kv_seq_len_;

    bool    use_batch_cache_idx_;
    bool    has_leftpad_;
    int64_t paged_block_size_;

    float             rotary_fraction_;
    flashck::RopeEnum rope_enum_;

    int64_t num_splits_;
};