#pragma once

#include <string>

static const std::string g_fmha_dtype_config_utils_source = R"(

template <typename DataType>
struct FmhaFwdTypeConfig;

template <>
struct FmhaFwdTypeConfig<ck_tile::half_t>
{
    using QDataType             = ck_tile::half_t;
    using KDataType             = ck_tile::half_t;
    using VDataType             = ck_tile::half_t;
    using BiasDataType          = ck_tile::half_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::half_t; // data type for A matrix of second gemm
    using OaccDataType          = float;           // data type for second gemm accumulation
    using ODataType             = ck_tile::half_t;
};

template <>
struct FmhaFwdTypeConfig<ck_tile::bf16_t>
{
    using QDataType             = ck_tile::bf16_t;
    using KDataType             = ck_tile::bf16_t;
    using VDataType             = ck_tile::bf16_t;
    using BiasDataType          = ck_tile::bf16_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::bf16_t; // data type for A matrix of second gemm
    using OaccDataType          = float;           // data type for second gemm accumulation
    using ODataType             = ck_tile::bf16_t;
};

template <>
struct FmhaFwdTypeConfig<ck_tile::fp8_t>
{
    using QDataType             = ck_tile::fp8_t;
    using KDataType             = ck_tile::fp8_t;
    using VDataType             = ck_tile::fp8_t;
    using BiasDataType          = float;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::fp8_t; // data type for A matrix of second gemm
    using OaccDataType          = float;          // data type for second gemm accumulation
    using ODataType             = ck_tile::fp8_t;
};

template <>
struct FmhaFwdTypeConfig<ck_tile::bf8_t>
{
    using QDataType             = ck_tile::bf8_t;
    using KDataType             = ck_tile::bf8_t;
    using VDataType             = ck_tile::bf8_t;
    using BiasDataType          = ck_tile::bf8_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::bf8_t; // data type for A matrix of second gemm
    using OaccDataType          = float;          // data type for second gemm accumulation
    using ODataType             = ck_tile::bf8_t;
};

)";

static const std::string g_fmha_rotary_utils_source = R"(
template <typename DataType>
std::tuple<ck_tile::HostTensor<DataType>, ck_tile::HostTensor<DataType>>
generate_rotary_cos_sin(ck_tile::index_t seqlen,
                        ck_tile::index_t rotary_dim,
                        std::optional<unsigned> seed = std::nullopt)
{
    // return dummy tensors if we won't apply RoPE at all
    if(rotary_dim <= 0)
    {
        ck_tile::HostTensor<DataType> dummy({1, 1});
        return std::make_tuple(dummy, dummy);
    }

    std::mt19937 random_engine(seed.has_value() ? *seed : std::random_device{}());
    std::uniform_real_distribution<float> generator(0.0f, 1.0f);

    const ck_tile::index_t num_rows = seqlen * 2;
    const ck_tile::index_t num_cols = rotary_dim / 2;

    using std::begin, std::end;

    ck_tile::HostTensor<float> angle({num_rows, num_cols});
    std::generate(begin(angle), end(angle), [&] { return generator(random_engine) * 2 * M_PI; });

    ck_tile::HostTensor<DataType> cos({num_rows, num_cols});
    std::transform(begin(angle), end(angle), begin(cos), [](float origin_value) {
        return ck_tile::type_convert<DataType>(std::cos(origin_value));
    });

    ck_tile::HostTensor<DataType> sin({num_rows, num_cols});
    std::transform(begin(angle), end(angle), begin(sin), [](float origin_value) {
        return ck_tile::type_convert<DataType>(std::sin(origin_value));
    });

    return std::make_tuple(cos, sin);
}

)";

static const std::string g_fmha_seq_utils_source = R"(
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ck_tile/core/container/span.hpp"

enum class mode_enum
{
    batch = 0,
    group
};

std::ostream& operator<<(std::ostream& stream, mode_enum mode)
{
    return stream << (mode == mode_enum::batch ? "batch" : "group");
}

std::vector<int32_t> to_seqstarts(ck_tile::span<const int32_t> seqlens)
{
    std::vector<int32_t> seqstarts = {0};
    for(int32_t seqlen : seqlens)
    {
        seqstarts.push_back(seqstarts.back() + seqlen);
    }
    assert(seqstarts.size() == seqlens.size() + 1);
    return seqstarts;
}

std::vector<int32_t> generate_seqlens(mode_enum mode,
                                      unsigned count,
                                      int32_t seqlen_avg,
                                      int32_t seqlen_min = -1, // if not negative, clamp min
                                      int32_t seqlen_max = -1, // if not negative, clamp max
                                      std::optional<unsigned> seed = std::nullopt)
{
    assert(0 < count);

    seqlen_min = (0 < seqlen_min ? seqlen_min : 1);
    seqlen_max = (0 < seqlen_max ? seqlen_max : std::numeric_limits<int32_t>::max());
    assert(seqlen_min <= seqlen_max);

    std::vector<int32_t> seqlens(count, std::clamp(seqlen_avg, seqlen_min, seqlen_max));

    if(mode == mode_enum::group && 1 < count)
    {
        using size_type = std::vector<int32_t>::size_type;

        std::mt19937 random_engine(seed.has_value() ? *seed : std::random_device{}());
        std::uniform_int_distribution<size_type> idx_dist(0, count - 1);
        auto next_idx = std::bind(idx_dist, std::ref(random_engine));

        std::uniform_int_distribution<size_type> step_dist(1, count - 1);
        auto next_step = std::bind(step_dist, std::ref(random_engine));

        for(unsigned repeat = seqlen_avg * (count / 2); 0 < repeat; --repeat)
        {
            const size_type to_decrease = next_idx();
            // make sure each elements of seqlens is in range [seqlen_min, seqlen_max]
            if(seqlens[to_decrease] == seqlen_min)
            {
                continue;
            }

            const size_type to_increase = (to_decrease + next_step()) % count;

            if(seqlens[to_increase] >= seqlen_max)
            {
                continue;
            }

            --seqlens[to_decrease];
            ++seqlens[to_increase];
        }
    }

    return seqlens;
}

// return random integer generated uniformly in range [low, high]
template <typename Int = int>
auto randint(Int low, Int high, std::optional<unsigned> seed = std::nullopt)
    -> std::enable_if_t<std::is_integral_v<Int>, Int>
{
    std::mt19937 engine(seed.has_value() ? *seed : std::random_device{}());
    std::uniform_int_distribution<Int> dist(low, high);
    return dist(engine);
}

// return random integers generated uniformly in range [low, high]
template <typename Int, typename ForwardIterator>
auto randints(ForwardIterator first,
              ForwardIterator last,
              Int low,
              Int high,
              std::optional<unsigned> seed = std::nullopt)
    -> std::enable_if_t<std::is_integral_v<Int>>
{
    std::mt19937 engine(seed.has_value() ? *seed : std::random_device{}());
    std::uniform_int_distribution<Int> dist(low, high);

    std::generate(first, last, [&] { return dist(engine); });
}

/*
 * decode the seqlen string from cmdline
 * example (assume batch=3)
 *   q_val=1,2,3 k_val=4,5,6 -> OK
 *   q_val=1,2,3             -> OK, k same as q
 *   q_val=1,2               -> OK, q will rand remaining 1 element, k same as q
 *   q_val=1,2   k_val=4,5   -> OK, q/k will rand remaining 1 element
 *   q_val=1,2,3,4           -> OK, but ignore exceed one
 *
 *   q_val=1,2   k_val=4,5,6 -> not OK, k must have same splits with q
 *   q_val=1,2   k_val=4     -> not OK, k must have same splits with q
 */

std::tuple<std::vector<ck_tile::index_t>,
           std::vector<ck_tile::index_t>,
           std::vector<ck_tile::index_t>>
decode_seqlen(mode_enum mode,
              ck_tile::index_t batch,
              std::string q_val,
              std::string k_val,
              std::string k_pad_val,
              ck_tile::index_t seqlen_k_min = 0,
              bool need_append_kvcache      = false,
              std::optional<unsigned> seed  = std::nullopt)
{
#define _S2I_(str_) static_cast<ck_tile::index_t>(std::atoi((str_).c_str()))
    if(mode == mode_enum::batch)
    {
        ck_tile::index_t q = _S2I_(q_val);
        ck_tile::index_t k = _S2I_(k_val);

        auto s_q = std::vector<ck_tile::index_t>(batch, q);
        auto s_k = [&] {
            const ck_tile::index_t seqlen_k_max = (k < 0 ? q : k);
            std::vector<ck_tile::index_t> seqlen_ks(batch, seqlen_k_max);

            if(1 < batch && need_append_kvcache)
            {
                // to keep the original s_k value, we always use seqlen_k_max in first batch
                randints(std::next(seqlen_ks.begin()),
                         seqlen_ks.end(),
                         seqlen_k_min,
                         seqlen_k_max,
                         seed);
                return seqlen_ks;
            }

            return seqlen_ks;
        }();
        auto s_kpad = std::vector<ck_tile::index_t>(batch, -1); // TODO: batch not support k_padding

        // s_k should be greater than or equal to seqlen_k_min if provided
        if(s_k.back() < seqlen_k_min)
        {
            throw std::runtime_error("s_k should be greater than or equal to seqlen_k_min if provided");
        }

        return std::make_tuple(s_q, s_k, s_kpad);
    }
    else
        {
            ck_tile::index_t idx          = 0;
            std::string::size_type pos_q  = 0;
            std::string::size_type pos_k  = 0;
            std::string::size_type pos_kp = 0;
            std::vector<ck_tile::index_t> s_q;
            std::vector<ck_tile::index_t> s_k;
            std::vector<ck_tile::index_t> s_kpad;
            while(true)
            {
                auto found_q  = q_val.find(',', pos_q);
                auto found_k  = k_val.find(',', pos_k);
                auto found_kp = k_pad_val.find(',', pos_kp);

                ck_tile::index_t q = _S2I_(
                    q_val.substr(pos_q, found_q == std::string::npos ? found_q : found_q - pos_q));
                ck_tile::index_t k = _S2I_(
                    k_val.substr(pos_k, found_k == std::string::npos ? found_k : found_k - pos_k));
                ck_tile::index_t kp = _S2I_(k_pad_val.substr(
                    pos_kp, found_kp == std::string::npos ? found_kp : found_kp - pos_kp));

                s_q.push_back(q);
                s_k.push_back(k < 0 ? q : k);
                s_kpad.push_back(kp);

                // s_k should be greater than or equal to seqlen_k_min
                if(s_k.back() < seqlen_k_min)
                {
                    throw std::runtime_error("s_k should be greater than or equal to seqlen_k_min if provided");
                }

                idx++;
            if(found_q == std::string::npos || idx >= batch)
            {
                break;
            }
            pos_q  = found_q + 1;
            pos_k  = found_k == std::string::npos ? pos_k : found_k + 1;
            pos_kp = found_kp == std::string::npos ? pos_kp : found_kp + 1;
        }
        if(idx < batch)
        {
            auto rem_q = generate_seqlens(mode, batch - idx, s_q.back(), 1, s_kpad.back(), seed);
            auto rem_k =
                generate_seqlens(mode, batch - idx, s_k.back(), seqlen_k_min, s_kpad.back(), seed);

            s_q.insert(s_q.end(), rem_q.begin(), rem_q.end());
            s_k.insert(s_k.end(), rem_k.begin(), rem_k.end());
            s_kpad.insert(s_kpad.end(), batch - idx, s_kpad.back());
        }
        return std::make_tuple(s_q, s_k, s_kpad);
    }
#undef _S2I_
}

template <typename RandomAccessIterator, typename Int>
std::enable_if_t<std::is_integral_v<Int>> iota_shuffle(RandomAccessIterator first,
                                                       RandomAccessIterator last,
                                                       Int value,
                                                       std::optional<unsigned> seed = std::nullopt)
{
    std::iota(first, last, value);

    std::mt19937 engine(seed.has_value() ? *seed : std::random_device{}());
    std::shuffle(first, last, engine);
}

)";

static const std::string g_fmha_utils_source = R"(

#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <iterator>
#include <optional>
#include <random>
#include <tuple>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/core/container/span.hpp"

{{dtype_config_utils}}
{{rotary_utils}}
{{seq_utils}}

{{args_decl}}
)";
