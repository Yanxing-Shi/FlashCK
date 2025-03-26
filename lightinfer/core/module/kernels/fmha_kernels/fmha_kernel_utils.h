#pragma once

namespace lightinfer {

int num_splits_heuristic(int batch_nhead_mblocks, int num_SMs, int num_n_blocks, int max_splits)
{
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nhead_mblocks >= 0.8f * num_SMs) {
        return 1;
    }
    max_splits                        = std::min({max_splits, num_SMs, num_n_blocks});
    float              max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        }
        else {
            float n_waves = float(batch_nhead_mblocks * num_splits) / num_SMs;
            float eff     = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) {
                max_efficiency = eff;
            }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            continue;
        }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

inline int
override_num_splits_if_necessary(int batch, int nhead, int max_seqlen_q, int hdim_v, float p_drop, int num_splits)
{
    int  device;
    auto status = hipGetDevice(&device);
    if (status != hipSuccess) {
        return num_splits;
    }

    hipDeviceProp_t props{};
    status = hipGetDeviceProperties(&props, device);
    if (status != hipSuccess) {
        return num_splits;
    }

    // tile size should match the generate.py
    const int kM0 = 64;
    const int kN1 = hdim_v;

    const int num_m_blocks = ck_tile::integer_divide_ceil(max_seqlen_q, kM0);
    const int num_n_blocks = ck_tile::integer_divide_ceil(hdim_v, kN1);

    if (num_splits < 1 && p_drop == 0.0f) {
        return num_splits_heuristic(batch * nhead * num_m_blocks, props.multiProcessorCount * 2, num_n_blocks, 128);
    }

    return num_splits;
}

inline int get_num_splits(int kM0, int kN1, )
{
    hipDeviceProp_t props{};
    status = hipGetDeviceProperties(&props, device);
    if (status != hipSuccess) {
        return num_splits;
    }

    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
}
}  // namespace lightinfer