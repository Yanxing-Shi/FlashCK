#pragma once

#include <vector>

#include "flashck/core/utils/rocm_info.h"

#include "flashck/core/profiling/target.h"

namespace flashck {

inline std::vector<int64_t> GetSplitSearchSpace(const int64_t batch_size,
                                                const int64_t q_num_heads,
                                                const int64_t q_max_seqlen,
                                                const int64_t mtile_size)
{
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

    int batch_nhead_mblocks = batch_size * q_num_heads * ceildiv(q_max_seqlen, mtile_size);

    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &batch_nhead_mblocks](int num_splits) {
        return num_splits == 1
               || ceildiv(batch_nhead_mblocks, num_splits) != ceildiv(batch_nhead_mblocks, num_splits - 1);
    };

    std::vector<int> selected_gpu = Target::Instance()->GetTargetSelectedDevices();
    // VLOG(1) << "Selected devices for running: " << selected_gpu;

    int                  num_SMs = GetGPUMultiProcessors(selected_gpu[0]);
    std::vector<int64_t> search_space;
    for (int num_splits = 1; num_splits <= num_SMs; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            continue;
        }

        if (batch_nhead_mblocks * num_splits >= num_SMs)
            break;

        search_space.push_back(num_splits);
    }

    return search_space;
}

}  // namespace flashck