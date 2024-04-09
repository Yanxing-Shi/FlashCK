
#include "ater/core/memory/memory_manager.h"

#include "ater/core/utils/enforce.h"
#include "ater/core/utils/log.h"
#include "ater/core/utils/printf.h"

namespace ater {

MemoryManager::MemoryManager(): allocator_ptr_(new Allocator()) {}

char* MemoryManager::GetMemory(const int unique_id)
{
    return tensor_ptr_map_.find(unique_id)->second;
}

void MemoryManager::UpdateTensorLifeIdx(const int          unique_id,
                                        const int          node_idx,
                                        const size_t       size,
                                        const std::string& name)
{
    if (size == 0) {
        return;
    }

    auto iter = tensor_usages_map_.find(unique_id);
    if (iter == tensor_usages_map_.end()) {
        tensor_usages_map_.emplace(unique_id, TensorUsage(name, size, unique_id, node_idx, node_idx));
        return;
    }

    iter->second.first_idx_ = std::min(iter->second.first_idx_, node_idx);

    iter->second.last_idx_ = std::max(iter->second.last_idx_, node_idx);

    return;
}

void MemoryManager::RemoveLifeCycle(const int unique_id)
{
    if (tensor_usages_map_.find(unique_id) != tensor_usages_map_.end()) {
        tensor_usages_map_.erase(unique_id);
    }
    else {
        LOG(WARNING) << "tensor usages map not find " << unique_id;
    }
}

void MemoryManager::CalculateBuffer()
{
    LOG(INFO) << "Execute MemoryManager calculate_buffer";

    tensor_ptr_map_.clear();

    // tensor_usages_vec means: <TensorUsage, offset>
    VLOG(1) << "tensor_usages_map_ size: " << tensor_usages_map_.size();
    std::vector<std::pair<TensorUsage, size_t>> tensor_usages_vec{};
    for (const auto& [_, tensor_usage] : tensor_usages_map_) {
        VLOG(1) << "idx: " << tensor_usage.unique_id_ << ", life cycle: [" << tensor_usage.first_idx_ << ","
                << tensor_usage.last_idx_ << "], name: " << tensor_usage.name_ << ", size: " << tensor_usage.size_;
        tensor_usages_vec.push_back(std::make_pair(tensor_usage, 0));
    }

    // sort tensor usage records in non-increasing order by their sizes
    std::sort(tensor_usages_vec.begin(),
              tensor_usages_vec.end(),
              [](const std::pair<TensorUsage, size_t>& a, const std::pair<TensorUsage, size_t>& b) -> bool {
                  return a.first.size_ > b.first.size_;
              });

    // Algorithm.3: Greedy by Size for Offset Calculation
    // arxiv url: https://arxiv.org/abs/2001.03288

    size_t                                      total_consumption = 0;
    std::vector<std::pair<TensorUsage, size_t>> ordered_tensor_usages{};

    for (int idx = 0; idx < tensor_usages_vec.size(); idx++) {
        size_t      prev_offset      = 0;
        size_t      best_offset      = 0;
        bool        best_offset_flag = false;
        size_t      smallest_gap     = SIZE_MAX;
        TensorUsage tensor_usage_id  = tensor_usages_vec[idx].first;

        // check already assigned tensors whose usage intervals
        // intersect with that of the current tensor
        for (const auto& allocated_tensor : ordered_tensor_usages) {
            TensorUsage allocated_tensor_usage = allocated_tensor.first;
            size_t      max_first_op     = std::max(tensor_usage_id.first_idx_, allocated_tensor_usage.first_idx_);
            size_t      min_last_op      = std::min(tensor_usage_id.last_idx_, allocated_tensor_usage.last_idx_);
            size_t      allocated_offset = allocated_tensor.second;

            // find the smallest gap in memory between them such that current
            // tensor fits into that gap , If such a gap is found,
            // the current tensor is allocated to this gap
            if (max_first_op <= min_last_op) {
                size_t gap = allocated_offset - prev_offset;
                if (allocated_offset > prev_offset && gap >= tensor_usage_id.size_
                    && gap < smallest_gap) {  // Note the subtraction handling for unsigned
                                              // types
                    smallest_gap     = gap;
                    best_offset      = prev_offset;
                    best_offset_flag = true;
                }
                prev_offset = std::max(prev_offset, allocated_offset + allocated_tensor_usage.size_);
            }
        }
        // allocate it after the rightmost tensor whose usage interval
        // intersect with that of the current tensor
        if (!best_offset_flag) {
            best_offset = prev_offset;
        }

        tensor_usages_vec[idx].second = best_offset;
        ordered_tensor_usages.push_back(tensor_usages_vec[idx]);

        std::sort(ordered_tensor_usages.begin(),
                  ordered_tensor_usages.end(),
                  [](const std::pair<TensorUsage, size_t>& x, const std::pair<TensorUsage, size_t>& y) -> bool {
                      return x.second < y.second;
                  });
        total_consumption = std::max(total_consumption, best_offset + tensor_usage_id.size_);
    }

    total_buffer_size_ = total_consumption;

    VLOG(1) << "shared buffer memory size " << HumanReadableSize(total_buffer_size_);

    try {
        for (const auto& iter : buffer_vec_) {
            allocator_ptr_->Free(iter);
        }
        buffer_vec_.clear();
    }
    catch (...) {
        ATER_THROW(ResourceExhausted("{}", "execute MemoryManager clear buffer failed!"));
    }

    // Furthermore, considering the phenomenon of memory fragmentation, directly
    // applying for a whole buffer may cause memory allocation failure. On the
    // premise of ensuring that the memory of each tensor is continuous, we open up
    // several small buffers to avoid the above phenomenon.
    size_t                                      max_last_addr    = 0;
    size_t                                      record_last_addr = 0;
    std::vector<std::pair<TensorUsage, size_t>> temp_usages_vec{};
    int                                         buffer_idx = 0;
    for (int i = 0; i < ordered_tensor_usages.size(); i++) {
        max_last_addr =
            std::max(max_last_addr, (size_t)(ordered_tensor_usages[i].first.size_ + ordered_tensor_usages[i].second));

        temp_usages_vec.push_back(ordered_tensor_usages[i]);
        if ((i + 1 == ordered_tensor_usages.size()) || (max_last_addr == ordered_tensor_usages[i + 1].second)) {
            VLOG(1) << "Buffer Idx: " << buffer_idx << ", "
                    << "buffer memory" << HumanReadableSize(max_last_addr - record_last_addr);

            char* current_buffer = nullptr;
            try {
                current_buffer = allocator_ptr_->Malloc(max_last_addr - record_last_addr);
            }
            catch (...) {
                ATER_THROW(ResourceExhausted("allocate shared buffer failed!, buffer size is {}",
                                             buffer_vec_.size(),
                                             HumanReadableSize(max_last_addr - record_last_addr)));
            }

            LOG(INFO) << "MemoryManager allocate success!";
            buffer_vec_.push_back(current_buffer);
            buffer_size_vec_.push_back(max_last_addr - record_last_addr);
            buffer_idx++;

            for (const auto& iter : temp_usages_vec) {
                int unique_id = iter.first.unique_id_;
                tensor_ptr_map_.emplace(unique_id, current_buffer + iter.second - record_last_addr);
            }
            temp_usages_vec.clear();
            record_last_addr = max_last_addr;
        }
    }

    // Add algorithm check module
    // return true means check success
    auto judge_func = [](const std::pair<TensorUsage, size_t>& x, const std::pair<TensorUsage, size_t>& y) {
        auto max_time_l = std::max(x.first.first_idx_, y.first.first_idx_);
        auto min_time_r = std::min(x.first.last_idx_, y.first.last_idx_);
        if (min_time_r < max_time_l) {
            return true;
        }
        auto max_space_l = std::max(x.second, y.second);
        auto min_space_r = std::min(x.first.size_ + x.second, y.first.size_ + y.second);
        if (min_space_r <= max_space_l) {
            return true;
        }
        return false;
    };

    temp_usages_vec.clear();

    // print order
    std::sort(tensor_usages_vec.begin(),
              tensor_usages_vec.end(),
              [](const std::pair<TensorUsage, size_t>& x, const std::pair<TensorUsage, size_t>& y) -> bool {
                  // return x.first.first_idx < y.first.first_idx;
                  if (x.second != y.second)
                      return x.second < y.second;
                  if (x.second + x.first.size_ != y.second + y.first.size_)
                      return x.second + x.first.size_ > y.second + y.first.size_;
                  return x.first.first_idx_ < y.first.first_idx_;
              });

    for (const auto& iter : tensor_usages_vec) {
        int    unique_id = iter.first.unique_id_;
        size_t size      = iter.first.size_;
        // print tensor ptr map
        char* addr = tensor_ptr_map_.find(unique_id)->second;

        // VLOG(1) << "idx: " << unique_id << ", life cycle: [" << iter.first.first_idx_ << "," << iter.first.last_idx_
        //         << "], name: " << iter.first.name_ << ", memory size: " << HumanReadableSize(size)
        //         << ", end memory: " << HumanReadableSize(iter.second + size) << ", offset: " << iter.second
        //         << ", size: " << size << ", end_offset: " << iter.second + size;
    }

    for (const auto& iter : tensor_usages_vec) {
        for (const auto& check_iter : temp_usages_vec) {
            if (judge_func(check_iter, iter)) {
                continue;
            }

            int    unique_id = iter.first.unique_id_;
            size_t size      = iter.first.size_;

            LOG(ERROR) << "shhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh";
            // Logically, this part of the processing will never be executed. If it is
            // executed, it means that there is a bug in the shared memory scheduling
            // algorithm.

            // LOG(ERROR) << "idx: " << unique_id << ", life cycle: [" << iter.first.first_idx_ << ","
            //            << iter.first.last_idx_ << "], name: " << iter.first.name_ << ", size: " << size
            //            << ", offset: " << iter.second;

            // int    check_unique_id = check_iter.first.unique_id_;
            // size_t check_size      = check_iter.first.size_;
            // LOG(ERROR) << "idx:" << check_unique_id << ", life cycle:[" << check_iter.first.first_idx_ << ","
            //            << check_iter.first.last_idx_ << "], name:" << check_iter.first.name_ << ", size:" <<
            //            check_size
            //            << ", offset:" << check_iter.second;
        }

        temp_usages_vec.push_back(iter);
    }

    VLOG(1) << "Finish MemoryManager calculate_buffer";
}

size_t MemoryManager::GetTotalBufferSize() const
{
    return total_buffer_size_;
}

std::shared_ptr<Allocator> MemoryManager::GetAllocator() const
{
    return allocator_ptr_;
}

}  // namespace ater