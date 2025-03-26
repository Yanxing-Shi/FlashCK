#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "lightinfer/core/memory/allocator.h"

namespace lightinfer {

/*
  Class: TensorUsage
  Description:
    Records the tensor's unique_id, life cycle and size information. This
    information will be recorded in the MemoryManager for memory sharing
    allocation.
    first_idx: the index of the first op that uses this tensor as its input
                  or output
    last_op_idx: the index of the last op that uses this tensor as its input or
                 output
    size: the size of this tensor
*/

class TensorUsage {
public:
    TensorUsage(std::string name, size_t size, int id, int start_idx, int end_idx):
        name_(name), size_(size), unique_id_(id), first_idx_(start_idx), last_idx_(end_idx)
    {
    }

    std::string name_;
    size_t      size_;
    int         unique_id_;
    int         first_idx_;
    int         last_idx_;
};

/*
  Class: MemoryManager
  Description:
    MemoryManager manages all tensor memory available for sharing. MemoryManager
    performs memory allocation planning based on the information provided by
    TensorUsage. The basic idea is to perform greedy filling. For more details,
    please refer to: https://arxiv.org/abs/2001.03288 - Algorithm.3: Greedy by
    Size for Offset Calculation
*/

class MemoryManager {
public:
    MemoryManager();

    char* GetMemory(const int unique_id);

    void UpdateTensorLifeIdx(const int unique_id, const int node_idx, const size_t size, const std::string& name);

    void RemoveLifeCycle(const int unique_id);

    void CalculateBuffer();

    size_t GetTotalBufferSize() const;

    std::shared_ptr<Allocator> GetAllocator() const;

private:
    std::shared_ptr<Allocator> allocator_ptr_;

    std::map<int, TensorUsage> tensor_usages_map_;  // key:tensor_id, value:TensorUsage

    std::map<int, char*> tensor_ptr_map_;  // key:tensor_id, value: Tensor address

    std::vector<char*>  buffer_vec_;
    std::vector<size_t> buffer_size_vec_;

    size_t total_buffer_size_;
};
}  // namespace lightinfer