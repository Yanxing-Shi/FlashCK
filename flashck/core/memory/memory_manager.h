#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "flashck/core/memory/allocator.h"

namespace flashck {

/**
 * @struct TensorUsage
 * @brief Records tensor metadata for memory sharing optimization
 *
 * Tracks tensor's lifecycle through operation indices and memory footprint.
 * Used by MemoryManager to optimize memory reuse between tensors with
 * non-overlapping lifecycles.
 */
struct TensorUsage {
    std::string name_;       ///< Human-readable tensor identifier
    size_t      size_;       ///< Memory requirement in bytes
    int         unique_id_;  ///< System-wide unique tensor identifier
    int         first_idx_;  ///< First operation index using this tensor
    int         last_idx_;   ///< Last operation index using this tensor

    TensorUsage() = default;
    TensorUsage(std::string name, size_t size, int id, int start_idx, int end_idx):
        name_(std::move(name)), size_(size), unique_id_(id), first_idx_(start_idx), last_idx_(end_idx)
    {
    }
};

/**
 * @struct MemoryBlock
 * @brief Represents an allocated memory block with offset and size
 */
struct MemoryBlock {
    size_t offset_;    ///< Offset in the memory pool
    size_t size_;      ///< Size of the memory block
    int    end_time_;  ///< When this block becomes free

    MemoryBlock(size_t offset, size_t size, int end_time): offset_(offset), size_(size), end_time_(end_time) {}
};

/**
 * @class MemoryManager
 * @brief Manages shared tensor memory pool with lifecycle-based optimization
 *
 * Implements Algorithm 3: Greedy by Size from https://arxiv.org/abs/2001.03288
 * The algorithm sorts tensors by size (descending) and assigns each tensor
 * to the earliest available memory location that can accommodate it.
 */
class MemoryManager {
public:
    /**
     * @brief Constructor - initializes with default allocator
     */
    MemoryManager();

    /**
     * @brief Destructor - cleans up allocated memory
     */
    ~MemoryManager();

    /**
     * @brief Retrieves memory address for specified tensor
     * @param unique_id Unique identifier of the tensor
     * @return Pointer to allocated memory or nullptr if not found
     */
    char* GetMemory(int unique_id) const;

    /**
     * @brief Updates tensor lifecycle information
     * @param unique_id Unique tensor identifier
     * @param node_idx Operation node index
     * @param size Tensor size in bytes
     * @param name Debugging identifier
     */
    void UpdateTensorLifeIdx(int unique_id, int node_idx, size_t size, const std::string& name);

    /**
     * @brief Removes lifecycle tracking for specified tensor
     * @param unique_id Target tensor identifier
     * @return true if successfully removed, false if not found
     */
    bool RemoveLifeCycle(int unique_id);

    /**
     * @brief Calculates memory allocation using Greedy by Size algorithm
     * @throws std::runtime_error if allocation fails
     */
    void CalculateBuffer();

    /**
     * @brief Returns total allocated buffer size
     * @return Total buffer size in bytes
     */
    size_t GetTotalBufferSize() const;

    /**
     * @brief Returns the allocator instance
     * @return Shared pointer to allocator
     */
    std::shared_ptr<Allocator> GetAllocator() const;

private:
    /**
     * @brief Implements Algorithm 3: Greedy by Size
     * @return Vector of tensor assignments with their memory offsets
     */
    std::vector<std::pair<TensorUsage, size_t>> GreedyBySizeAllocation();

    /**
     * @brief Finds the best memory offset for a tensor
     * @param size Required memory size
     * @param start_time When tensor becomes active
     * @param end_time When tensor becomes inactive
     * @return Best offset position
     */
    size_t FindBestOffset(size_t size, int start_time, int end_time);

    /**
     * @brief Allocates physical memory buffer
     * @param total_size Total required memory
     */
    void AllocatePhysicalMemory(size_t total_size);

    /**
     * @brief Assigns memory addresses to tensors
     * @param assignments Vector of tensor-offset pairs
     */
    void AssignMemoryAddresses(const std::vector<std::pair<TensorUsage, size_t>>& assignments);

    /**
     * @brief Validates memory allocations for conflicts
     * @param assignments Vector of tensor-offset pairs
     */
    void ValidateAllocations(const std::vector<std::pair<TensorUsage, size_t>>& assignments);

    /**
     * @brief Checks if two tensors' lifetimes overlap
     * @param t1 First tensor
     * @param t2 Second tensor
     * @return true if lifetimes overlap
     */
    bool HasTemporalOverlap(const TensorUsage& t1, const TensorUsage& t2) const;

    /**
     * @brief Aligns size to specified boundary
     * @param size Size to align
     * @param alignment Alignment boundary (must be power of 2)
     * @return Aligned size
     */
    size_t AlignSize(size_t size, size_t alignment = 128) const;

private:
    std::shared_ptr<Allocator> allocator_ptr_;      ///< Memory allocator
    std::map<int, TensorUsage> tensor_usages_map_;  ///< Tensor lifecycle map
    std::map<int, char*>       tensor_ptr_map_;     ///< Tensor address map
    std::vector<MemoryBlock>   active_blocks_;      ///< Currently active memory blocks

    char*  memory_pool_;        ///< Main memory pool
    size_t total_buffer_size_;  ///< Total allocated buffer size
    bool   is_allocated_;       ///< Whether memory has been allocated

    static constexpr size_t kDefaultAlignment = 128;  ///< Default memory alignment
};

}  // namespace flashck