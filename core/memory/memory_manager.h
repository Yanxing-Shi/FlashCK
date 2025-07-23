#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "core/memory/allocator.h"

namespace flashck {

/**
 * @struct TensorUsage
 * @brief Tracks tensor lifecycle and memory requirements
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
 * @brief Represents an allocated memory block with timing information
 */
struct MemoryBlock {
    size_t offset_;    ///< Offset in the memory pool
    size_t size_;      ///< Size of the memory block
    int    end_time_;  ///< When this block becomes free

    MemoryBlock(size_t offset, size_t size, int end_time): offset_(offset), size_(size), end_time_(end_time) {}
};

/**
 * @class MemoryManager
 * @brief Manages shared tensor memory with lifecycle-based optimization
 *
 * Implements the "Greedy by Size" algorithm for optimal memory sharing.
 * Sorts tensors by size and assigns memory to minimize total usage by
 * reusing memory between tensors with non-overlapping lifecycles.
 */
class MemoryManager {
public:
    /// Constructor - initializes with default allocator
    MemoryManager();

    /// Destructor - cleans up allocated memory
    ~MemoryManager();

    /**
     * @brief Get memory address for specified tensor
     * @param unique_id Unique identifier of the tensor
     * @return Pointer to allocated memory or nullptr if not found
     */
    char* GetMemory(int unique_id) const;

    /**
     * @brief Update tensor lifecycle information
     * @param unique_id Unique tensor identifier
     * @param node_idx Operation node index
     * @param size Tensor size in bytes
     * @param name Debugging identifier
     */
    void UpdateTensorLifeIdx(int unique_id, int node_idx, size_t size, const std::string& name);

    /**
     * @brief Remove lifecycle tracking for specified tensor
     * @param unique_id Target tensor identifier
     * @return true if successfully removed, false if not found
     */
    bool RemoveLifeCycle(int unique_id);

    /**
     * @brief Calculate and allocate shared memory using Greedy by Size algorithm
     * @throws std::runtime_error if allocation fails
     */
    void CalculateBuffer();

    /**
     * @brief Get total allocated buffer size
     * @return Total buffer size in bytes
     */
    size_t GetTotalBufferSize() const;

    /**
     * @brief Get the allocator instance
     * @return Shared pointer to allocator
     */
    std::shared_ptr<Allocator> GetAllocator() const;

private:
    /// Implements Greedy by Size allocation algorithm
    std::vector<std::pair<TensorUsage, size_t>> GreedyBySizeAllocation();

    /// Finds optimal memory offset for a tensor
    size_t FindBestOffset(size_t size, int start_time, int end_time);

    /// Allocates physical memory buffer
    void AllocatePhysicalMemory(size_t total_size);

    /// Assigns memory addresses to tensors
    void AssignMemoryAddresses(const std::vector<std::pair<TensorUsage, size_t>>& assignments);

    /// Validates memory allocations for conflicts
    void ValidateAllocations(const std::vector<std::pair<TensorUsage, size_t>>& assignments);

    /// Checks if two tensors' lifetimes overlap
    bool HasTemporalOverlap(const TensorUsage& t1, const TensorUsage& t2) const;

    /// Aligns size to specified boundary
    size_t AlignSize(size_t size, size_t alignment = 128) const;

private:
    std::shared_ptr<Allocator> allocator_ptr_;      ///< Memory allocator
    std::map<int, TensorUsage> tensor_usages_map_;  ///< Tensor lifecycle tracking
    std::map<int, char*>       tensor_ptr_map_;     ///< Tensor address mapping
    std::vector<MemoryBlock>   active_blocks_;      ///< Currently active memory blocks

    char*  memory_pool_;        ///< Main memory pool
    size_t total_buffer_size_;  ///< Total allocated buffer size
    bool   is_allocated_;       ///< Whether memory has been allocated

    static constexpr size_t kDefaultAlignment = 128;  ///< Default memory alignment
};

}  // namespace flashck