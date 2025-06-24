#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "flashck/core/memory/allocator.h"

namespace flashck {

/**
 * @class TensorUsage
 * @brief Records tensor metadata for memory sharing optimization in MemoryManager.
 *
 * Tracks tensor's lifecycle through operation indices and memory footprint.
 * MemoryManager uses this information to optimize memory reuse between tensors
 * with non-overlapping lifecycles.
 *
 * @note All index values should be non-negative and maintain start_idx <= end_idx
 */
struct TensorUsage {

    /// @brief Human-readable tensor identifier (debugging purposes)
    std::string name_;

    /// @brief Memory requirement in bytes (should be >0)
    size_t size_;

    /// @brief System-wide unique tensor identifier (immutable)
    int unique_id_;

    /// @brief First operation index using this tensor (input/output)
    int first_idx_;

    /// @brief Last operation index using this tensor (input/output)
    int last_idx_;
};
/**
 * @class MemoryManager
 * @brief Manages shared tensor memory pool with lifecycle-based optimization
 * @details Implements memory allocation strategy using greedy algorithm described in:
 *          https://arxiv.org/abs/2001.03288 (Algorithm 3: Greedy by Size)
 */
class MemoryManager {
public:
    /**
     * @brief Constructs a MemoryManager with default allocator
     * @details Initializes memory allocator and member variables
     */
    MemoryManager();

    /**
     * @brief Retrieves memory address for specified tensor
     * @param[in] unique_id Unique identifier of the tensor
     * @return Pointer to allocated memory or nullptr
     * @retval nullptr Indicates invalid tensor ID or uninitialized buffer
     * @note Must be called after successful CalculateBuffer() execution
     */
    char* GetMemory(int unique_id) noexcept;

    /**
     * @brief Updates tensor lifecycle information
     * @param[in] unique_id Unique tensor identifier
     * @param[in] node_idx Operation node index (non-negative)
     * @param[in] size Tensor size in bytes (must be positive)
     * @param[in] name Debugging identifier
     * @throw std::invalid_argument For invalid parameters (size=0 or node_idx<0)
     */
    void UpdateTensorLifeIdx(int unique_id, int node_idx, size_t size, const std::string& name);

    /**
     * @brief Removes lifecycle tracking for specified tensor
     * @param[in] unique_id Target tensor identifier
     * @return Operation status
     * @retval true Successfully removed
     * @retval false Tensor not found
     */
    bool RemoveLifeCycle(int unique_id) noexcept;

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
}  // namespace flashck