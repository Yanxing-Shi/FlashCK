
#include "flashck/core/memory/memory_manager.h"

#include "flashck/core/utils/enforce.h"
#include "flashck/core/utils/log.h"
#include "flashck/core/utils/printf.h"

namespace flashck {

MemoryManager::MemoryManager(): allocator_ptr_(std::make_shared<Allocator>()), total_buffer_size_(0) {}

char* MemoryManager::GetMemory(int unique_id) noexcept
{
    if (const auto it = tensor_ptr_map_.find(unique_id); it != tensor_ptr_map_.end()) {
        return it->second;
    }
    LOG(ERROR) << "GetMemory failed for invalid tensor id: " << unique_id;
    return nullptr;
}

void MemoryManager::UpdateTensorLifeIdx(int unique_id, int node_idx, size_t size, const std::string& name)
{
    // Parameter validation
    if (size == 0) {
        throw std::invalid_argument("Tensor size must be positive");
    }
    if (node_idx < 0) {
        throw std::invalid_argument("Node index cannot be negative");
    }

    // Insert or update lifecycle record
    auto [iter, inserted] =
        tensor_usages_map_.try_emplace(unique_id, TensorUsage{name, size, unique_id, node_idx, node_idx});

    // Update existing entry
    if (!inserted) {
        auto& usage      = iter->second;
        usage.first_idx_ = std::min(usage.first_idx_, node_idx);
        usage.last_idx_  = std::max(usage.last_idx_, node_idx);
        VLOG(3) << "Updated tensor " << unique_id << " lifecycle to [" << usage.first_idx_ << ", " << usage.last_idx_
                << "]";
    }
}

bool MemoryManager::RemoveLifeCycle(int unique_id) noexcept
{
    if (const auto count = tensor_usages_map_.erase(unique_id); count > 0) {
        VLOG(2) << "Removed lifecycle for tensor " << unique_id;
        return true;
    }
    LOG(WARNING) << "Failed to remove tensor " << unique_id << ": not found in usage map";
    return false;
}

/**
 * @brief Main entry point for buffer calculation algorithm
 *
 * @throws ResourceExhausted If system memory allocation fails
 * @throws LogicError If memory conflicts detected during validation
 *
 * @par Algorithm workflow:
 * 1. Prepare tensor usage data from storage
 * 2. Sort tensors by descending size
 * 3. Perform core allocation with temporal conflict checks
 * 4. Allocate physical memory blocks
 * 5. Assign addresses and validate spatial conflicts
 */
void MemoryManager::CalculateBuffer()
{
    VLOG(1) << "Starting buffer calculation on AMD CDNA3 architecture";

    const auto tensor_usages_vec = PrepareTensorUsageVector();
    SortTensorUsagesBySize(tensor_usages_vec);

    const auto [ordered_usages, total_size] = AllocateMemoryGreedy(tensor_usages_vec);

    const auto buffer_segments = AllocateMemorySegments(total_size);

    AssignMemoryAddresses(ordered_usages, buffer_segments);
    ValidateMemoryAllocations(ordered_usages);

    total_buffer_size_ = total_size;
    VLOG(1) << "Buffer calculation completed. Total allocated: " << HumanReadableSize(total_size);
}

/**
 * @brief Converts internal tensor usage map to processing vector
 * @return Vector of tensor usage records with initialized offsets
 *
 * @details
 * - Maintains original insertion order for equal-sized tensors
 * - Generates preprocessing log entries at VLOG level 3
 */
std::vector<std::pair<TensorUsage, size_t>> MemoryManager::PrepareTensorUsageVector() const
{
    std::vector<std::pair<TensorUsage, size_t>> vec;
    vec.reserve(tensor_usages_map_.size());

    for (const auto& [id, usage] : tensor_usages_map_) {
        VLOG(3) << "Preprocessing tensor " << id << " (size: " << HumanReadableSize(usage.size) << ")";
        vec.emplace_back(usage, 0);
    }
    return vec;
}

/**
 * @brief Sorts tensors in descending size order
 * @param[in,out] usages Tensor usage vector to sort
 *
 * @details
 * Sorting criteria:
 * - Primary key: Tensor size (descending)
 * - Secondary key: Last access time (ascending)
 */
void MemoryManager::SortTensorUsagesBySize(std::vector<std::pair<TensorUsage, size_t>>& usages) const
{
    std::sort(usages.begin(), usages.end(), [](const auto& a, const auto& b) {
        if (a.first.size != b.first.size) {
            return a.first.size > b.first.size;
        }
        return a.first.last_idx < b.first.last_idx;
    });
    VLOG(2) << "Sorted " << usages.size() << " tensors by descending size";
}

/**
 * @brief Core greedy allocation algorithm implementation
 * @param usages Sorted tensor usage records
 * @return Allocation results containing ordered placements and total size
 *
 * @details Features:
 * - 128-byte alignment for AMD MI350 memory subsystem
 * - Temporal conflict detection using interval tree
 * - Progressive memory consumption tracking
 */
std::pair<std::vector<std::pair<TensorUsage, size_t>>, size_t>
MemoryManager::AllocateMemoryGreedy(const std::vector<std::pair<TensorUsage, size_t>>& usages)
{
    std::vector<std::pair<TensorUsage, size_t>> ordered;
    ordered.reserve(usages.size());
    TemporalConflictDetector conflict_detector;
    size_t                   total_consumption = 0;

    // MI350-specific memory alignment
    constexpr size_t kAlignment = 128;  // CDNA3 optimal alignment
    static_assert((kAlignment & (kAlignment - 1)) == 0, "Alignment must be power of two");

    for (const auto& [usage, _] : usages) {
        const auto conflicts = conflict_detector.FindConflicts(usage.first_idx, usage.last_idx);

        size_t best_offset = 0;
        for (const auto& [_, end] : conflicts) {
            best_offset = std::max(best_offset, end);
        }

        // Apply 128-byte alignment calculation
        best_offset = (best_offset + kAlignment - 1) & ~(kAlignment - 1);  // NOLINT(whitespace/operators)

        ordered.emplace_back(usage, best_offset);
        conflict_detector.AddInterval(usage.first_idx, usage.last_idx, best_offset + usage.size);
        total_consumption = std::max(total_consumption, best_offset + usage.size);

        VLOG(3) << "Allocated " << HumanReadableSize(usage.size) << " at offset " << HumanReadableSize(best_offset)
                << " for tensor " << usage.unique_id;
    }

    return {ordered, total_consumption};
}

/**
 * @brief Allocates physical memory segments using RAII pattern
 * @param total_size Total required memory in bytes
 * @return Vector of managed memory buffers
 *
 * @throws ResourceExhausted When allocation exceeds available system memory
 */
std::vector<std::unique_ptr<char[], AllocDeleter>> MemoryManager::AllocateMemorySegments(size_t total_size)
{
    std::vector<std::unique_ptr<char[], AllocDeleter>> buffers;

    try {
        auto buffer = allocator_ptr_->Malloc(total_size);
        buffers.emplace_back(buffer, AllocDeleter{allocator_ptr_});
        VLOG(2) << "Allocated main buffer: " << HumanReadableSize(total_size);
    }
    catch (const std::bad_alloc& e) {
        LI_THROW(ResourceExhausted("Failed to allocate main buffer of size {}", HumanReadableSize(total_size)));
    }

    return buffers;
}

/**
 * @brief Assigns memory addresses to tensors
 * @param usages Ordered allocation records
 * @param buffers Allocated memory segments
 *
 * @throws LogicError If tensor exceeds buffer boundary
 */
void MemoryManager::AssignMemoryAddresses(const std::vector<std::pair<TensorUsage, size_t>>&        usages,
                                          const std::vector<std::unique_ptr<char[], AllocDeleter>>& buffers)
{
    tensor_ptr_map_.clear();
    const auto& main_buffer = buffers.front().get();

    for (const auto& [usage, offset] : usages) {
        if (offset + usage.size > total_buffer_size_) {
            LI_THROW(LogicError("Memory overflow detected for tensor {}", usage.unique_id));
        }
        tensor_ptr_map_[usage.unique_id] = main_buffer + offset;
    }
}

/**
 * @brief Validates spatial memory allocations
 * @param usages Ordered allocation records
 *
 * @throws LogicError If overlapping memory regions detected
 */
void MemoryManager::ValidateMemoryAllocations(const std::vector<std::pair<TensorUsage, size_t>>& usages) const
{
    SpatialConflictDetector spatial_detector;
    for (const auto& [usage, offset] : usages) {
        spatial_detector.AddInterval(offset, offset + usage.size, usage.unique_id);
    }

    for (const auto& [usage, offset] : usages) {
        const auto end       = offset + usage.size;
        const auto conflicts = spatial_detector.FindConflicts(offset, end);

        for (const auto& [conflict_id, conflict_start, conflict_end] : conflicts) {
            if (conflict_id != usage.unique_id) {
                LI_THROW(LogicError("Memory conflict between tensor {} ({}-{}) and {} ({}-{})",
                                    usage.unique_id,
                                    offset,
                                    end,
                                    conflict_id,
                                    conflict_start,
                                    conflict_end));
            }
        }
    }
}

std::shared_ptr<Allocator> MemoryManager::GetAllocator() const
{
    return allocator_ptr_;
}

}  // namespace flashck