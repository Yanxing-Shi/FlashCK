#include "core/memory/memory_manager.h"
#include "core/utils/common.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace flashck {

MemoryManager::MemoryManager():
    allocator_ptr_(std::make_shared<Allocator>()), memory_pool_(nullptr), total_buffer_size_(0), is_allocated_(false)
{
    VLOG(2) << "MemoryManager initialized";
}

MemoryManager::~MemoryManager()
{
    if (memory_pool_ != nullptr && is_allocated_) {
        allocator_ptr_->Free(memory_pool_, true);
        VLOG(2) << "MemoryManager cleaned up memory pool";
    }
}

char* MemoryManager::GetMemory(int unique_id) const
{
    auto it = tensor_ptr_map_.find(unique_id);
    if (it != tensor_ptr_map_.end()) {
        return it->second;
    }
    LOG(ERROR) << "Tensor " << unique_id << " not found in memory map";
    return nullptr;
}

void MemoryManager::UpdateTensorLifeIdx(int unique_id, int node_idx, size_t size, const std::string& name)
{
    // Validate input parameters
    if (size == 0) {
        throw std::invalid_argument("Tensor size must be positive");
    }
    if (node_idx < 0) {
        throw std::invalid_argument("Node index cannot be negative");
    }

    auto it = tensor_usages_map_.find(unique_id);
    if (it != tensor_usages_map_.end()) {
        // Update existing tensor usage
        auto& usage      = it->second;
        usage.first_idx_ = std::min(usage.first_idx_, node_idx);
        usage.last_idx_  = std::max(usage.last_idx_, node_idx);

        // Warn about size mismatches (should be consistent)
        if (usage.size_ != size) {
            LOG(WARNING) << "Tensor " << unique_id << " size mismatch: " << usage.size_ << " vs " << size;
            usage.size_ = std::max(usage.size_, size);  // Use larger size for safety
        }

        VLOG(3) << "Updated tensor " << unique_id << " lifecycle: [" << usage.first_idx_ << ", " << usage.last_idx_
                << "]";
    }
    else {
        // Create new tensor usage record
        tensor_usages_map_[unique_id] = TensorUsage(name, size, unique_id, node_idx, node_idx);
        VLOG(3) << "Created tensor " << unique_id << " with size " << size << " bytes";
    }
}

bool MemoryManager::RemoveLifeCycle(int unique_id)
{
    auto it = tensor_usages_map_.find(unique_id);
    if (it != tensor_usages_map_.end()) {
        tensor_usages_map_.erase(it);
        tensor_ptr_map_.erase(unique_id);
        VLOG(2) << "Removed lifecycle for tensor " << unique_id;
        return true;
    }

    // LOG(WARNING) << "Failed to remove tensor " << unique_id << ": not found";
    return false;
}

void MemoryManager::CalculateBuffer()
{
    if (tensor_usages_map_.empty()) {
        // LOG(WARNING) << "No tensors to allocate memory for";
        return;
    }

    VLOG(1) << "Starting memory allocation for " << tensor_usages_map_.size() << " tensors";

    // Step 1: Run Greedy by Size allocation algorithm
    auto assignments = GreedyBySizeAllocation();

    // Step 2: Calculate total memory requirement
    size_t total_size = 0;
    for (const auto& [usage, offset] : assignments) {
        total_size = std::max(total_size, offset + AlignSize(usage.size_));
    }

    // Step 3: Allocate physical memory pool
    AllocatePhysicalMemory(total_size);

    // Step 4: Map tensor IDs to memory addresses
    AssignMemoryAddresses(assignments);

    // Step 5: Validate allocations for correctness
    ValidateAllocations(assignments);

    VLOG(1) << "Memory allocation completed. Total size: " << total_size << " bytes (" << (total_size / 1024.0 / 1024.0)
            << " MB)";
}

std::vector<std::pair<TensorUsage, size_t>> MemoryManager::GreedyBySizeAllocation()
{
    // Create and sort tensors by size (descending order)
    std::vector<TensorUsage> sorted_tensors;
    sorted_tensors.reserve(tensor_usages_map_.size());

    for (const auto& [id, usage] : tensor_usages_map_) {
        sorted_tensors.push_back(usage);
    }

    // Sort by size (larger first), then by start time for deterministic behavior
    std::sort(sorted_tensors.begin(), sorted_tensors.end(), [](const TensorUsage& a, const TensorUsage& b) {
        if (a.size_ != b.size_) {
            return a.size_ > b.size_;  // Larger tensors first
        }
        return a.first_idx_ < b.first_idx_;  // Earlier tensors first for ties
    });

    VLOG(2) << "Sorted " << sorted_tensors.size() << " tensors by size for allocation";

    // Assign memory offsets using greedy algorithm
    std::vector<std::pair<TensorUsage, size_t>> assignments;
    assignments.reserve(sorted_tensors.size());
    active_blocks_.clear();

    for (const auto& tensor : sorted_tensors) {
        size_t aligned_size = AlignSize(tensor.size_);
        size_t offset       = FindBestOffset(aligned_size, tensor.first_idx_, tensor.last_idx_);

        assignments.emplace_back(tensor, offset);
        active_blocks_.emplace_back(offset, aligned_size, tensor.last_idx_);

        VLOG(3) << "Tensor " << tensor.unique_id_ << " (size: " << tensor.size_ << ") assigned to offset " << offset;
    }

    return assignments;
}

size_t MemoryManager::FindBestOffset(size_t size, int start_time, int end_time)
{
    // Remove blocks that have expired before this tensor starts
    active_blocks_.erase(
        std::remove_if(active_blocks_.begin(),
                       active_blocks_.end(),
                       [start_time](const MemoryBlock& block) { return block.end_time_ < start_time; }),
        active_blocks_.end());

    // Sort remaining blocks by offset for gap detection
    std::sort(active_blocks_.begin(), active_blocks_.end(), [](const MemoryBlock& a, const MemoryBlock& b) {
        return a.offset_ < b.offset_;
    });

    // Find the first available gap that can fit this tensor
    size_t current_offset = 0;
    for (const auto& block : active_blocks_) {
        if (current_offset + size <= block.offset_) {
            // Found a suitable gap between existing blocks
            return current_offset;
        }
        current_offset = std::max(current_offset, block.offset_ + block.size_);
    }

    // No suitable gap found, allocate at the end
    return current_offset;
}

void MemoryManager::AllocatePhysicalMemory(size_t total_size)
{
    // Free existing memory if already allocated
    if (is_allocated_ && memory_pool_ != nullptr) {
        allocator_ptr_->Free(memory_pool_, true);
        memory_pool_  = nullptr;
        is_allocated_ = false;
    }

    total_buffer_size_ = total_size;

    if (total_size > 0) {
        memory_pool_ = allocator_ptr_->Malloc(total_size, 0, true);
        if (memory_pool_ == nullptr) {
            throw std::runtime_error("Failed to allocate memory pool of size " + std::to_string(total_size));
        }
        is_allocated_ = true;
        VLOG(2) << "Allocated memory pool: " << total_size << " bytes (" << (total_size / 1024.0 / 1024.0) << " MB)";
    }
}

void MemoryManager::AssignMemoryAddresses(const std::vector<std::pair<TensorUsage, size_t>>& assignments)
{
    tensor_ptr_map_.clear();

    for (const auto& [usage, offset] : assignments) {
        if (memory_pool_ != nullptr) {
            tensor_ptr_map_[usage.unique_id_] = memory_pool_ + offset;
        }
    }

    VLOG(2) << "Assigned memory addresses to " << assignments.size() << " tensors";
}

void MemoryManager::ValidateAllocations(const std::vector<std::pair<TensorUsage, size_t>>& assignments)
{
    // Check for memory conflicts between overlapping tensor lifetimes
    for (size_t i = 0; i < assignments.size(); ++i) {
        const auto& [usage1, offset1] = assignments[i];
        size_t end1                   = offset1 + AlignSize(usage1.size_);

        for (size_t j = i + 1; j < assignments.size(); ++j) {
            const auto& [usage2, offset2] = assignments[j];
            size_t end2                   = offset2 + AlignSize(usage2.size_);

            // Check for spatial overlap in memory
            bool spatial_overlap = (offset1 < end2) && (offset2 < end1);

            if (spatial_overlap && HasTemporalOverlap(usage1, usage2)) {
                // Memory conflict detected - same memory used by overlapping tensors
                std::stringstream ss;
                ss << "Memory conflict: tensors " << usage1.unique_id_ << " and " << usage2.unique_id_
                   << " overlap in both space and time. Tensor " << usage1.unique_id_ << " uses [" << offset1 << ", "
                   << end1 << ") during [" << usage1.first_idx_ << ", " << usage1.last_idx_ << "], tensor "
                   << usage2.unique_id_ << " uses [" << offset2 << ", " << end2 << ") during [" << usage2.first_idx_
                   << ", " << usage2.last_idx_ << "]";
                throw std::runtime_error(ss.str());
            }
        }
    }

    VLOG(2) << "Memory allocation validation passed - no conflicts detected";
}

bool MemoryManager::HasTemporalOverlap(const TensorUsage& t1, const TensorUsage& t2) const
{
    // Two tensors overlap temporally if their lifetime intervals intersect
    return (t1.first_idx_ <= t2.last_idx_) && (t2.first_idx_ <= t1.last_idx_);
}

size_t MemoryManager::AlignSize(size_t size, size_t alignment) const
{
    // Align size to specified boundary (alignment must be power of 2)
    return (size + alignment - 1) & ~(alignment - 1);
}

size_t MemoryManager::GetTotalBufferSize() const
{
    return total_buffer_size_;
}

std::shared_ptr<Allocator> MemoryManager::GetAllocator() const
{
    return allocator_ptr_;
}

}  // namespace flashck
