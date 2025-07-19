#pragma once

#include <memory>
#include <string>

#include "flashck/core/graph/common.h"
#include "flashck/core/graph/context.h"
#include "flashck/core/graph/shape.h"
#include "flashck/core/memory/memory_manager.h"

namespace flashck {

/**
 * @class Tensor
 * @brief Multi-dimensional tensor with flexible memory management
 *
 * Supports three memory types:
 * - FixedMemory: Pre-allocated memory with fixed pointer
 * - SharedMemory: Dynamically managed shared memory pool
 * - OffsetMemory: Sub-tensor with offset from parent tensor
 */
class Tensor {
public:
    /// Constructor for FixedMemory (max_shape_size=0) or SharedMemory tensors
    Tensor(std::string name, DataType dtype, size_t max_shape_size = 0);

    /// Constructor for OffsetMemory tensor (sub-tensor of original)
    Tensor(std::string name, std::shared_ptr<Tensor> original_tensor);

    virtual ~Tensor() = default;

    // Memory management
    void  SetTensor(char* input_tensor);
    void  SetShape(const Shape& shape);
    void  SetOffset(const size_t offset, const Shape& shape);
    char* BuildTensor(bool is_open_interval = false);

    // Lifecycle management
    void UpdateLifeIdx(const int node_idx);
    void RemoveLifeCycle();
    void ResetFixed();

    // Getters
    std::vector<DDim> GetShape() const;
    size_t            GetMaxElementSize() const;
    size_t            GetMaxSizeBytes() const;
    std::string       GetMemoryLocationStr() const;
    std::string       GetMemoryTypeStr() const;
    DataType          GetDtype() const;
    size_t            GetMaxShapeSize() const;
    int               GetUniqueId() const;

    static int global_tensor_id;  ///< Global tensor ID counter

private:
    int                            id_;                 ///< Unique tensor identifier
    std::string                    name_;               ///< Tensor name
    DataType                       dtype_;              ///< Data type
    std::shared_ptr<Tensor>        original_tensor_;    ///< Parent tensor for OffsetMemory
    MemoryType                     mem_type_;           ///< Memory management type
    size_t                         offset_  = 0;        ///< Offset in parent tensor
    char*                          data_    = nullptr;  ///< Raw data pointer
    std::shared_ptr<MemoryManager> mem_ptr_ = nullptr;  ///< Memory manager
    size_t                         max_shape_size_;     ///< Maximum shape size (0 for Fixed/Offset)
    Shape                          shape_;              ///< Tensor shape
    Context*                       ctx_ptr_;            ///< Context pointer
};

}  // namespace flashck