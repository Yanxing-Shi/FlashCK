#pragma once

#include <memory>
#include <string>

#include "core/graph/common.h"
#include "core/graph/context.h"
#include "core/graph/shape.h"
#include "core/memory/memory_manager.h"

namespace flashck {

class Tensor {
public:
    // Applies to tensors using FixedMemory and SharedMemory memory types.
    // When the mx_shape parameter is empty, it means that the tensor uses the
    // FixedMemory memory type, and then manually set the specific pointer address
    // and tensor shape.
    Tensor(std::string name, DataType dtype, size_t max_shape_size = 0);

    // Applicable to tensors whose video memory type is OffsetMemory.
    // In this case the initialized tensor is a partial fragment of the original
    // tensor. Later, the offset value and real shape info will be set through
    // the set_offset function.
    Tensor(std::string name, std::shared_ptr<Tensor> original_tensor);

    virtual ~Tensor() = default;

    void SetTensor(char* input_tensor);

    void SetShape(const Shape& shape);

    void SetOffset(const size_t offset, const Shape& shape);

    char* BuildTensor(bool is_open_interval = false);

    void UpdateLifeIdx(const int node_idx);

    void RemoveLifeCycle();

    void ResetFixed();

    // Attribute
    std::vector<DDim> GetShape() const;

    size_t GetMaxElementSize() const;
    size_t GetMaxSizeBytes() const;

    std::string GetMemoryLocationStr() const;
    // Tensor memory types are divided into three types: SharedMemory,
    //  FixedMemory, OffsetMemory.
    std::string GetMemoryTypeStr() const;
    DataType    GetDtype() const;
    size_t      GetMaxShapeSize() const;

    // unique id of the tensor.
    int GetUniqueId() const;

    static int global_tensor_id;

private:
    int id_;

    std::string             name_;
    DataType                dtype_;
    std::shared_ptr<Tensor> original_tensor_;
    MemoryType              mem_type_;
    size_t                  offset_ = 0;

    char* data_ = nullptr;

    std::shared_ptr<MemoryManager> mem_ptr_ = nullptr;

    // If mx_shape is 0, then tensor's memory type is FixedMemory or OffsetMemory.
    size_t max_shape_size_;
    Shape  shape_;

    Context* ctx_ptr_;
};

}  // namespace flashck