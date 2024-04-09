#pragma once

#include <memory>
#include <string>

#include "ater/core/graph/common.h"
#include "ater/core/graph/context.h"
#include "ater/core/graph/shape.h"
#include "ater/core/memory/memory_manager.h"
#include "ater/core/utils/dtype.h"

namespace ater {

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

    // Set the specific memory space address and max tensor shape for the tensor
    // object. After setting, the memory space type of the tensor object is
    // changed to FixedMemroy.
    void SetTensor(char* input_tensor);

    // Just only set tensor shape for tensor object.
    void SetShape(const Shape& shape);

    // Set a specific offset value for a tensor whose memory type is OffsetMemory.
    // Note that the `offset` value here represents the number of elements, not
    // bytes.
    void SetOffset(const size_t offset, const Shape& shape);

    // This method executes logic differently in different situations.
    //
    // Before the context is constructed, this method will check the life cycle of
    // the SharedMemory type tensor according to the global timestamp recorded by
    // the context. When is_open_interval is true, the lifetime is updated to
    // (timestamp - 1), otherwise updated to the latest timestamp. And returns an
    // invalid pointer for subsequent possible scheduling operations
    // After the context is constructed, this method will return the real pointer
    // address.
    char* BuildTensor(bool is_open_interval = false);

    // Update the lifetime of the tensor, where node_idx represents the timestamp.
    void UpdateLifeIdx(const int node_idx);

    // Remove tensor life cycle.
    void RemoveLifeCycle();

    // Remove the life cycle information registered by the tensor from the
    // MemoryManager, do not use shared memory.
    void ResetFixed();

    // Attribute
    std::vector<DDim> GetShape() const;

    int    GetMaxElementSize() const;
    size_t GetMaxSizeBytes() const;

    std::string GetMemoryLocationStr() const;
    // Tensor memory types are divided into three types: SharedMemory,
    //  FixedMemory, OffsetMemory.
    std::string GetMemoryTypeStr() const;
    DataType    GetDtype() const;
    size_t      GetMaxShapeSize() const;

    // unique id of the tensor.
    int GetUniqueId() const;

    // Use the corresponding data type to print the tensor according to the
    // cuda::DataType information. Print the head and tail of the tensor according
    // to the shape information. The size parameter is used to indicate the number
    // of elements to be printed separately.
    void PrintTensor();

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

}  // namespace ater