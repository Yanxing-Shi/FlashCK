#include "flashck/core/graph/tensor.h"

#include "flashck/core/utils/enforce.h"

namespace flashck {

int Tensor::global_tensor_id = 0;

Tensor::Tensor(std::string name, DataType dtype, size_t max_shape_size):
    id_(global_tensor_id++),
    dtype_(dtype),
    max_shape_size_(max_shape_size),
    ctx_ptr_(Context::GetGlobalInstance().get())
{
    // Generate tensor name with node prefix
    std::string prefix_name = ctx_ptr_->GetLastNodeName();
    if (!prefix_name.empty()) {
        prefix_name += "_";
    }
    name_ = prefix_name + name;

    // Determine memory type based on max_shape_size
    mem_type_ = max_shape_size_ > 0 ? MemoryType::Shared : MemoryType::Fixed;

    if (mem_type_ == MemoryType::Shared) {
        mem_ptr_ = ctx_ptr_->GetMemoryManagerPtr();
        // Update context max tensor size if needed
        size_t tensor_bytes = max_shape_size * SizeOf(dtype_);
        if (ctx_ptr_->max_tensor_size_ < tensor_bytes) {
            ctx_ptr_->max_tensor_size_ = tensor_bytes;
            ctx_ptr_->max_tensor_name_ = name_;
        }
    }
}

Tensor::Tensor(std::string name, std::shared_ptr<Tensor> original_tensor): Tensor(name, original_tensor->GetDtype())
{
    original_tensor_ = original_tensor;
    mem_type_        = MemoryType::Offset;
}

void Tensor::SetTensor(char* input_tensor)
{
    if (mem_type_ == MemoryType::Fixed) {
        data_ = input_tensor;
    }
    // SharedMemory and OffsetMemory don't need manual pointer setting
}

void Tensor::SetShape(const Shape& shape)
{
    shape_ = shape;
}

void Tensor::SetOffset(const size_t offset, const Shape& shape)
{
    FC_ENFORCE_EQ(mem_type_, MemoryType::Offset, InvalidArgument("SetOffset only valid for OffsetMemory tensors"));
    FC_ENFORCE_NE(original_tensor_, nullptr, InvalidArgument("OffsetMemory tensor missing original tensor"));

    shape_  = shape;
    offset_ = offset;
}

char* Tensor::BuildTensor(bool is_open_interval)
{
    switch (mem_type_) {
        case MemoryType::Offset:
            VLOG(1) << "Building OffsetMemory tensor: " << name_ << " (id=" << id_ << ")";
            return original_tensor_->BuildTensor(is_open_interval) + offset_ * SizeOf(dtype_);

        case MemoryType::Fixed:
            if (!ctx_ptr_->IsBuilt() && data_ == nullptr) {
                VLOG(1) << "Building FixedMemory tensor (temp): " << name_ << " (id=" << id_ << ")";
                return ctx_ptr_->tmp_buff_;
            }
            VLOG(1) << "Building FixedMemory tensor: " << name_ << " (id=" << id_ << ")";
            return data_;

        case MemoryType::Shared:
            if (data_ == nullptr) {
                if (!ctx_ptr_->IsBuilt()) {
                    UpdateLifeIdx(ctx_ptr_->GetNodeIdx() - is_open_interval);
                    VLOG(1) << "Building SharedMemory tensor (temp): " << name_ << " (id=" << id_ << ")";
                    return ctx_ptr_->tmp_buff_;
                }
                VLOG(1) << "Building SharedMemory tensor: " << name_ << " (id=" << id_ << ")";
                data_ = mem_ptr_->GetMemory(id_);
            }
            return data_;

        default:
            FC_THROW(InvalidArgument("Invalid memory type for tensor: {}", name_));
    }
}

void Tensor::UpdateLifeIdx(const int node_idx)
{
    if (mem_type_ != MemoryType::Fixed && mem_ptr_) {
        mem_ptr_->UpdateTensorLifeIdx(id_, node_idx, max_shape_size_ * SizeOf(dtype_), name_);
    }
}

void Tensor::RemoveLifeCycle()
{
    if (mem_ptr_) {
        mem_ptr_->RemoveLifeCycle(id_);
    }
    mem_type_ = MemoryType::Fixed;
}

void Tensor::ResetFixed()
{
    if (mem_type_ != MemoryType::Fixed) {
        RemoveLifeCycle();
        max_shape_size_ = 0;
    }
}

std::vector<DDim> Tensor::GetShape() const
{
    return shape_.ToVector();
}

size_t Tensor::GetMaxElementSize() const
{
    return std::get<1>(shape_.GetElementSizeTuple());
}

size_t Tensor::GetMaxSizeBytes() const
{
    return GetMaxElementSize() * SizeOf(dtype_);
}

std::string Tensor::GetMemoryLocationStr() const
{
    return std::string("Device");  // Simplified implementation
}

std::string Tensor::GetMemoryTypeStr() const
{
    switch (mem_type_) {
        case MemoryType::Fixed:
            return "FixedMemory";
        case MemoryType::Shared:
            return "SharedMemory";
        case MemoryType::Offset:
            return "OffsetMemory";
        default:
            return "Unknown";
    }
}

DataType Tensor::GetDtype() const
{
    return dtype_;
}

size_t Tensor::GetMaxShapeSize() const
{
    return max_shape_size_;
}

int Tensor::GetUniqueId() const
{
    return id_;
}

}  // namespace flashck