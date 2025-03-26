#include "lightinfer/core/graph/tensor.h"

#include "lightinfer/core/utils/printf.h"

namespace lightinfer {

int Tensor::global_tensor_id = 0;

Tensor::Tensor(std::string name, DataType dtype, size_t max_shape_size):
    id_(global_tensor_id++),
    dtype_(dtype),
    max_shape_size_(max_shape_size),
    ctx_ptr_(Context::GetGlobalInstance().get())
{
    std::string prefix_name = ctx_ptr_->GetLastNode() ? (ctx_ptr_->GetLastNode()->GetName() + "_") : "";

    name_     = prefix_name + name;
    mem_type_ = max_shape_size_ > 0 ? MemoryType::Shared : MemoryType::Fixed;

    if (mem_type_ == MemoryType::Shared) {
        mem_ptr_ = ctx_ptr_->GetMemoryManagerPtr();
        if (ctx_ptr_->max_tensor_size_ < max_shape_size * SizeOf(dtype_)) {
            ctx_ptr_->max_tensor_size_ = max_shape_size * SizeOf(dtype_);
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
        // LOG(INFO) << "set_tensor for " << name_ << ", which is FixedMemory!";
        return;
    }
    else if (mem_type_ == MemoryType::Shared) {
        LOG(INFO) << "set_tensor for " << name_ << ", which is SharedMemory!";
        return;
    }
    else if (mem_type_ == MemoryType::Offset) {
        LOG(INFO) << "set_tensor for " << name_ << ", which is OffsetMemory!";
        return;
    }
}

void Tensor::SetShape(const Shape& shape)
{
    shape_ = shape;
}

void Tensor::SetOffset(const size_t offset, const Shape& shape)
{
    if (original_tensor_ == nullptr) {
        LOG(ERROR) << "Error! tensor " << name_ << " SetOffset without original tensor";
        exit(-1);
    }
    if (mem_type_ != MemoryType::Offset) {
        LOG(ERROR) << "Error! tensor " << name_ << " SetOffset without original tensor";
        exit(-1);
    }

    shape_  = shape;
    offset_ = offset;
}

char* Tensor::BuildTensor(bool is_open_interval)
{
    if (mem_type_ == MemoryType::Offset) {
        VLOG(1) << "build Offset tensor for " << name_ << ", id is " << id_ << ", data is "
                << static_cast<const void*>(data_);
        return original_tensor_->BuildTensor(is_open_interval) + offset_ * SizeOf(dtype_);
    }
    if (mem_type_ == MemoryType::Fixed) {
        if (!ctx_ptr_->IsBuilt() && data_ == nullptr) {
            VLOG(1) << "build fixed tensor for " << name_ << ", id is " << id_ << ", data is "
                    << static_cast<const void*>(ctx_ptr_->tmp_buff_);
            return ctx_ptr_->tmp_buff_;
        }
        VLOG(1) << "build fixed tensor for " << name_ << ", id is " << id_ << ", data is "
                << static_cast<const void*>(data_);
        return data_;
    }
    if (mem_type_ == MemoryType::Shared) {
        if (data_ == nullptr) {
            if (!ctx_ptr_->IsBuilt()) {
                UpdateLifeIdx(ctx_ptr_->GetNodeIdx() - is_open_interval);
                VLOG(1) << "build shared tensor for " << name_ << ", id is " << id_ << ", data is "
                        << static_cast<const void*>(ctx_ptr_->tmp_buff_);
                return ctx_ptr_->tmp_buff_;
            }
            VLOG(1) << "build shared tensor for " << name_ << ", id is " << id_ << ", data is "
                    << static_cast<const void*>(data_);
            data_ = mem_ptr_->GetMemory(id_);
        }
        return data_;
    }
    LOG(ERROR) << "Error! tensor " << name_ << " without mem_type_!";
    return nullptr;
}

void Tensor::UpdateLifeIdx(const int node_idx)
{
    if (mem_type_ == MemoryType::Fixed) {
        return;
    }
    mem_ptr_->UpdateTensorLifeIdx(id_, node_idx, max_shape_size_ * SizeOf(dtype_), name_);
}

void Tensor::RemoveLifeCycle()
{
    mem_type_ = MemoryType::Fixed;
    if (mem_ptr_) {
        mem_ptr_->RemoveLifeCycle(id_);
    }
}

void Tensor::ResetFixed()
{
    if (mem_type_ == MemoryType::Fixed) {
        return;
    }
    this->RemoveLifeCycle();
    mem_type_       = MemoryType::Fixed;
    max_shape_size_ = 0;
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

void Tensor::PrintTensor()
{
    ctx_ptr_->Synchronize();
    if (GetMaxElementSize() == 0) {
        VLOG(1) << "error occurred! this tensor is" << name_ << " with size 0";
        exit(-1);
    }
    else {
        VLOG(1) << "tensor shape is" << shape_.ToString();
        VLOG(1) << "tensor dtype is" << dtype_;
    }

    if (mem_type_ == MemoryType::Offset) {
        VLOG(1) << "tensor is offset tensor, offset is " << offset_;
    }
}

}  // namespace lightinfer