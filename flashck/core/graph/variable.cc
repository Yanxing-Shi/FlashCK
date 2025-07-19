#include "flashck/core/graph/node.h"

#include "flashck/core/utils/enforce.h"

namespace flashck {

Variable::Variable(std::string name, DataType dtype): Node(name, NodeType::Variable), max_size_(0), dtype_(dtype)
{
    value_.reset(new Tensor("value", dtype));
    type_ = VarType::FixedVar;
}

Variable::Variable(std::string name, size_t max_size, DataType dtype, VarType type):
    Node(name, NodeType::Variable), max_size_(max_size), dtype_(dtype), type_(type)
{
    value_.reset(new Tensor("value", dtype, max_size_));
    VLOG(1) << "Created Variable " << name_ << " with size " << max_size_;

    switch (type_) {
        case VarType::SharedVar:
            break;  // No additional setup needed
        case VarType::FixedVar:
            MallocMemory(max_size_);
            break;
        default:
            FC_THROW(InvalidArgument("Invalid variable type: {}", static_cast<int>(type_)));
    }
}

Variable::Variable(std::string name, Variable* parent_var, bool is_first):
    Node(name, NodeType::Variable),
    max_size_(0),
    dtype_(parent_var->GetDtype()),
    type_(VarType::OffsetVar),
    parent_var_(parent_var),
    is_first_(is_first)
{
    value_.reset(new Tensor("value", parent_var->value_));
    parent_var->AddDescendants(this);
}

void Variable::SetFixedMemory()
{
    FC_ENFORCE_NE(type_, VarType::OffsetVar, Unavailable("Variable {} is offset var, cannot set fixed memory", name_));

    // Skip if has children and parents (intermediate node)
    if (children_var_.size() && GetParentsNode().size() > 0) {
        return;
    }

    // Only allow I/O nodes to set fixed memory
    FC_ENFORCE_EQ(GetParentsNode().size() == 0 || GetChildrenNode().size() == 0,
                  true,
                  Unavailable("Variable {} is not an I/O node, cannot set fixed memory", name_));

    value_->ResetFixed();
}

void Variable::SwapTwoTensor(Variable* var_x, Variable* var_y)
{
    // Swap underlying tensor data between two variables
    Tensor tmp_tensor      = *(var_x->value_.get());
    *(var_x->value_.get()) = *(var_y->value_.get());
    *(var_y->value_.get()) = tmp_tensor;
}

void Variable::SetValue(char* value_ptr)
{
    VLOG(1) << "Setting value for " << name_ << " at address " << static_cast<void*>(value_ptr);
    value_->ResetFixed();
    value_->SetTensor(value_ptr);
}

char* Variable::GetValue(bool is_open_interval)
{
    // Build and return tensor data pointer
    return value_->BuildTensor(is_open_interval);
}

void Variable::SetShape(const Shape& shape)
{
    // Update both variable and underlying tensor shape
    shape_ = shape;
    value_->SetShape(shape);
}

Shape Variable::GetShape() const
{
    return shape_;
}

VarType Variable::GetType() const
{
    return type_;
}

DataType Variable::GetDtype() const
{
    return dtype_;
}

bool Variable::IsAncestor() const
{
    // Check if this variable has child variables (offset variables)
    return children_var_.size() > 0;
}

const Variable* Variable::GetAncestor() const
{
    return parent_var_;
}

const std::unordered_set<Variable*> Variable::GetDescendants() const
{
    return children_var_;
}

void Variable::AddDescendants(Variable* var)
{
    // Add child variable to descendants set
    children_var_.insert(var);
}

void Variable::SetOffset(const size_t offset, const Shape shape)
{
    // Set offset and shape for offset variables
    shape_ = shape;
    value_->SetOffset(offset, shape);
}

void Variable::MallocMemory(const size_t size)
{
    // Allocate fixed memory for the variable
    size_t value_byte_size = size * SizeOf(dtype_);

    type_           = VarType::FixedVar;
    char* value_ptr = context_ptr_->GetAllocator()->Malloc(value_byte_size);

    value_->RemoveLifeCycle();
    value_->SetTensor(value_ptr);
}

void Variable::PrintVar()
{
    // Debug function to print variable information
    LOG(INFO) << "Variable: " << name_ << ", Type: " << static_cast<int>(type_)
              << ", DataType: " << static_cast<int>(dtype_) << ", Shape: " << shape_.ToString()
              << ", MaxSize: " << max_size_;
}

}  // namespace flashck