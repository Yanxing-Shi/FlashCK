#include "ater/core/graph/node.h"

namespace ater {

Variable::Variable(std::string name, DataType dtype): Node(name, NodeType::Variable), max_size_(0), dtype_(dtype)
{
    value_.reset(new Tensor("value", dtype));
    type_ = VarType::FixedVar;
}

Variable::Variable(std::string name, size_t max_size, DataType dtype, VarType type):
    Node(name, NodeType::Variable), max_size_(max_size), dtype_(dtype), type_(type)
{
    value_.reset(new Tensor("value", dtype, max_size_));

    if (type_ == VarType::SharedVar) {
        return;
    }
    else if (type_ == VarType::FixedVar) {
        MallocMemory(max_size_);
    }
    else if (type_ == VarType::RegressiveVar) {
        return;
    }
    else {
        LOG(ERROR) << "Variable" << name_ << "useless type" << static_cast<int>(type_);
        exit(-1);
    }
}

Variable::Variable(std::string name, Variable* parent_var):
    Node(name, NodeType::Variable),
    max_size_(0),
    dtype_(parent_var->GetDtype()),
    type_(VarType::OffsetVar),
    parent_var_(parent_var)
{
    value_.reset(new Tensor("value", parent_var->value_));

    parent_var->AddDescendants(this);
}

void Variable::SetFixedMemory()
{
    if (type_ == VarType::OffsetVar) {
        ATER_THROW(Unavailable("{} variable is offset var, can not set fixed memory", name_));
    }

    if (children_var_.size() && GetParentsNode().size() > 0) {
        return;
    }
    if (GetParentsNode().size() > 0 && GetChildrenNode().size() > 0) {
        ATER_THROW(Unavailable("{} node is not a IONode, can not set fixed memory", name_));
    }

    value_->ResetFixed();

    return;
}

void Variable::SwapTwoTensor(Variable* var_x, Variable* var_y)
{
    Tensor tmp_tensor      = *(var_x->value_.get());
    *(var_x->value_.get()) = *(var_y->value_.get());
    *(var_y->value_.get()) = tmp_tensor;
}

void Variable::SetValue(char* value_ptr)
{
    value_->ResetFixed();
    value_->SetTensor(value_ptr);
}

char* Variable::GetValue(bool is_open_interval)
{
    return value_->BuildTensor(is_open_interval);
}

void Variable::SetShape(const Shape& shape)
{
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
    return children_var_.size();
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
    children_var_.insert(var);
}

void Variable::SetOffset(const size_t offset, const Shape shape)
{
    shape_ = shape;
    value_->SetOffset(offset, shape);
}

void Variable::MallocMemory(const size_t size)
{
    size_t value_byte = size * SizeOf(dtype_);
    VLOG(1) << "Varaible" << GetName() << "malloc memory, value size: " << value_byte / (1024 * 1024) << "MB";

    type_           = VarType::FixedVar;
    char* value_ptr = context_ptr_->GetAllocator()->Malloc(value_byte);

    value_->RemoveLifeCycle();
    value_->SetTensor(value_ptr);
}

void Variable::UpdateRegressiveIdx()
{
    if (GetType() != VarType::RegressiveVar)
        return;

    value_->UpdateLifeIdx(context_ptr_->GetBeginRegressIdx());
    value_->UpdateLifeIdx(context_ptr_->GetEndRegressIdx());
}

void Variable::PrintVar()
{
    if (!context_ptr_->IsBuilt()) {
        return;
    }

    if (value_ == nullptr) {
        VLOG(1) << "node " << name_ << "not have value object.";
    }
    else if (GetValue() == nullptr) {
        VLOG(1) << "node " << name_ << "value address is nullptr.";
    }
    else {
        try {
            value_->PrintTensor();
        }
        catch (...) {
            ATER_THROW(Unavailable("{} variable print tensor value failed", name_));
        }
    }
}
}  // namespace ater