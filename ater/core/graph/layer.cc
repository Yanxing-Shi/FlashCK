#include "ater/core/graph/layer.h"

namespace ater {

Layer::Layer(std::string name): op_vec_({})
{
    context_ptr_ = Context::GetGlobalInstance();
    std::string layer_name =
        context_ptr_->GetLastLayer() ? (context_ptr_->GetLastLayer()->GetName() + "/" + name) : name;
    int idx = context_ptr_->layer_name_cnt[name];
    context_ptr_->layer_name_cnt[name] += 1;
    name_ = layer_name + "_" + std::to_string(idx);
    context_ptr_->EnterLayer(this, true);
}

std::string Layer::GetName() const
{
    return name_;
}

void Layer::Forward()
{
    context_ptr_->BuildContext();
    ClearFwdFlag();
    context_ptr_->UpdateNodeIdx();

    ForwardProcess();
    for (Variable* var : output_var_vec_) {
        if (var == nullptr)
            continue;
        VLOG(1) << "layer:" << GetName() << ",output: " << var->GetName();
        var->RecursiveForward();
    }
}

void Layer::SetInputs(const std::vector<Variable*>& input_var_vec)
{
    input_var_vec_ = input_var_vec;
    context_ptr_->EnterLayer(this, false);
}

void Layer::SetOutputs(const std::vector<Variable*>& output_var_vec)
{
    output_var_vec_ = output_var_vec;
    context_ptr_->ExitLayer();
}

Variable* Layer::GetInput(const int idx) const
{
    if (idx >= input_var_vec_.size()) {
        LOG(ERROR) << "layer " << GetName() << " input idx is out of range!";
        exit(0);
    }
    return input_var_vec_[idx];
}

Variable* Layer::GetOutput(const int idx) const
{
    if (idx >= output_var_vec_.size()) {
        LOG(ERROR) << "layer " << GetName() << " input idx is out of range!";
        exit(0);
    }
    return output_var_vec_[idx];
}

void Layer::ClearFwdFlag()
{
    for (Operation* op : op_vec_) {
        op->ClearFwdFlag();
        for (Node* var : op->GetChildrenNode()) {
            Variable* this_var = static_cast<Variable*>(var);
            for (Variable* iter : this_var->GetDescendants()) {
                iter->ClearFwdFlag();
            }
            var->ClearFwdFlag();
        }
    }
}

void Layer::TagFwdFlag()
{
    for (Operation* op : op_vec_) {
        op->TagFwdFlag();
        for (Node* var : op->GetChildrenNode()) {
            Variable* this_var = static_cast<Variable*>(var);
            for (Variable* iter : this_var->GetDescendants()) {
                iter->TagFwdFlag();
            }
            var->TagFwdFlag();
        }
    }
}

std::shared_ptr<Context> Layer::GetContextPtr() const
{
    return context_ptr_;
}

}  // namespace ater