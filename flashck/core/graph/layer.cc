#include "flashck/core/graph/layer.h"

#include "flashck/core/utils/enforce.h"

namespace flashck {

Layer::Layer(std::string name): op_vec_({})
{
    context_ptr_ = Context::GetGlobalInstance();

    // Generate unique layer name with parent prefix
    std::string layer_name =
        context_ptr_->GetLastLayer() ? (context_ptr_->GetLastLayer()->GetName() + "_" + name) : name;

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
    context_ptr_->UpdateNodeIdx();
    ClearFwdUpdateFlag();

    // Execute forward pass for all output variables
    for (Variable* var : output_var_vec_) {
        if (var != nullptr) {
            VLOG(1) << "Layer " << GetName() << " output: " << var->GetName();
            var->RecursiveForward();
        }
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
    FC_ENFORCE_LT(idx, input_var_vec_.size(), InvalidArgument("Layer {} input index {} out of range", GetName(), idx));
    return input_var_vec_[idx];
}

Variable* Layer::GetOutput(const int idx) const
{
    FC_ENFORCE_LT(
        idx, output_var_vec_.size(), InvalidArgument("Layer {} output index {} out of range", GetName(), idx));
    return output_var_vec_[idx];
}

std::shared_ptr<Context> Layer::GetContextPtr() const
{
    return context_ptr_;
}

void Layer::ClearFwdUpdateFlag()
{
    for (Operation* op : op_vec_) {
        op->ClearFwdUpdateFlag();
        for (Node* var : op->GetChildrenNode()) {
            Variable* this_var = static_cast<Variable*>(var);
            for (Variable* iter : this_var->GetDescendants()) {
                iter->ClearFwdUpdateFlag();
            }
            var->ClearFwdUpdateFlag();
        }
    }
}

}  // namespace flashck