#include "ater/core/graph/node.h"

#include "ater/core/utils/timer.h"

namespace ater {

Node::Node(std::string name, NodeType type): context_ptr_(Context::GetGlobalInstance().get()), type_(type)
{
    std::string prefix_name = context_ptr_->GetLastLayer() ? (context_ptr_->GetLastLayer()->GetName() + "_") : "";
    std::string real_name   = prefix_name + name;
    int         idx         = context_ptr_->node_name_cnt[real_name];
    context_ptr_->node_name_cnt[real_name] += 1;
    name_ = real_name + "_" + std::to_string(idx);
    if (context_ptr_->GetRegressStatus()) {
        in_regress_scope_ = true;
    }

    context_ptr_->AddNode(this);
}

Node::~Node()
{
    parents_node_.clear();
    children_node_.clear();
}

std::string Node::GetName() const
{
    return name_;
}

NodeType Node::GetType() const
{
    return type_;
}

void Node::SetParentsNode(const std::vector<Node*>& parents_node)
{
    int idx = 0;
    for (Node* iter : parents_node) {
        // VLOG(1) << " node name " << name_ << " append parent node: " << iter->GetName();
        parents_node_.push_back(iter);
        if (iter == nullptr)
            LOG(WARNING) << "Node " << name_ << " parent #" << idx << " is nullptr";
        else
            iter->AddChildrenNode(this);
        idx++;
    }

    if (GetType() == NodeType::Operation) {
        if (context_ptr_->GetRegressStatus()) {
            in_regress_scope_ = true;
        }
    }
}

void Node::AddChildrenNode(Node* child_node)
{
    children_node_.push_back(child_node);
}

std::vector<Node*> Node::GetParentsNode() const
{
    return parents_node_;
}

std::vector<Node*> Node::GetChildrenNode() const
{
    return children_node_;
}

void Node::RecursiveForward()
{
    if (fwd_flag_)
        return;

    for (Node* iter : parents_node_) {
        iter->RecursiveForward();
    }

    if (GetType() == NodeType::Variable) {
        Variable* this_var = static_cast<Variable*>(this);
        if (this_var->GetType() == VarType::OffsetVar) {
            this_var->parent_var_->RecursiveForward();
        }
    }

    fwd_flag_ = true;

    if (GetType() == NodeType::Operation) {
        context_ptr_->UpdateNodeIdx();
    }

    // auto regressive model
    if (!context_ptr_->IsBuilt()) {
        fwd_node_idx_ = context_ptr_->GetNodeIdx();
        if (in_regress_scope_) {
            context_ptr_->UpdateBeginRegressIdx(fwd_node_idx_);
            context_ptr_->UpdateEndRegressIdx(fwd_node_idx_);
        }
    }

    /*-----------------------------------------debug-----------------------*/
    // if (GetType() == NodeType::Operation && context_ptr_->IsBuilt()) {
    //     Operation* this_op = static_cast<Operation*>(this);
    //     VLOG(1) << this_op->name_ << " forward, fwd node idx: " << fwd_node_idx_;

    //     for (int idx = 0; idx < parents_node_.size(); idx++) {
    //         if (parents_node_[idx] == nullptr) {
    //             VLOG(1) << "parents_node_" << idx << " is nullptr";
    //         }
    //         else {
    //             this_op->GetParentNode(idx)->PrintVar();
    //         }
    //     }
    // }

    // if (GetType() == NodeType::Operation) {
    //     VLOG(1) << GetName() << "forward, fwd node idx: " << fwd_node_idx_;
    // }

    // context_ptr_->Synchronize();

    // CPUTimer cpu_timer;
    // cpu_timer.Tic();

    Forward();

    /*-----------------------------------------debug-----------------------*/
    // if (GetType() != NodeType::Operation || !context_ptr_->IsBuilt()) {
    //     return;
    // }

    // context_ptr_->Synchronize();

    // VLOG(1) << "time cost: " << cpu_timer.Toc() << " ms";
    // Operation* this_op = static_cast<Operation*>(this);
    // VLOG(1) << "children_node_.size()" << children_node_.size();

    // for (int idx = 0; idx < children_node_.size(); idx++) {
    //     if (children_node_[idx] == nullptr)
    //         VLOG(1) << "node idx: " << idx << " nullptr";
    //     else
    //         this_op->GetChildNode(idx)->PrintVar();
    // }
}

}  // namespace ater