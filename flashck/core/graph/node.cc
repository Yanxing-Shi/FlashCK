#include "flashck/core/graph/node.h"

#include "flashck/core/utils/enforce.h"

namespace flashck {

Node::Node(std::string name, NodeType type): context_ptr_(Context::GetGlobalInstance().get()), type_(type)
{
    std::cout << "=== Node Constructor Called: " << name << " (type: " << static_cast<int>(type)
              << ") ===" << std::endl;

    // Check if context is valid
    if (context_ptr_ == nullptr) {
        std::cerr << "ERROR: context_ptr_ is null in Node constructor!" << std::endl;
        return;
    }

    // Generate unique node name with layer prefix
    std::string last_layer_name = context_ptr_->GetLastLayerName();
    std::string prefix_name     = last_layer_name.empty() ? "" : (last_layer_name + "_");
    std::string real_name       = prefix_name + name;

    int idx = context_ptr_->node_name_cnt[real_name];
    context_ptr_->node_name_cnt[real_name] += 1;
    name_ = real_name + "_" + std::to_string(idx);

    std::cout << "Node final name: " << name_ << std::endl;

    context_ptr_->AddNode(this);

    std::cout << "Node constructor completed for: " << name_ << std::endl;
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
    for (size_t idx = 0; idx < parents_node.size(); ++idx) {
        Node* iter = parents_node[idx];
        parents_node_.push_back(iter);

        if (iter == nullptr) {
            LOG(WARNING) << "Node " << name_ << " parent #" << idx << " is nullptr";
        }
        else {
            iter->AddChildrenNode(this);
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
    if (is_fwd_update_) {
        return;  // Already processed
    }

    // Process all parent nodes first
    for (Node* iter : parents_node_) {
        iter->RecursiveForward();
    }

    // Handle offset variable special case
    if (GetType() == NodeType::Variable) {
        Variable* this_var = static_cast<Variable*>(this);
        if (this_var->GetType() == VarType::OffsetVar && this_var->IsFirst()) {
            Variable* parent_var = this_var->GetParentVar();
            if (parent_var) {
                parent_var->RecursiveForward();
            }
        }
    }

    // Mark as updated and execute forward
    is_fwd_update_ = true;

    if (GetType() == NodeType::Operation) {
        context_ptr_->UpdateNodeIdx();
        VLOG(1) << GetName() << " forward, node idx: " << fwd_node_idx_;
    }

    Forward();
}

// Operation class implementations

Operation::Operation(std::string name): Node(name, NodeType::Operation) {}

void Operation::SetChildrenNode(std::vector<Node*> children_node)
{
    // Set output variables from children nodes
    output_var_.clear();
    for (Node* node : children_node) {
        if (node && node->GetType() == NodeType::Variable) {
            output_var_.push_back(static_cast<Variable*>(node));
        }
    }
    SetParentsNode(children_node);
}

Variable* Operation::GetChildNode(const int index)
{
    // Get output variable at specified index
    FC_ENFORCE_LT(
        index, output_var_.size(), InvalidArgument("Operation {} output index {} out of range", GetName(), index));
    return output_var_[index];
}

Variable* Operation::GetParentNode(const int index)
{
    // Get input variable at specified index
    FC_ENFORCE_LT(
        index, input_var_.size(), InvalidArgument("Operation {} input index {} out of range", GetName(), index));
    return input_var_[index];
}

}  // namespace flashck