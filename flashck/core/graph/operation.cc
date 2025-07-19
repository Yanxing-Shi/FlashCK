#include "flashck/core/graph/node.h"

namespace flashck {

Operation::Operation(std::string name): Node(name, NodeType::Operation)
{
    context_ptr_->AddOp(this);
    VLOG(1) << "Create Operation: " << name;
}

void Operation::SetChildrenNode(std::vector<Node*> children_node)
{
    // children node is vector of variable
    if (!this->children_node_.empty()) {
        LOG(ERROR) << "The children node not empty!";
        exit(-1);
    }

    for (Node* iter : children_node) {
        iter->SetParentsNode({this});
    }
}

Variable* Operation::GetChildNode(int index)
{
    return static_cast<Variable*>(children_node_[index]);
}

Variable* Operation::GetParentNode(int index)
{
    return static_cast<Variable*>(parents_node_[index]);
}

}  // namespace flashck