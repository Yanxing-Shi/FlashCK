#include "flashck/core/graph/node.h"
#include <unistd.h>  // for write()

namespace flashck {

Operation::Operation(std::string name): Node(name, NodeType::Operation)
{
    // 最简单的输出，立即刷新
    write(STDERR_FILENO, "OPERATION CONSTRUCTOR ENTRY\n", 28);

    // 在最开始就输出，确保这行代码被执行
    fprintf(stderr, ">>> OPERATION CONSTRUCTOR START: %s <<<\n", name.c_str());
    fflush(stderr);

    // 添加更明显的调试输出
    std::cout << "=== Operation Constructor Called: " << name << " ===" << std::endl;
    std::cerr << "=== Operation Constructor Called: " << name << " ===" << std::endl;

    // 检查context_ptr_是否有效
    if (context_ptr_ == nullptr) {
        std::cerr << "ERROR: context_ptr_ is null in Operation constructor!" << std::endl;
        return;
    }

    context_ptr_->AddOp(this);

    // 多种输出方式确保能看到
    std::cout << "000000000000000000000000000000000000000000000000000000000" << std::endl;
    std::cerr << "Create Operation: " << name << std::endl;
    VLOG(1) << "000000000000000000000000000000000000000000000000000000000";
    VLOG(1) << "Create Operation: " << name;
    LOG(INFO) << "Operation created: " << name;
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