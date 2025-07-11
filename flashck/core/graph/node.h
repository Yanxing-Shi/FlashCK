#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "flashck/core/graph/common.h"
#include "flashck/core/graph/context.h"
#include "flashck/core/graph/node.h"
#include "flashck/core/graph/shape.h"
#include "flashck/core/graph/tensor.h"

#include "flashck/core/profiling/gpu_profiling_runner.h"
#include "flashck/core/profiling/profiling_strategy.h"

namespace flashck {

class Node {
public:
    Node() = default;
    Node(std::string node_name, NodeType node_type);
    virtual ~Node();

    std::string GetName() const;
    NodeType    GetType() const;

    void               SetParentsNode(const std::vector<Node*>& parents_node);
    void               AddChildrenNode(Node* child_node);
    std::vector<Node*> GetParentsNode() const;
    std::vector<Node*> GetChildrenNode() const;

    virtual void Forward() = 0;

    void RecursiveForward();

    // bool IsCover();  // true means assign, false means accumulate

    bool IsFwdUpdate()
    {
        return is_fwd_update_;
    }

    void SetFwdUpdateFlag()
    {
        is_fwd_update_ = true;
    }

    void ClearFwdUpdateFlag()
    {
        is_fwd_update_ = false;
    }

protected:
    Context*    context_ptr_;
    std::string name_;
    NodeType    type_;

    std::vector<Node*> parents_node_{};
    std::vector<Node*> children_node_{};

    int  fwd_node_idx_;
    bool in_regress_scope_ = false;

    bool is_fwd_update_ = false;
};

class Variable: public Node {
public:
    // Applicable to variables using fixed memory, usually as input or output
    // nodes of the entire network.
    Variable(std::string name, DataType dtype);
    /*
    Applicable to the situation of self-developed memory.
    MemoryManager, which is the core of memory sharing management.
        RegressVariable - Only applicable to tensors that need to be passed
    across steps in autoregressive models.
  */
    Variable(std::string name, size_t max_size, DataType var_dtype, VarType var_type = VarType::SharedVar);

    /*
    Applicable when a variable object is a fragment of another variable object.
    For example, the calculation of qkv is a matrix multiplication operation to
    obtain the continuous tensor of qkv, but subsequent operations need to
    obtain q, k, and v respectively, so the fragments in qkv need to be
    intercepted.
  */
    Variable(std::string name, Variable* parent_var, bool is_first = false);

    virtual ~Variable() {}

    void Forward() {}

    void SetFixedMemory();

    static void SwapTwoTensor(Variable* var_x, Variable* var_y);

    void SetValue(char* value_ptr);

    char* GetValue(bool is_open_interval = false);

    template<typename T>
    T* GetValue(bool is_open_interval = false)
    {
        return (T*)GetValue(is_open_interval);
    }
    void  SetShape(const Shape& shape);
    Shape GetShape() const;

    VarType  GetType() const;
    DataType GetDtype() const;

    bool                                IsAncestor() const;
    const Variable*                     GetAncestor() const;
    const std::unordered_set<Variable*> GetDescendants() const;
    void                                AddDescendants(Variable* var);

    bool IsFirst() const
    {
        return is_first_;
    }

    void SetOffset(const size_t offset, const Shape shape);

    void MallocMemory(const size_t size);

    // auto regressive model
    void UpdateRegressiveIdx();

    // debug
    void PrintVar();

    friend class Node;

private:
    // If mx_shape is constructed by default, then tensor's memory type is
    // FixedMemory or OffsetMemory.
    size_t   max_size_;
    DataType dtype_;
    VarType  type_;
    Shape    shape_;

    Variable* parent_var_ = nullptr;
    bool      is_first_;

    std::unordered_set<Variable*> children_var_;

protected:
    std::shared_ptr<Tensor> value_ = nullptr;
};

class Operation: public Node {
public:
    Operation() = default;
    Operation(std::string name);
    virtual ~Operation() {}

    virtual void Forward() {}

    void      SetChildrenNode(std::vector<Node*> children_node);
    Variable* GetChildNode(const int index);
    Variable* GetParentNode(const int index);

    virtual void Profile(const GPUProfilingRunner& profiler_runner, const std::string& folder_name = "kernel_profile")
    {
    }

    virtual std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenOpProfiler(const ProfilingStrategy& dynamic_profiling_strategy)
    {
        FC_THROW(Unimplemented("{}", "GenOpProfiler is not implemented."));
    }

    virtual std::string GenOpFunction()
    {
        FC_THROW(Unimplemented("{}", "GenOpFunction is not implemented."));
    }

    bool has_profiler_     = true;
    bool has_gen_function_ = true;
};

}  // namespace flashck
