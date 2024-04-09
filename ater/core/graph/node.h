#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "ater/core/graph/common.h"
#include "ater/core/graph/context.h"
#include "ater/core/graph/node.h"
#include "ater/core/graph/shape.h"
#include "ater/core/graph/tensor.h"
#include "ater/core/utils/dtype.h"

#include "ater/core/profiler/base.h"
#include "ater/core/profiler/gemm_gpu_profiler_runner.h"

namespace ater {

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

    void ClearFwdFlag()
    {
        fwd_flag_ = false;
    }
    void TagFwdFlag()
    {
        fwd_flag_ = true;
    }

    // bool IsCover();  // true means assign, false means accumulate

protected:
    Context*    context_ptr_;
    std::string name_;
    NodeType    type_;

    bool fwd_flag_ = false;

    std::vector<Node*> parents_node_{};
    std::vector<Node*> children_node_{};

    int  fwd_node_idx_;
    bool in_regress_scope_ = false;
};

class Variable: public Node {
public:
    // Applicable to variables using fixed memory, usually as input or output
    // nodes of the entire network.
    Variable(std::string name, DataType dtype);
    /*
    Applicable to the situation of self-developed memory.
    Parameters:
      std::string   name
        Indicates the node name, usually named according
        to the op node that produces the var node.
      Shape   shape
        Represents the shape of the tensor recorded by the var node.
      cuda::DataType      fw_dtype
        Data type representing the forward pass tensor.
      cuda::DataType      bw_dtype
        Data type representing the backward pass tensor.
      VariableType  vt
        FixedVariable   - The memory is allocated by the var node itself.
        SharedVariable  - The memory space is managed uniformly by
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
    Variable(std::string name, Variable* parent_var_);

    ~Variable() {}

    virtual void Forward() {}

    void SetFixedMemory();

    static void SwapTwoTensor(Variable* var_x, Variable* var_y);

    void  SetValue(char* value_ptr);
    char* GetValue(bool is_open_interval = false);

    void  SetShape(const Shape& shape);
    Shape GetShape() const;

    VarType  GetType() const;
    DataType GetDtype() const;

    bool                                IsAncestor() const;
    const Variable*                     GetAncestor() const;
    const std::unordered_set<Variable*> GetDescendants() const;
    void                                AddDescendants(Variable* var);

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

    Variable*                     parent_var_ = nullptr;
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

    virtual void Profile(const std::shared_ptr<GPUProfilerRunner>& profiler_runner_ptr,
                         const std::string&                        folder_name = "kernel_profile")
    {
    }

    virtual std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenOpProfiler(const DynamicProfileStrategy& dynamic_profiling_strategy)
    {
        ATER_THROW(Unimplemented("{}", "GenOpProfiler is not implemented."));
    }

    virtual std::string GenOpFunction()
    {
        ATER_THROW(Unimplemented("{}", "GenOpFunction is not implemented."));
    };

    bool has_profiler_ = true;
};

}  // namespace ater