#pragma once

#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

#include "flashck/core/graph/common.h"
#include "flashck/core/graph/context.h"
#include "flashck/core/graph/shape.h"
#include "flashck/core/graph/tensor.h"
#include "flashck/core/utils/enforce.h"

namespace flashck {

// Forward declarations
enum class ProfilingStrategy : int;
class GPUProfilingRunner;

/**
 * @class Node
 * @brief Base class for computational graph nodes
 */
class Node {
public:
    Node() = default;
    Node(std::string node_name, NodeType node_type);
    virtual ~Node();

    // Node properties
    std::string GetName() const;
    NodeType    GetType() const;

    // Graph connectivity
    void               SetParentsNode(const std::vector<Node*>& parents_node);
    void               AddChildrenNode(Node* child_node);
    std::vector<Node*> GetParentsNode() const;
    std::vector<Node*> GetChildrenNode() const;

    // Execution
    virtual void Forward() = 0;
    void         RecursiveForward();

    // Update tracking
    bool IsFwdUpdate() const
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
    Context*    context_ptr_;  ///< Context pointer
    std::string name_;         ///< Node name
    NodeType    type_;         ///< Node type

    std::vector<Node*> parents_node_{};   ///< Parent nodes
    std::vector<Node*> children_node_{};  ///< Child nodes

    int  fwd_node_idx_;              ///< Forward node index
    bool in_regress_scope_ = false;  ///< In regression scope flag
    bool is_fwd_update_    = false;  ///< Forward update flag
};

/**
 * @class Variable
 * @brief Variable node representing tensor data in computational graph
 */
class Variable: public Node {
public:
    /// Constructor for fixed memory variables (input/output)
    Variable(std::string name, DataType dtype);

    /// Constructor for shared/managed memory variables
    Variable(std::string name, size_t max_size, DataType var_dtype, VarType var_type = VarType::SharedVar);

    /// Constructor for offset variables (sub-tensor of parent)
    Variable(std::string name, Variable* parent_var, bool is_first = false);

    virtual ~Variable() {}

    void Forward() {}

    // Memory management
    void  SetFixedMemory();
    void  SetValue(char* value_ptr);
    char* GetValue(bool is_open_interval = false);

    template<typename T>
    T* GetValue(bool is_open_interval = false)
    {
        return (T*)GetValue(is_open_interval);
    }

    // Tensor properties
    void     SetShape(const Shape& shape);
    Shape    GetShape() const;
    VarType  GetType() const;
    DataType GetDtype() const;
    void     SetOffset(const size_t offset, const Shape shape);
    void     MallocMemory(const size_t size);

    // Hierarchy management
    bool            IsAncestor() const;
    const Variable* GetAncestor() const;
    Variable*       GetParentVar() const
    {
        return parent_var_;
    }  ///< Get parent variable for offset vars
    const std::unordered_set<Variable*> GetDescendants() const;
    void                                AddDescendants(Variable* var);
    bool                                IsFirst() const
    {
        return is_first_;
    }

    // Utilities
    static void SwapTwoTensor(Variable* var_x, Variable* var_y);
    void        PrintVar();

private:
    size_t   max_size_;  ///< Maximum size for shared memory
    DataType dtype_;     ///< Data type
    VarType  type_;      ///< Variable type
    Shape    shape_;     ///< Tensor shape

    Variable*                     parent_var_ = nullptr;  ///< Parent variable for offset
    bool                          is_first_;              ///< First variable flag
    std::unordered_set<Variable*> children_var_;          ///< Child variables

protected:
    std::shared_ptr<Tensor> value_ = nullptr;  ///< Underlying tensor
};

/**
 * @class Operation
 * @brief Operation node representing computation in graph
 */
class Operation: public Node {
public:
    Operation() { 
        fprintf(stderr, ">>> OPERATION DEFAULT CONSTRUCTOR CALLED <<<\n"); 
        fflush(stderr); 
    }
    Operation(std::string name);
    virtual ~Operation() {}

    virtual void Forward() {}

    // Node access
    void      SetChildrenNode(std::vector<Node*> children_node);
    Variable* GetChildNode(const int index);
    Variable* GetParentNode(const int index);

    // Code generation for profiling and execution
    virtual std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    CodeGenForTuning(const ProfilingStrategy& dynamic_profiling_strategy)
    {
        FC_THROW(Unimplemented("CodeGenForTuning not implemented"));
    }

    virtual std::string CodeGenForRunning()
    {
        FC_THROW(Unimplemented("CodeGenForRunning not implemented"));
    }

    virtual void Tuning(GPUProfilingRunner& profiler_runner, const std::string& folder_name = "kernel_profile")
    {
        FC_THROW(Unimplemented("Tuning not implemented"));
    }

    bool has_profiling_engine_ = true;  ///< Has profiling engine flag
    bool has_gen_function_     = true;  ///< Has code generation flag

    std::vector<Variable*> input_var_;   ///< Input variables
    std::vector<Variable*> output_var_;  ///< Output variables
};

}  // namespace flashck
