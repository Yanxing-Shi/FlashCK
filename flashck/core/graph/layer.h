#pragma once

#include <memory>
#include <vector>

#include "flashck/core/graph/common.h"
#include "flashck/core/graph/node.h"

namespace flashck {

/**
 * @class Layer
 * @brief High-level abstraction for neural network layers
 */
class Layer {
public:
    Layer(std::string layer_name);
    virtual ~Layer() {}

    // Layer properties
    std::string              GetName() const;
    std::shared_ptr<Context> GetContextPtr() const;

    // Execution
    virtual void Forward() final;
    void         ForwardProcess() {}  ///< Override for custom processing
    void         ClearFwdUpdateFlag();

    // I/O management
    void      SetInputs(const std::vector<Variable*>& input_var_vec);
    void      SetOutputs(const std::vector<Variable*>& output_var_vec);
    Variable* GetInput(const int idx) const;
    Variable* GetOutput(const int idx) const;

    std::vector<Operation*> op_vec_;  ///< Operations in this layer

protected:
    std::shared_ptr<Context> context_ptr_;     ///< Context pointer
    std::string              name_;            ///< Layer name
    std::vector<Variable*>   input_var_vec_;   ///< Input variables
    std::vector<Variable*>   output_var_vec_;  ///< Output variables
};

}  // namespace flashck