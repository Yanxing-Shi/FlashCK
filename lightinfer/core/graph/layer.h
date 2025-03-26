#pragma once

#include <memory>
#include <vector>

#include "lightinfer/core/graph/common.h"
#include "lightinfer/core/graph/node.h"

namespace lightinfer {

class Layer {
public:
    Layer(std::string layer_name);
    virtual ~Layer() {}

    std::string GetName() const;

    virtual void Forward() final;

    void ForwardProcess() {}
    // set inputs and outputs for layers in building graph
    void SetInputs(const std::vector<Variable*>& input_var_vec);
    void SetOutputs(const std::vector<Variable*>& output_var_vec);

    // get inputs and outputs for layers
    Variable* GetInput(const int idx) const;
    Variable* GetOutput(const int idx) const;

    std::shared_ptr<Context> GetContextPtr() const;

    void ClearFwdUpdateFlag();

    std::vector<Operation*> op_vec_;

protected:
    std::shared_ptr<Context> context_ptr_;
    std::string              name_ = "";

    std::vector<Variable*> input_var_vec_  = {};
    std::vector<Variable*> output_var_vec_ = {};
};

}  // namespace lightinfer