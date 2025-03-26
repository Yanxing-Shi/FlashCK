#pragma once

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "lightinfer/core/graph/common.h"
#include "lightinfer/core/graph/layer.h"
#include "lightinfer/core/memory/allocator.h"
#include "lightinfer/core/memory/memory_manager.h"

#include "lightinfer/core/profiler/base.h"

namespace lightinfer {

class Context {
public:
    Context(std::string context_name, Mode mode, int dev_id = -1);

    virtual ~Context();

    // Property
    std::string                    GetName() const;
    Mode                           GetMode() const;
    std::string                    GetModeStr() const;
    bool                           IsBuilt() const;
    bool                           IsBuilding() const;
    std::shared_ptr<MemoryManager> GetMemoryManagerPtr() const;
    std::shared_ptr<Allocator>     GetAllocator() const;
    hipStream_t                    GetStream() const;
    void                           SetStream(hipStream_t stream);

    // Graph
    int    GetNodeIdx() const;
    void   UpdateNodeIdx();
    void   AddOp(Operation* op);
    void   AddNode(Node* node);
    void   EnterLayer(Layer* cur_layer, bool is_initial);
    void   ExitLayer();
    Layer* GetLastLayer() const;
    Node*  GetLastNode() const;
    bool   CheckIfInit();

    // codegen profiler

    void CodegenAndProfileKernel(const DynamicProfileStrategy& strategy = DynamicProfileStrategy::MAX);

    // Register
    static void
    RegisterPyBindLayer(const std::string& layer_name, const int layer_id, const std::shared_ptr<void>& layer_ptr);
    static std::shared_ptr<char> GetPyBindLayer(const std::string& layer_name, const int layer_id);
    void                         RegisterObject(const std::string& object_name, char* object);
    char*                        GetObject(const std::string& object_name) const;

    // Context
    void BuildContext();

    static int
    CreateGlobalContext(const std::string& model_name, Mode mode = Mode::Inference, const int device_id = -1);
    static std::shared_ptr<Context> GetGlobalInstance();
    static void                     SetGlobalContext(const std::string& context_name);

    // Auto Regression model
    void SetBeginRegress();
    void SetEndRegress();
    int  GetBeginRegressIdx() const;
    int  GetEndRegressIdx() const;
    void UpdateBeginRegressIdx(int node_idx);
    void UpdateEndRegressIdx(int node_idx);
    bool GetRegressStatus() const;

    // Debug
    void Synchronize();

    // it will record layer info when layer constructor
    std::map<std::string, int> layer_name_cnt;
    // it will record node info when node constructor
    std::map<std::string, int> node_name_cnt;

    // Memory Allocate
    // Before the memory allocation, the tensor is not allocated the actual
    // effective address space, so it is necessary to give a temporary space for
    // some steps to test.
    size_t      max_tensor_size_ = 0;
    std::string max_tensor_name_ = "";
    char*       tmp_buff_        = nullptr;

private:
    std::string context_name_;
    int         dev_id_;
    hipStream_t stream_ = 0;

    // Property
    Mode mode_;

    bool                           is_context_built_    = false;
    bool                           is_context_building_ = false;
    std::shared_ptr<MemoryManager> mem_manager_ptr_;
    std::shared_ptr<Allocator>     allocator_ptr_;

    // Graph
    int node_idx_ = 0;

    std::vector<Operation*> model_ops_{};
    std::vector<Node*>      all_node_vec_{};
    std::vector<Layer*>     root_layers_{};  // record root layers
    std::vector<Layer*>     all_layers_{};   // record all layers
    std::deque<Layer*>      layer_context_;  // contain all layers in context

    // Register
    static std::unordered_map<std::string, std::shared_ptr<char>> register_layers_;
    std::unordered_map<std::string, void*>                        register_pool_;

    // context
    static int global_context_id_;  // record how many times context is initialized
    static std::unordered_map<std::string, std::shared_ptr<Context>> global_contexts_map_;
    static std::shared_ptr<Context>                                  global_context_ptr_;

    // Auto Regression model
    int  begin_regress_idx_ = -1;
    int  end_regress_idx_   = -1;
    bool in_regress_        = false;
};

};  // namespace lightinfer