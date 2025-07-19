#pragma once

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "flashck/core/graph/common.h"
#include "flashck/core/memory/allocator.h"
#include "flashck/core/memory/memory_manager.h"

namespace flashck {

// Forward declarations
class Layer;
class Node;
class Operation;

/**
 * @class Context
 * @brief Execution context managing computational graph and memory
 */
class Context {
public:
    Context(std::string context_name, int dev_id = -1);
    virtual ~Context();

    // Properties
    std::string GetName() const;
    std::string GetModeStr() const;
    bool        IsBuilt() const;
    bool        IsBuilding() const;
    int         GetNodeIdx() const;
    hipStream_t GetStream() const;
    void        SetStream(hipStream_t stream);

    // Memory management
    std::shared_ptr<MemoryManager> GetMemoryManagerPtr() const;
    std::shared_ptr<Allocator>     GetAllocator() const;

    // Graph management
    void                    UpdateNodeIdx();
    void                    AddOp(Operation* op);
    void                    AddNode(Node* node);
    void                    EnterLayer(Layer* cur_layer, bool is_initial);
    void                    ExitLayer();
    Layer*                  GetLastLayer() const;
    Node*                   GetLastNode() const;
    std::string             GetLastNodeName() const;   ///< Get name of last node (helper to avoid circular deps)
    std::string             GetLastLayerName() const;  ///< Get name of last layer (helper to avoid circular deps)
    bool                    CheckIfInit();
    std::vector<Operation*> GetModelOps() const
    {
        return model_ops_;
    }

    // Context lifecycle
    void BuildContext();

    // Global context management
    static int                      CreateGlobalContext(const std::string& model_name, const int device_id = -1);
    static std::shared_ptr<Context> GetGlobalInstance();
    static void                     SetGlobalContext(const std::string& context_name);

    // Name counters for uniqueness
    std::map<std::string, int> layer_name_cnt;  ///< Layer name counter
    std::map<std::string, int> node_name_cnt;   ///< Node name counter

    // Temporary memory for graph building
    size_t      max_tensor_size_ = 0;  ///< Maximum tensor size
    std::string max_tensor_name_;      ///< Name of largest tensor
    char*       tmp_buff_ = nullptr;   ///< Temporary buffer

private:
    std::string context_name_;  ///< Context name
    int         dev_id_;        ///< Device ID
    hipStream_t stream_ = 0;    ///< HIP stream

    bool                           is_context_built_    = false;  ///< Context built flag
    bool                           is_context_building_ = false;  ///< Context building flag
    std::shared_ptr<MemoryManager> mem_manager_ptr_;              ///< Memory manager
    std::shared_ptr<Allocator>     allocator_ptr_;                ///< Allocator

    // Graph data structures
    int node_idx_ = 0;  ///< Current node index

    std::vector<Operation*> model_ops_{};      ///< All operations
    std::vector<Node*>      all_node_vec_{};   ///< All nodes
    std::vector<Layer*>     root_layers_{};    ///< Root layers
    std::vector<Layer*>     all_layers_{};     ///< All layers
    std::deque<Layer*>      layer_context_{};  ///< Layer context stack

    // Global context management
    static int                                                       global_context_id_;    ///< Global context counter
    static std::unordered_map<std::string, std::shared_ptr<Context>> global_contexts_map_;  ///< Context map
    static std::shared_ptr<Context>                                  global_context_ptr_;   ///< Current global context
};

}  // namespace flashck