#include "flashck/core/graph/context.h"

#include "flashck/core/graph/layer.h"
#include "flashck/core/graph/node.h"
#include "flashck/core/memory/memory_manager.h"
#include "flashck/core/utils/enforce.h"

namespace flashck {

Context::Context(std::string context_name, int dev_id):
    context_name_(context_name),
    dev_id_(dev_id),
    mem_manager_ptr_(new MemoryManager()),
    allocator_ptr_(mem_manager_ptr_->GetAllocator())
{
    LOG(INFO) << "Initialized Context: " << context_name_;
}

Context::~Context()
{
    // Note: Nodes are managed externally, not deleted here
}

// Properties
std::string Context::GetName() const
{
    return context_name_;
}

bool Context::IsBuilt() const
{
    return is_context_built_;
}

bool Context::IsBuilding() const
{
    return is_context_building_;
}

std::string Context::GetModeStr() const
{
    // Return context execution mode as string
    return is_context_built_ ? "Built" : (is_context_building_ ? "Building" : "Idle");
}

std::shared_ptr<MemoryManager> Context::GetMemoryManagerPtr() const
{
    return mem_manager_ptr_;
}

std::shared_ptr<Allocator> Context::GetAllocator() const
{
    return allocator_ptr_;
}

hipStream_t Context::GetStream() const
{
    return stream_;
}

void Context::SetStream(hipStream_t stream)
{
    stream_ = stream;
}

// Graph management
int Context::GetNodeIdx() const
{
    return node_idx_;
}

void Context::UpdateNodeIdx()
{
    // Only update node index during building phase
    if (!is_context_built_) {
        node_idx_++;
    }
}

void Context::AddOp(Operation* op)
{
    // Add operation to current layer context
    FC_ENFORCE_EQ(IsBuilt(), false, Unavailable("Cannot add operator to built context"));

    // Add operation to all active layers in context stack
    for (Layer* ly : layer_context_) {
        ly->op_vec_.push_back(op);
    }

    model_ops_.push_back(op);
}

void Context::AddNode(Node* node)
{
    // Add node to global node vector
    all_node_vec_.push_back(node);
}

void Context::EnterLayer(Layer* cur_layer, bool is_initial)
{
    // Enter layer context for graph construction
    FC_ENFORCE_EQ(IsBuilt(), false, Unavailable("Cannot modify built context"));

    // Handle root layer registration
    if (layer_context_.empty() && !is_initial) {
        root_layers_.push_back(cur_layer);
    }
    // Handle all layer registration
    else if (is_initial) {
        all_layers_.push_back(cur_layer);
    }

    layer_context_.push_back(cur_layer);
}

// delete layer in context

void Context::ExitLayer()
{
    // Exit current layer context
    if (!layer_context_.empty()) {
        layer_context_.pop_back();
    }
}

Layer* Context::GetLastLayer() const
{
    // Return the last active layer in context stack
    return layer_context_.empty() ? nullptr : layer_context_.back();
}

Node* Context::GetLastNode() const
{
    // Return the most recently added node
    return all_node_vec_.empty() ? nullptr : all_node_vec_.back();
}

std::string Context::GetLastNodeName() const
{
    // Helper method to get last node name without circular dependency
    Node* last_node = GetLastNode();
    return last_node ? last_node->GetName() : "";
}

std::string Context::GetLastLayerName() const
{
    // Helper method to get last layer name without circular dependency
    Layer* last_layer = GetLastLayer();
    return last_layer ? last_layer->GetName() : "";
}

bool Context::CheckIfInit()
{
    // Validate that all components are properly initialized
    bool check_flag = true;

    // Check layers have valid names
    for (Layer* layer : all_layers_) {
        if (layer->GetName().empty()) {
            LOG(ERROR) << "Error: Some layers not initialized";
            check_flag = false;
        }
    }

    // Check operations have valid names
    for (Operation* op : model_ops_) {
        if (op->GetName().empty()) {
            LOG(ERROR) << "Error: Some operations not initialized";
            check_flag = false;
        }
    }

    return check_flag;
}

void Context::BuildContext()
{
    // Build execution context if not already built
    if (is_context_built_ || is_context_building_) {
        return;
    }

    is_context_building_ = true;
    VLOG(1) << "Starting FlashCK context build";

    // Validate all components are initialized
    FC_ENFORCE_EQ(CheckIfInit(), true, Unavailable("Context validation failed"));

    VLOG(1) << "Context validation passed";
    VLOG(1)
        << "Please pay attention to whether the build order of the layer is consistent with the actual execution order";

    try {
        // Allocate temporary buffer for tensor operations during graph building
        tmp_buff_ = allocator_ptr_->Malloc(max_tensor_size_);
        VLOG(1) << "Allocated temporary buffer: " << max_tensor_size_ / (1024 * 1024) << " MB";
    }
    catch (...) {
        FC_THROW(ResourceExhausted("Failed to allocate temporary buffer for tensor: {}, size: {} MB",
                                   max_tensor_name_,
                                   max_tensor_size_ / (1024 * 1024)));
    }

    // Execute forward pass for all operations
    for (size_t idx = 0; idx < model_ops_.size(); idx++) {
        model_ops_[idx]->RecursiveForward();
    }

    // Execute forward pass for all root layers
    for (Layer* root_layer : root_layers_) {
        VLOG(1) << "Building layer: " << root_layer->GetName();
        root_layer->Forward();
    }

    VLOG(1) << "Layer building completed";

    // Free temporary buffer
    try {
        allocator_ptr_->Free(tmp_buff_);
    }
    catch (...) {
        FC_THROW(ResourceExhausted("Failed to free temporary buffer {}", tmp_buff_));
    }

    // Calculate final memory buffer layout
    mem_manager_ptr_->CalculateBuffer();
    is_context_built_ = true;

    // Synchronize GPU stream
    HIP_ERROR_CHECK(hipStreamSynchronize(stream_));
    VLOG(1) << "Context build completed successfully";
}

int Context::CreateGlobalContext(const std::string& context_name, const int device_id)
{
    // Create and register new global context
    global_context_id_++;
    auto context_ptr    = std::make_shared<Context>(context_name, device_id);
    global_context_ptr_ = context_ptr;

    FC_ENFORCE_EQ(global_contexts_map_.find(context_name),
                  global_contexts_map_.end(),
                  InvalidArgument("Context '{}' already exists", context_name));

    global_contexts_map_.emplace(context_name, context_ptr);
    LOG(INFO) << "Created global context '" << context_name << "' with ID: " << global_context_id_;
    return global_context_id_;
}

void Context::SetGlobalContext(const std::string& context_name)
{
    // Switch to specified global context
    auto iter = global_contexts_map_.find(context_name);
    FC_ENFORCE_NE(iter, global_contexts_map_.end(), InvalidArgument("Context '{}' does not exist", context_name));

    global_context_ptr_ = iter->second;
}

std::shared_ptr<Context> Context::GetGlobalInstance()
{
    return global_context_ptr_;
}

// Static member definitions
int                                                       Context::global_context_id_   = 0;
std::shared_ptr<Context>                                  Context::global_context_ptr_  = nullptr;
std::unordered_map<std::string, std::shared_ptr<Context>> Context::global_contexts_map_ = {};

}  // namespace flashck
