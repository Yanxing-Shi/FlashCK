#include "core/graph/context.h"

#include "core/memory/memory_manager.h"

namespace flashck {

Context::Context(std::string context_name, int dev_id):
    context_name_(context_name),
    dev_id_(dev_id),
    mem_manager_ptr_(new MemoryManager()),
    allocator_ptr_(mem_manager_ptr_->GetAllocator())
{
    VLOG(1) << "Initial Context\n";
}

Context::~Context()
{
    // for (auto iter : all_node_vec_) {
    //     delete iter;
    // }
}

/*---------------- Property -----------------------*/
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

/*---------------- graph -----------------------*/
int Context::GetNodeIdx() const
{
    return node_idx_;
}

void Context::UpdateNodeIdx()
{
    if (is_context_built_)
        return;
    node_idx_++;
}

// add operations into layers
void Context::AddOp(Operation* op)
{
    if (IsBuilt()) {
        LOG(ERROR) << "Context has constructed! should not add new operator!";
        exit(-1);
    }

    if (layer_context_.size()) {
        for (Layer* ly : layer_context_) {
            ly->op_vec_.push_back(op);
        }
    }

    model_ops_.push_back(op);
}

// add node into node vector
void Context::AddNode(Node* node)
{
    all_node_vec_.push_back(node);
}

// add layer into layer_context, is_initial show if root layer
// Init layer and set input var need to call
void Context::EnterLayer(Layer* cur_layer, bool is_initial)
{
    if (IsBuilt()) {
        LOG(ERROR) << "Context has constructed! should not modify network";
        exit(-1);
    }

    // root layer
    if (layer_context_.size() == 0 && is_initial == false) {
        root_layers_.push_back(cur_layer);
    }

    // all layer
    else if (is_initial == true)
        all_layers_.push_back(cur_layer);

    layer_context_.push_back(cur_layer);
}

// delete layer in context
void Context::ExitLayer()
{
    layer_context_.pop_back();
}

// get last layer in conetxt
Layer* Context::GetLastLayer() const
{
    return layer_context_.size() ? layer_context_.back() : nullptr;
}

// get last node in context
Node* Context::GetLastNode() const
{
    return all_node_vec_.size() ? all_node_vec_[all_node_vec_.size() - 1] : nullptr;
}

bool Context::CheckIfInit()
{
    bool check_flag = true;

    // Check Layer
    for (Layer* layer : all_layers_) {
        if (layer->GetName().size() == 0) {
            LOG(ERROR) << "error! Some Layers not initialize";
            check_flag = false;
        }
    }

    // Check Op
    for (Operation* op : model_ops_) {
        if (op->GetName().size() == 0) {
            LOG(ERROR) << "error! some OPERATORS didn't initialize!";
            check_flag = false;
        }
    }

    return check_flag;
}

void Context::BuildContext()
{
    if (is_context_built_ || is_context_building_)
        return;

    // start context build
    is_context_building_ = true;

    VLOG(1) << "start flashck context build ";

    // check if layer and op init
    if (!CheckIfInit()) {
        LOG(ERROR) << "Check validate error!";
        exit(-1);
    }

    VLOG(1)
        << "Please pay attention to whether the build order of the layer is consistent with the actual execution order ";

    // try {
    //     // Before the memory allocation, the tensor is not allocated the actual
    //     // effective address space, so it is necessary to give a temporary space for
    //     // some steps to test.
    //     tmp_buff_ = allocator_ptr_->Malloc(max_tensor_size_);
    // }
    // catch (...) {
    //     FC_THROW(ResourceExhausted(
    //         "allocate temporary buffer failed!\n, max_tensor_name_ is: {}, max_tensor_size_ is: {} MB",
    //         max_tensor_name_,
    //         max_tensor_size_ / (1024 * 1024)));
    // }

    for (int idx = 0; idx < model_ops_.size(); idx++) {
        model_ops_[idx]->RecursiveForward();
    }

    for (Layer* root_layer : root_layers_) {
        VLOG(1) << "context start build layer: " << root_layer->GetName();
        root_layer->Forward();
    }

    VLOG(1) << "Context has build layer ";

    // try {
    //     allocator_ptr_->Free(tmp_buff_);
    // }
    // catch (...) {
    //     FC_THROW(ResourceExhausted("free temporary buffer {} failed!", tmp_buff_));
    // }

    mem_manager_ptr_->CalculateBuffer();

    is_context_built_ = true;

    // HIP_ERROR_CHECK(hipStreamSynchronize(stream_));

    VLOG(1) << "Finish context build success ";
}

int Context::CreateGlobalContext(const std::string& context_name)
{
    global_context_id_++;
    std::shared_ptr<Context> context_ptr = std::make_shared<Context>(context_name);
    global_context_ptr_                  = context_ptr;
    if (global_contexts_map_.find(context_name) != global_contexts_map_.end()) {
        // LOG(WARNING) << "Error occured! context_id " << context_name << " already exists!";
    }
    global_contexts_map_.emplace(context_name, context_ptr);
    VLOG(1) << "create global context success" << "context name: " << context_name
            << "context_id:" << global_context_id_;
    return global_context_id_;
}

void Context::SetGlobalContext(const std::string& context_name)
{
    auto iter = global_contexts_map_.find(context_name);
    if (iter == global_contexts_map_.end()) {
        LOG(WARNING) << "Error occured! context_id " << context_name << " does not exist!";
        return;
    }

    global_context_ptr_ = iter->second;
}

std::shared_ptr<Context> Context::GetGlobalInstance()
{
    return global_context_ptr_;
}

int                                                       Context::global_context_id_   = 0;
std::shared_ptr<Context>                                  Context::global_context_ptr_  = nullptr;
std::unordered_map<std::string, std::shared_ptr<Context>> Context::global_contexts_map_ = {};

}  // namespace flashck
