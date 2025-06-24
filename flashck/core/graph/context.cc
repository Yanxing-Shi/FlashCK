#include "flashck/core/graph/context.h"

#include "flashck/core/memory/memory_manager.h"
#include "flashck/core/utils/rocm_info.h"

#include "flashck/core/profiler/builder.h"
#include "flashck/core/profiler/codegen.h"
#include "flashck/core/profiler/gpu_profiler_runner.h"
#include "flashck/core/profiler/target.h"

namespace flashck {

Context::Context(std::string context_name, Mode mode, int dev_id):
    context_name_(context_name),
    dev_id_(dev_id),
    mode_(mode),
    mem_manager_ptr_(new MemoryManager()),
    allocator_ptr_(mem_manager_ptr_->GetAllocator())
{
    LOG(INFO) << "Initial Context, status_type: " << ModeToStr(mode_) << "\n";
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

Mode Context::GetMode() const
{
    return mode_;
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

// kernel tunning
void Context::CodegenAndProfileKernel(const DynamicProfileStrategy& strategy)
{
    // step1: gen profiler
    // using FLAGS_selected_gpus to Get devices
    std::vector<int> selected_gpu = Target::Instance()->GetTargetSelectedDevices();
    VLOG(1) << "Selected devices for profiling: " << selected_gpu;

    std::vector<std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>> graph_generated_profilers =
        GenProfiler(model_ops_, strategy);

    // // step.2 profile result
    VLOG(1) << "Profiler generated " << graph_generated_profilers.size() << " model operations";
    Builder builder;
    builder.MakeProfilers(graph_generated_profilers, context_name_);
    auto postprocess_ptr     = std::make_shared<ProfilerPostprocess>();
    auto profiler_runner_ptr = std::make_shared<GPUProfilerRunner>(selected_gpu.size(), postprocess_ptr, 100000);
    for (Operation* op_ptr : model_ops_) {
        op_ptr->Profile(profiler_runner_ptr);
    }
    profiler_runner_ptr->Join();

    // // step3 gen kernel source function
    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>> file_tuples =
        GenFunctionSource(model_ops_, context_name_);
    builder.MakeExecutors(file_tuples, "generated_kernel.so", context_name_);

    // read dll and load function
    Target::Instance()->DllReader("kernel_profile", context_name_, "generated_kernel.so");
}

/*----------------Register--------------------------------*/
// void Context::RegisterPyBindLayer(const std::string&           layer_name,
//                                   const int                    layer_id,
//                                   const std::shared_ptr<void>& layer_ptr)
// {
//     std::string register_name = layer_name + std::to_string(layer_id);
//     if (register_layers_.find(register_name) != register_layers_.end()) {
//         LOG(ERROR) << "The layer applied for registration has been occupied! Layer name is " << register_name;
//         exit(-1);
//     }

//     register_layers_.emplace(register_name, layer_ptr);
// }

// std::shared_ptr<char> Context::GetPyBindLayer(const std::string& layer_name, const int layer_id)
// {
//     std::string register_name = layer_name + std::to_string(layer_id);
//     auto        iter          = register_layers_.find(register_name);
//     if (iter == register_layers_.end()) {
//         LOG(ERROR) << "The requested layer was not found! Layer name is " << register_name;
//         exit(-1);
//     }
//     return iter->second;
// }

// void Context::RegisterObject(const std::string& object_name, void* object)
// {
//     if (register_pool_.find(object_name) != register_pool_.end()) {
//         LOG(ERROR) << "Error! register same name( " << object_name << " ) twice!";
//         exit(-1);
//     }
//     register_pool_.emplace(object_name, object);
// }

// char* Context::GetObject(const std::string object_name) const
// {
//     auto iter = register_pool_.find(object_name);
//     if (iter == register_pool_.end()) {
//         LOG(ERROR) << "Error! can't get " << object_name;
//         exit(-1);
//     }
//     return iter->second;
// }

/*---------------- Context -----------------------*/

void Context::BuildContext()
{
    if (is_context_built_ || is_context_building_)
        return;

    // start context build
    is_context_building_ = true;

    VLOG(1) << "start flashck context build, Mode: " << ModeToStr(mode_);

    // check if layer and op init
    if (!CheckIfInit()) {
        LOG(ERROR) << "Check validate error!";
        exit(-1);
    }

    VLOG(1)
        << "Please pay attention to whether the build order of the layer is consistent with the actual execution order ";

    try {
        // Before the memory allocation, the tensor is not allocated the actual
        // effective address space, so it is necessary to give a temporary space for
        // some steps to test.
        tmp_buff_ = allocator_ptr_->Malloc(max_tensor_size_);
    }
    catch (...) {
        LI_THROW(ResourceExhausted(
            "allocate temporary buffer failed!\n, max_tensor_name_ is: {}, max_tensor_size_ is: {} MB",
            max_tensor_name_,
            max_tensor_size_ / (1024 * 1024)));
    }

    for (int idx = 0; idx < model_ops_.size(); idx++) {
        model_ops_[idx]->RecursiveForward();
    }

    for (Layer* root_layer : root_layers_) {
        VLOG(1) << "context start build layer: " << root_layer->GetName();
        root_layer->Forward();
    }

    VLOG(1) << "Context has build layer, Mode: " << ModeToStr(mode_);

    for (auto iter : all_node_vec_) {
        if (iter->GetType() == NodeType::Variable) {
            static_cast<Variable*>(iter)->UpdateRegressiveIdx();
        }
    }

    try {
        allocator_ptr_->Free(tmp_buff_);
    }
    catch (...) {
        LI_THROW(ResourceExhausted("free temporary buffer {} failed!", tmp_buff_));
    }

    mem_manager_ptr_->CalculateBuffer();

    is_context_built_ = true;

    Synchronize();

    VLOG(1) << "Finish context build success, Mode: " << ModeToStr(mode_);
}

int Context::CreateGlobalContext(const std::string& context_name, Mode mode, const int device_id)
{
    global_context_id_++;
    std::shared_ptr<Context> context_ptr = std::make_shared<Context>(context_name, mode, device_id);
    global_context_ptr_                  = context_ptr;
    if (global_contexts_map_.find(context_name) != global_contexts_map_.end()) {
        LOG(ERROR) << "Error occured! context_id " << context_name << " already exists!";
        exit(-1);
    }
    global_contexts_map_.emplace(context_name, context_ptr);
    LOG(INFO) << "create global context success" << "context name: " << context_name
              << "context_id:" << global_context_id_;
    return global_context_id_;
}

void Context::SetGlobalContext(const std::string& context_name)
{
    auto iter = global_contexts_map_.find(context_name);
    if (iter == global_contexts_map_.end()) {
        LOG(ERROR) << "Error occured! context_id " << context_name << " does not exist!";
        exit(-1);
    }

    global_context_ptr_ = iter->second;
}

std::shared_ptr<Context> Context::GetGlobalInstance()
{
    return global_context_ptr_;
}

/*-----------------Auto Regression model--------------------------------*/

// During the model network construction process, mark the start and end
// positions of the autoregressive part.
void Context::SetBeginRegress()
{
    in_regress_ = true;
}
void Context::SetEndRegress()
{
    in_regress_ = false;
}

// Get the start and end timestamps of the autoregressive part of the network
// structure.
int Context::GetBeginRegressIdx() const
{
    return begin_regress_idx_;
}

int Context::GetEndRegressIdx() const
{
    return end_regress_idx_;
}

void Context::UpdateBeginRegressIdx(int node_idx)
{
    if (node_idx < 0) {
        LOG(ERROR) << "Error! UpdateBeginRegressIdx with node_idx" << node_idx;
        exit(-1);
    }
    begin_regress_idx_ = (begin_regress_idx_ == -1) ? node_idx : std::min(node_idx, begin_regress_idx_);
}

void Context::UpdateEndRegressIdx(int node_idx)
{
    if (node_idx_ < 0) {
        LOG(ERROR) << "Error! UpdateEndRegressIdx with node_idx" << node_idx;
        exit(-1);
    }
    end_regress_idx_ = (end_regress_idx_ == -1) ? node_idx : std::max(node_idx, end_regress_idx_);
}

bool Context::GetRegressStatus() const
{
    return in_regress_;
}

// Debug
void Context::Synchronize()
{
    GpuStreamSync(GetStream());
}

int                                                       Context::global_context_id_   = 0;
std::shared_ptr<Context>                                  Context::global_context_ptr_  = nullptr;
std::unordered_map<std::string, std::shared_ptr<Context>> Context::global_contexts_map_ = {};

}  // namespace flashck
