#include "lightinfer/core/module/kernels/embedding_kernels/embedding_kernel.h"
#include "lightinfer/core/utils/rocm_info.h"

namespace lightinfer {

std::map<std::string, std::shared_ptr<void>> EmbeddingKernel::Init(const OperationKind&   op_kind,
                                                                   const TensorOperation& extra_kind)
{
    return ExtractConfig(std::get<EmbeddingOperationKind>(op_kind), extra_kind);
}

std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
EmbeddingKernel::GenKernelProfiler(const std::string&                               model_name,
                                   const std::unordered_map<std::string, std::any>& kernel_func_map,
                                   const std::string&                               folder_name)
{
    return GenCommonKernelProfiler(model_name, kernel_func_map, "embedding");
}

std::string EmbeddingKernel::GenKernelFunction(const std::string&                               func_name,
                                               const std::string&                               model_name,
                                               const std::unordered_map<std::string, std::any>& kernel_func_map)
{
    return GenCommonKernelFunction(func_name, kernel_func_map, "embedding");
}

void EmbeddingKernel::KernelLauncher(const std::string& kernel_func_name, const KernelArgs& args)
{
    decltype(&Embedding) kernel_func = nullptr;

    LOAD_SYMBOL(kernel_func, kernel_func_name);

    auto embedding_kernel_args = std::get<EmbeddingKernelArgs>(args);

    kernel_func(embedding_kernel_args.y_ptr_,
                embedding_kernel_args.emb_x0_ptr_,
                embedding_kernel_args.index_x0_ptr_,
                embedding_kernel_args.gamma_ptr_,
                embedding_kernel_args.beta_ptr_,
                embedding_kernel_args.embedding_dims_,
                embedding_kernel_args.num_indices_,
                embedding_kernel_args.epsilon_,
                embedding_kernel_args.stream_);
}

}  // namespace lightinfer