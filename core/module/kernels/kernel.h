#pragma once

#include "core/profiling/profiling_engine.h"
#include "core/utils/common.h"

#include "core/module/kernels/norm_kernels/layer_norm/layer_norm_call.h"
#include "core/module/kernels/norm_kernels/rms_norm/rms_norm_call.h"

#include "core/module/kernels/attention_kernels/fmha_fwd/fmha_fwd_call.h"
#include "core/module/kernels/attention_kernels/fmha_fwd_append_kv/fmha_fwd_append_kv_call.h"
#include "core/module/kernels/attention_kernels/fmha_fwd_split_kv/fmha_fwd_split_kv_call.h"
#include "core/module/kernels/attention_kernels/fmha_fwd_split_kv_combine/fmha_fwd_split_kv_combine_call.h"

#include "core/module/kernels/gemm_kernels/gemm/gemm_call.h"
#include "core/module/kernels/gemm_kernels/gemm_multi_d/gemm_multi_d_call.h"
#include "core/module/kernels/gemm_kernels/flatmm/flatmm_call.h"
#include "core/module/kernels/gemm_kernels/batch_gemm/batch_gemm_call.h"
#include "core/module/kernels/gemm_kernels/group_gemm/group_gemm_call.h"


#include "core/profiling/norm/norm_codegen.h"
#include "core/profiling/gemm/gemm_codegen.h"
#include "core/profiling/attention/fmha_fwd/fmha_fwd_codegen.h"
#include "core/profiling/attention/fmha_fwd_append_kv/fmha_fwd_append_kv_codegen.h"
#include "core/profiling/attention/fmha_fwd_split_kv/fmha_fwd_split_kv_codegen.h"
#include "core/profiling/attention/fmha_fwd_split_kv_combine/fmha_fwd_split_kv_combine_codegen.h"

#include "core/module/kernels/kernel.h"

namespace flashck {


/**
 * @brief Abstract base class for all kernel implementations
 *
 * Provides the interface for kernel code generation, profiling, and execution.
 * All concrete kernel implementations must inherit from this class.
 */
class Kernel {
public:
    /// @brief Type alias for kernel argument variants
    using KernelArgs_t = std::variant<LayerNormKernelArgs, RmsNormKernelArgs, legacy::GemmKernelArgs, FmhaKernelArgs>;

    /// @brief Type alias for code generation map
    using norm_codegen_map_t = std::map<std::string, LayerNormCodeGen>;
    using gemm_codegen_map_t = std::map<std::string, GemmCodeGen>;
    using fmha_fwd_codegen_map_t = std::map<std::string, FmhaFwdCodeGen>;
    using fmha_fwd_append_kv_codegen_map_t = std::map<std::string, FmhaFwdAppendKVCodeGen>;
    using fmha_fwd_split_kv_codegen_map_t = std::map<std::string, FmhaFwdSplitKVCodeGen>;
    using fmha_fwd_split_kv_combine_codegen_map_t = std::map<std::string, FmhaFwdSplitKVCombineCodeGen>;

    /// @brief Type alias for instance map variants
    using instance_map_t = std::variant<norm_codegen_map_t, legacy_gemm_codegen_map_t, tile_gemm_codegen_map_t, fmha_fwd_codegen_map_t, fmha_fwd_append_kv_codegen_map_t, fmha_fwd_split_kv_codegen_map_t, fmha_fwd_split_kv_combine_codegen_map_t>;

    Kernel()          = default;
    virtual ~Kernel() = default;

    /// @brief Generate code for kernel tuning phase
    /// @param model_name Name of the model being tuned
    /// @param kind_name Kind/type identifier of the kernel
    /// @param instance_map Map of kernel instances and configurations
    /// @param folder_name Output folder for generated code
    /// @return Vector of tuples containing source and object file paths
    /// @throws Unimplemented in base class (must be overridden)
    virtual std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    CodeGenForTuning(const std::string&    model_name,
                     const instance_map_t& instance_map,
                     const std::string&    folder_name = "kernel_profile")
    {
        FC_THROW(Unimplemented("Kernel base CodeGenForTuning is not implemented."));
    }

    /// @brief Generate code for kernel runtime execution
    /// @param func_name Function name for the generated kernel
    /// @param model_name Name of the model
    /// @param running_infos Runtime configuration information
    /// @param instance_map Map of kernel instances and configurations
    /// @param folder_name Output folder for generated code
    /// @return Generated source code as string
    /// @throws Unimplemented in base class (must be overridden)
    virtual std::string CodeGenForRunning(const std::string&                        func_name,
                                          const std::string&                        model_name,
                                          const std::map<std::string, RunningItem>& running_infos,
                                          const instance_map_t&                     instance_map,
                                          const std::string&                        folder_name = "kernel_profile")
    {
        FC_THROW(Unimplemented("Kernel base CodeGenForRunning is not implemented."));
    }

    /// @brief Launch/execute the kernel with given arguments
    /// @param kernel_func_name Name of the kernel function to launch
    /// @param args Kernel arguments (variant type)
    /// @throws Unimplemented in base class (must be overridden)
    virtual void KernelLauncher(const std::string& kernel_func_name, const KernelArgs_t& args)
    {
        FC_THROW(Unimplemented("Kernel base KernelLauncher is not implemented."));
    }
};

}  // namespace flashck