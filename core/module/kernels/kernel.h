#pragma once

#include "core/profiling/profiling_engine.h"
#include "core/utils/common.h"

#include "core/module/kernels/norm_kernels/norm_kernel_args.h"
#include "core/module/kernels/gemm_kernels/gemm_kernel_args.h"
#include "core/module/kernels/fmha_kernels/fmha_kernel_args.h"


#include "core/module/kernels/norm_kernels/norm_common_kernel.h"
#include "core/module/kernels/norm_kernels/norm_kernel_call_def.h"

#include "core/profiling/tile/norm/norm_codegen.h"
#include "core/profiling/tile/gemm/gemm_codegen.h"
#include "core/profiling/tile/fmha/fmha_fwd_codegen.h"
#include "core/profiling/tile/fmha/fmha_fwd_append_kv_codegen.h"
#include "core/profiling/tile/fmha/fmha_fwd_split_kv_codegen.h"
#include "core/profiling/tile/fmha/fmha_fwd_split_kv_combine_codegen.h"
#include "core/profiling/tile/fmha/fmha_batch_prefill_codegen.h"
#include "core/profiling/tile/fmha/fmha_paged_kv_prefill_codegen.h"




namespace flashck {

/// @brief Macro for safe kernel symbol loading with error checking
/// @param kernel_func Function pointer variable to assign
/// @param name_str String name of the symbol to load
#define LOAD_SYMBOL(kernel_func, name_str)                                                                             \
    do {                                                                                                               \
        if (!ProfilingEngine::GetInstance()->GetKernelLibrary()->has_symbol(name_str)) {                               \
            FC_THROW(Unavailable("Kernel symbol not found: {}", name_str));                                            \
        }                                                                                                              \
        kernel_func =                                                                                                  \
            ProfilingEngine::GetInstance()->GetKernelLibrary()->get_function_ptr<decltype(kernel_func)>(name_str);     \
    } while (0)

/**
 * @brief Template configuration for kernel tuning phase
 *
 * Contains template strings for various code generation aspects
 * during the kernel tuning and profiling process.
 */
struct TuningTpl {
    std::string dtype_config_tpl_;    ///< Data type configuration template
    std::string dtype_decl_tpl_;      ///< Data type declaration template
    std::string func_signature_tpl_;  ///< Function signature template
    std::string make_args_tpl_;       ///< Argument construction template
    std::string tensor_decl_tpl_;     ///< Tensor declaration template
    std::string func_call_tpl_;       ///< Function call template

    TuningTpl() = default;

    /// @brief Constructor from template vector
    /// @param templates Vector of template strings (expects at least 6 elements)
    TuningTpl(const std::vector<std::string>& templates)
    {
        if (templates.size() >= 6) {
            dtype_config_tpl_   = templates[0];
            dtype_decl_tpl_     = templates[1];
            func_signature_tpl_ = templates[2];
            make_args_tpl_      = templates[3];
            tensor_decl_tpl_    = templates[4];
            func_call_tpl_      = templates[5];
        }
    }
};

/**
 * @brief Template configuration for kernel runtime execution
 *
 * Contains template strings for code generation during the
 * actual kernel execution phase (subset of TuningTpl).
 */
struct RunningTpl {
    std::string dtype_config_tpl_;    ///< Data type configuration template
    std::string dtype_decl_tpl_;      ///< Data type declaration template
    std::string func_signature_tpl_;  ///< Function signature template
    std::string make_args_tpl_;       ///< Argument construction template

    RunningTpl() = default;

    /// @brief Constructor from template vector
    /// @param templates Vector of template strings (expects at least 4 elements)
    RunningTpl(const std::vector<std::string>& templates)
    {
        if (templates.size() >= 4) {
            dtype_config_tpl_   = templates[0];
            dtype_decl_tpl_     = templates[1];
            func_signature_tpl_ = templates[2];
            make_args_tpl_      = templates[3];
        }
    }
};

/**
 * @brief Abstract base class for all kernel implementations
 *
 * Provides the interface for kernel code generation, profiling, and execution.
 * All concrete kernel implementations must inherit from this class.
 */
class Kernel {
public:
    /// @brief Type alias for kernel argument variants
    using KernelArgs_t = std::variant<NormKernelArgs, GemmKernelArgs, FmhaKernelArgs>;

    /// @brief Type alias for code generation map
    using norm_codegen_map_t = std::map<std::string, NormCodeGen>;
    using gemm_codegen_map_t = std::map<std::string, GemmCodeGen>;
    using fmha_fwd_codegen_map_t = std::map<std::string, FmhaFwdCodeGen>;
    using fmha_fwd_append_kv_codegen_map_t = std::map<std::string, FmhaFwdAppendKVCodeGen>;
    using fmha_fwd_split_kv_codegen_map_t = std::map<std::string, FmhaFwdSplitKVCodeGen>;
    using fmha_fwd_split_kv_combine_codegen_map_t = std::map<std::string, FmhaFwdSplitKVCombineCodeGen>;

    /// @brief Type alias for instance map variants
    using instance_map_t = std::variant<norm_codegen_map_t>;

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
                     const std::string&    kind_name,
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