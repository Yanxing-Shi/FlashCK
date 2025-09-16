#pragma once

#include <string>

namespace flashck{

/**
 * @brief Template configuration for kernel tuning phase
 *
 * Contains template strings for various code generation aspects
 * during the kernel tuning and profiling process.
 */
struct TuningTpl {
    std::string header_tpl_; ///< Header template
    std::string dtype_config_tpl_;    ///< Data type configuration template
    std::string dtype_decl_tpl_;      ///< Data type declaration template
    std::string func_signature_tpl_;  ///< Function signature template
    std::string make_args_tpl_;       ///< Argument construction template
    std::string tensor_decl_tpl_;     ///< Tensor declaration template
    std::string func_call_tpl_;       ///< Function call template

    // norm
    TuningTpl(std::string header_tpl, std::string dtype_config_tpl, std::string dtype_decl_tpl,
              std::string func_signature_tpl, std::string make_args_tpl, std::string tensor_decl_tpl,
              std::string func_call_tpl)
        : header_tpl_(header_tpl),
          func_signature_tpl_(func_signature_tpl),
          make_args_tpl_(make_args_tpl),
          tensor_decl_tpl_(tensor_decl_tpl),
          func_call_tpl_(func_call_tpl) {}
    
    // fmha
    TuningTpl(std::string header_tpl, std::string dtype_config_tpl, std::string dtype_decl_tpl,
              std::string func_signature_tpl, std::string make_args_tpl, std::string tensor_decl_tpl,
              std::string func_call_tpl)
        : header_tpl_(header_tpl),
          dtype_config_tpl_(dtype_config_tpl),
          dtype_decl_tpl_(dtype_decl_tpl),
          func_signature_tpl_(func_signature_tpl),
          make_args_tpl_(make_args_tpl),
          tensor_decl_tpl_(tensor_decl_tpl),
          func_call_tpl_(func_call_tpl) {}
};

/**
 * @brief Template configuration for kernel runtime execution
 *
 * Contains template strings for code generation during the
 * actual kernel execution phase.
 */
struct RunningTpl {
    std::string header_tpl_; ///< Header template
    std::string dtype_config_tpl_;    ///< Data type configuration template
    std::string dtype_decl_tpl_;      ///< Data type declaration template
    std::string func_signature_tpl_;  ///< Function signature template
    std::string make_args_tpl_;       ///< Argument construction template
};

} // namespace flashck