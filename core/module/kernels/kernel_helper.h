#pragma once

#include <string>

namespace flashck{

/**
 * @brief Template configuration for kernel tuning phase
 *
 * Contains template strings for various code generation aspects
 * during the kernel tuning and profiling process.
 */
struct NormTuningTpl {
    std::string header_tpl_;
    std::string dtype_config_tpl_;    ///< Data type configuration template
    std::string dtype_decl_tpl_;      ///< Data type declaration template
    std::string func_signature_tpl_;  ///< Function signature template
    std::string make_args_tpl_;       ///< Argument construction template
    std::string tensor_decl_tpl_;     ///< Tensor declaration template
    std::string func_call_tpl_;       ///< Function call template
};

struct GemmTuningTpl {
    std::string header_tpl_;
    std::string create_args_tpl_;
    std::string arg_parser_tpl_;
    std::string make_args_tpl_;       ///< Argument construction template
    std::string func_signature_tpl_;  ///< Function signature template
    std::string func_call_tpl_;       ///< Function call template
    std::string tensor_decl_tpl_;     ///< Tensor declaration template
};

struct FmhaTuningTpl {
    std::string func_signature_tpl_;  ///< Function signature template
    std::string func_call_tpl_;
    std::string prepare_args_tpl_;
    std::string make_args_tpl_;       ///< Argument construction template
    std::string tensor_decl_tpl_;     ///< Tensor declaration template
    std::string tensor_generate_tpl_; 
};

/**
 * @brief Template configuration for kernel runtime execution
 *
 * Contains template strings for code generation during the
 * actual kernel execution phase (subset of NormTuningTpl).
 */
struct NormRunningTpl {
    std::string dtype_config_tpl_;    ///< Data type configuration template
    std::string dtype_decl_tpl_;      ///< Data type declaration template
    std::string func_signature_tpl_;  ///< Function signature template
    std::string make_args_tpl_;       ///< Argument construction template
};

struct GemmRunningTpl {
    std::string header_tpl_;    ///< Data type configuration template
    std::string make_args_tpl_;      ///< Data type declaration template
    std::string func_signature_tpl_;  ///< Function signature template
};

struct FmhaRunningTpl {
    std::string args_decl_tpl_;   
    std::string func_signature_tpl_;  
    std::string prepare_args_tpl_;  
    std::string make_args_tpl_;      
};



} // namespace flashck