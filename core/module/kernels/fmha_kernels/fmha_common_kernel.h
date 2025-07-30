namespace flashck {

class FmhaCommonKernel: public Kernel {
public:
    std::map<std::string, std::shared_ptr<void>> ExtractConfig(const FmhaOperationKind& op_kind,
                                                               const TensorOperation&   extra_kind);

    std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenFmhaCommonKernelProfiler(const std::string&                               model_name,
                                const std::unordered_map<std::string, std::any>& kernel_func_map,
                                const std::string&                               create_args_source,
                                const std::string&                               args_parser_source,
                                const std::string&                               args_decl_source,
                                const std::string&                               func_signature_source,
                                const std::string&                               tensor_decl_source,
                                const std::string&                               tensor_generate_source,
                                const std::string&                               func_call_source,
                                const std::string&                               prepare_args_source,
                                const std::string&                               make_args_source,
                                const std::string&                               fmha_flag,
                                const std::string&                               folder_name = "kernel_profile");

    std::string GenFmhaCommonKernelFunction(const std::string&                               func_name,
                                            const std::string&                               model_name,
                                            const std::unordered_map<std::string, std::any>& kernel_func_map,
                                            const std::string&                               args_decl_source,
                                            const std::string&                               func_signature_source,
                                            const std::string&                               prepare_args_source,
                                            const std::string&                               make_args_source,
                                            const std::string&                               fmha_flag,
                                            const std::string& folder_name = "kernel_profile");
};
}  // namespace lightinfer