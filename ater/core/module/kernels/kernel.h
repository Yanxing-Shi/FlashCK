#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ater/core/profiler/base.h"
#include "ater/core/utils/flags.h"
#include "ater/core/utils/memory_utils.h"

ATER_DECLARE_string(ATER_HOME_PATH);

// extern declare the function since hip/hip_ext.h header is broken
extern hipError_t hipExtModuleLaunchKernel(hipFunction_t,  // NOLINT
                                           uint32_t,
                                           uint32_t,
                                           uint32_t,
                                           uint32_t,
                                           uint32_t,
                                           uint32_t,
                                           size_t,
                                           hipStream_t,
                                           void**,
                                           void**,
                                           hipEvent_t = nullptr,
                                           hipEvent_t = nullptr,
                                           uint32_t   = 0);

namespace ater {

struct kernel_argument {
    template<class T,
             class U = std::remove_reference_t<T>,
             class   = std::enable_if_t<not std::is_base_of<kernel_argument, T>{}>>
    kernel_argument(T&& x): size(sizeof(U)), align(alignof(U)), data(&x)  // NOLINT
    {
    }
    std::size_t size;
    std::size_t align;
    void*       data;
};

inline std::vector<char> PackArgs(const std::vector<kernel_argument>& args)
{
    std::vector<char> kernargs;
    for (auto&& arg : args) {
        std::size_t n = arg.size;
        const auto* p = static_cast<const char*>(arg.data);
        // Insert padding
        std::size_t padding = (arg.align - (kernargs.size() % arg.align)) % arg.align;
        kernargs.insert(kernargs.end(), padding, 0);
        kernargs.insert(kernargs.end(), p, p + n);
    }
    return kernargs;
}

class Kernel {
public:
    Kernel() = default;

    template<class... Ts>
    auto LaunchKernel(const std::string& kernel_func, size_t global, size_t local, hipStream_t stream, Ts... zs) const
    {
        return [=](auto&&... xs) {
            LaunchKernel(kernel_func, global, local, std::vector<kernel_argument>{xs...}, stream, zs...);
        };
    }

    void LaunchKernel(std::string                         kernel_func,
                      size_t                              global,
                      size_t                              local,
                      const std::vector<kernel_argument>& args,
                      hipStream_t                         stream       = nullptr,
                      const std::string                   context_name = "",
                      const std::string&                  folder_name  = "kernel_profile") const
    {

        std::filesystem::path build_dir = std::filesystem::path(FLAGS_ATER_HOME_PATH) / folder_name / context_name;
        std::vector<char>     buffer    = ReadBuffer((build_dir / (kernel_func + ".o")).string());
        hipModule_t           buffer_data;

        ATER_ENFORCE_HIP_SUCCESS(hipModuleLoadData(&buffer_data, buffer.data()));

        hipFunction_t kernel;
        ATER_ENFORCE_HIP_SUCCESS(hipModuleGetFunction(&kernel, buffer_data, kernel_func.c_str()));

        std::vector<char> kernel_args = PackArgs(args);
        std::size_t       size        = kernel_args.size();

        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                          kernel_args.data(),
                          HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &size,
                          HIP_LAUNCH_PARAM_END};
        ATER_ENFORCE_HIP_SUCCESS(hipExtModuleLaunchKernel(kernel,
                                                          global,
                                                          1,
                                                          1,
                                                          local,
                                                          1,
                                                          1,
                                                          0,
                                                          stream,
                                                          nullptr,
                                                          reinterpret_cast<void**>(&config),
                                                          nullptr,
                                                          nullptr));
        LOG(INFO) << kernel_func << "kernel launch success";
    }

    virtual ~Kernel() {}
    virtual std::map<std::string, std::shared_ptr<void>> Init() = 0;
    virtual std::vector<std::tuple<std::filesystem::path, std::filesystem::path>>
    GenKernelProfiler(const std::string&                                                  kernel_name,
                      const std::string&                                                  model_name,
                      const std::map<std::string, std::shared_ptr<void>>&                 kernel_instance_map,
                      const std::map<std::string, std::vector<std::shared_ptr<DimInfo>>>& dim_info_map) = 0;
    virtual std::string
    GenKernelFunction(const std::string&                                                  func_name,
                      const std::map<std::string, std::shared_ptr<void>>&                 kernel_instance_map,
                      const std::map<std::string, std::shared_ptr<ExecItem>>&             exec_path,
                      const std::string                                                   permute_shape,
                      const std::map<std::string, std::vector<std::shared_ptr<DimInfo>>>& dim_info_map) = 0;
    // virtual void GenFunctionDecl() const    = 0;
    // virtual void GenFunctionCall() const    = 0;
    virtual bool FunctionFilter() = 0;
};

}  // namespace ater