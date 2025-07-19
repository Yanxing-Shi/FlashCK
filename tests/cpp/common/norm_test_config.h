/**
 * @file norm_test_config.h
 * @brief Configuration classes for normalization operation tests using the unified framework
 */

#pragma once

#include "test_framework.h"
#include <sstream>

namespace flashck {
namespace test {

/**
 * @brief Configuration for LayerNorm tests implementing the unified interface
 */
template<typename T>
class LayerNormConfig: public OpConfigBase<T> {
public:
    LayerNormConfig(int m, int n, float epsilon = 1e-5f, const std::string& name = ""): m_(m), n_(n), epsilon_(epsilon)
    {
        if (name.empty()) {
            name_ = "LayerNorm_" + std::to_string(m) + "x" + std::to_string(n);
        }
        else {
            name_ = name;
        }

        // Calculate sizes
        input_size_  = m * n;
        gamma_size_  = n;
        beta_size_   = n;
        output_size_ = m * n;

        // Allocate CPU memory for input data
        input_data_ = std::make_unique<T[]>(input_size_);
        gamma_data_ = std::make_unique<T[]>(gamma_size_);
        beta_data_  = std::make_unique<T[]>(beta_size_);

        // Initialize GPU pointers to null
        gpu_input_  = nullptr;
        gpu_gamma_  = nullptr;
        gpu_beta_   = nullptr;
        gpu_output_ = nullptr;
    }

    // OpConfigBase interface implementation
    std::string name() const override
    {
        return name_;
    }
    std::string operation_type() const override
    {
        return "LayerNorm";
    }
    size_t output_size() const override
    {
        return output_size_;
    }

    size_t total_bytes() const override
    {
        // Read: input + gamma + beta, Write: output
        return (input_size_ + gamma_size_ + beta_size_ + output_size_) * sizeof(T);
    }

    void init_test_data(DataGenerator<T>& data_gen) override
    {
        data_gen.uniform(input_data_.get(), input_size_, -2.0f, 2.0f);
        data_gen.uniform(gamma_data_.get(), gamma_size_, 0.5f, 1.5f);
        data_gen.uniform(beta_data_.get(), beta_size_, -0.5f, 0.5f);
    }

    void setup_gpu_inputs(GpuMemoryManager<T>& gpu_mem) override
    {
        gpu_input_  = gpu_mem.allocate_and_copy(input_data_.get(), input_size_, "input");
        gpu_gamma_  = gpu_mem.allocate_and_copy(gamma_data_.get(), gamma_size_, "gamma");
        gpu_beta_   = gpu_mem.allocate_and_copy(beta_data_.get(), beta_size_, "beta");
        gpu_output_ = gpu_mem.allocate(output_size_, "output");
    }

    T* get_gpu_output(GpuMemoryManager<T>& gpu_mem) override
    {
        return gpu_output_;
    }

    void get_cpu_inputs_for_reference(std::vector<const T*>& inputs) const override
    {
        inputs.clear();
        inputs.push_back(input_data_.get());
        inputs.push_back(gamma_data_.get());
        inputs.push_back(beta_data_.get());
    }

    // LayerNorm-specific getters
    int m() const
    {
        return m_;
    }
    int n() const
    {
        return n_;
    }
    float epsilon() const
    {
        return epsilon_;
    }

    size_t input_size() const
    {
        return input_size_;
    }
    size_t gamma_size() const
    {
        return gamma_size_;
    }
    size_t beta_size() const
    {
        return beta_size_;
    }

    // GPU data access for FlashCK wrapper
    T* gpu_input() const
    {
        return gpu_input_;
    }
    T* gpu_gamma() const
    {
        return gpu_gamma_;
    }
    T* gpu_beta() const
    {
        return gpu_beta_;
    }
    T* gpu_output() const
    {
        return gpu_output_;
    }

    // CPU data access
    const T* input_data() const
    {
        return input_data_.get();
    }
    const T* gamma_data() const
    {
        return gamma_data_.get();
    }
    const T* beta_data() const
    {
        return beta_data_.get();
    }

private:
    int         m_, n_;
    float       epsilon_;
    std::string name_;

    size_t               input_size_, gamma_size_, beta_size_, output_size_;
    std::unique_ptr<T[]> input_data_;
    std::unique_ptr<T[]> gamma_data_;
    std::unique_ptr<T[]> beta_data_;

    // GPU pointers (managed by GpuMemoryManager)
    T* gpu_input_;
    T* gpu_gamma_;
    T* gpu_beta_;
    T* gpu_output_;
};

/**
 * @brief Configuration for RMSNorm tests implementing the unified interface
 */
template<typename T>
class RMSNormConfig: public OpConfigBase<T> {
public:
    RMSNormConfig(int m, int n, float epsilon = 1e-5f, const std::string& name = ""): m_(m), n_(n), epsilon_(epsilon)
    {
        if (name.empty()) {
            name_ = "RMSNorm_" + std::to_string(m) + "x" + std::to_string(n);
        }
        else {
            name_ = name;
        }

        // Calculate sizes
        input_size_  = m * n;
        gamma_size_  = n;
        output_size_ = m * n;

        // Allocate CPU memory for input data
        input_data_ = std::make_unique<T[]>(input_size_);
        gamma_data_ = std::make_unique<T[]>(gamma_size_);

        // Initialize GPU pointers to null
        gpu_input_  = nullptr;
        gpu_gamma_  = nullptr;
        gpu_output_ = nullptr;
    }

    // OpConfigBase interface implementation
    std::string name() const override
    {
        return name_;
    }
    std::string operation_type() const override
    {
        return "RMSNorm";
    }
    size_t output_size() const override
    {
        return output_size_;
    }

    size_t total_bytes() const override
    {
        // Read: input + gamma, Write: output
        return (input_size_ + gamma_size_ + output_size_) * sizeof(T);
    }

    void init_test_data(DataGenerator<T>& data_gen) override
    {
        data_gen.uniform(input_data_.get(), input_size_, -2.0f, 2.0f);
        data_gen.uniform(gamma_data_.get(), gamma_size_, 0.5f, 1.5f);
    }

    void setup_gpu_inputs(GpuMemoryManager<T>& gpu_mem) override
    {
        gpu_input_  = gpu_mem.allocate_and_copy(input_data_.get(), input_size_, "input");
        gpu_gamma_  = gpu_mem.allocate_and_copy(gamma_data_.get(), gamma_size_, "gamma");
        gpu_output_ = gpu_mem.allocate(output_size_, "output");
    }

    T* get_gpu_output(GpuMemoryManager<T>& gpu_mem) override
    {
        return gpu_output_;
    }

    void get_cpu_inputs_for_reference(std::vector<const T*>& inputs) const override
    {
        inputs.clear();
        inputs.push_back(input_data_.get());
        inputs.push_back(gamma_data_.get());
    }

    // RMSNorm-specific getters
    int m() const
    {
        return m_;
    }
    int n() const
    {
        return n_;
    }
    float epsilon() const
    {
        return epsilon_;
    }

    size_t input_size() const
    {
        return input_size_;
    }
    size_t gamma_size() const
    {
        return gamma_size_;
    }

    // GPU data access for FlashCK wrapper
    T* gpu_input() const
    {
        return gpu_input_;
    }
    T* gpu_gamma() const
    {
        return gpu_gamma_;
    }
    T* gpu_output() const
    {
        return gpu_output_;
    }

    // CPU data access
    const T* input_data() const
    {
        return input_data_.get();
    }
    const T* gamma_data() const
    {
        return gamma_data_.get();
    }

private:
    int         m_, n_;
    float       epsilon_;
    std::string name_;

    size_t               input_size_, gamma_size_, output_size_;
    std::unique_ptr<T[]> input_data_;
    std::unique_ptr<T[]> gamma_data_;

    // GPU pointers (managed by GpuMemoryManager)
    T* gpu_input_;
    T* gpu_gamma_;
    T* gpu_output_;
};

/**
 * @brief Factory functions for creating test configurations
 */
template<typename T>
class TestConfigFactory {
public:
    /**
     * @brief Create standard LayerNorm test configurations
     */
    static std::vector<std::shared_ptr<LayerNormConfig<T>>> create_layernorm_configs()
    {
        std::vector<std::shared_ptr<LayerNormConfig<T>>> configs;

        // // Standard sizes
        // configs.push_back(std::make_shared<LayerNormConfig<T>>(32, 768, 1e-5f, "Small"));
        // configs.push_back(std::make_shared<LayerNormConfig<T>>(128, 1024, 1e-5f, "Medium"));
        // configs.push_back(std::make_shared<LayerNormConfig<T>>(256, 2048, 1e-5f, "Large"));

        // // Edge cases
        // configs.push_back(std::make_shared<LayerNormConfig<T>>(1, 1, 1e-5f, "Minimal"));
        // configs.push_back(std::make_shared<LayerNormConfig<T>>(1, 4096, 1e-5f, "Single_Row"));
        // configs.push_back(std::make_shared<LayerNormConfig<T>>(512, 64, 1e-5f, "Many_Rows"));

        return configs;
    }

    /**
     * @brief Create standard RMSNorm test configurations
     */
    static std::vector<std::shared_ptr<RMSNormConfig<T>>> create_rmsnorm_configs()
    {
        std::vector<std::shared_ptr<RMSNormConfig<T>>> configs;

        // Standard sizes
        configs.push_back(std::make_shared<RMSNormConfig<T>>(32, 768, 1e-5f, "Small"));
        configs.push_back(std::make_shared<RMSNormConfig<T>>(128, 1024, 1e-5f, "Medium"));
        configs.push_back(std::make_shared<RMSNormConfig<T>>(256, 2048, 1e-5f, "Large"));

        // Edge cases
        configs.push_back(std::make_shared<RMSNormConfig<T>>(1, 1, 1e-5f, "Minimal"));
        configs.push_back(std::make_shared<RMSNormConfig<T>>(1, 4096, 1e-5f, "Single_Row"));
        configs.push_back(std::make_shared<RMSNormConfig<T>>(512, 64, 1e-5f, "Many_Rows"));

        return configs;
    }
};

}  // namespace test
}  // namespace flashck
