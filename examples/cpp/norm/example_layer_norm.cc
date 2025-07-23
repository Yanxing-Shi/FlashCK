#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <hip/hip_runtime.h>

// Include the LayerNorm header-only wrapper
#include "wrapper/cpp/norm/layer_norm.h"

// Helper function for HIP error checking
#define HIP_CHECK(call)                                                                                                \
    do {                                                                                                               \
        hipError_t error = call;                                                                                       \
        if (error != hipSuccess) {                                                                                     \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(error)           \
                      << std::endl;                                                                                    \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

int main()
{
    try {
        std::cout << "FlashCK LayerNorm Header-Only Wrapper Example\n";
        std::cout << "=============================================\n\n";

        // Configuration
        const int   batch_size = 32;
        const int   hidden_dim = 768;
        const float epsilon    = 1e-5f;

        std::cout << "Configuration:\n";
        std::cout << "  Batch size: " << batch_size << "\n";
        std::cout << "  Hidden dimension: " << hidden_dim << "\n";
        std::cout << "  Epsilon: " << epsilon << "\n\n";

        // Generate random input data on CPU first, then copy to GPU
        std::mt19937                          gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        // Allocate CPU memory for initialization
        auto input_cpu = std::make_unique<float[]>(batch_size * hidden_dim);
        auto gamma_cpu = std::make_unique<float[]>(hidden_dim);
        auto beta_cpu  = std::make_unique<float[]>(hidden_dim);

        // Initialize data on CPU
        for (int i = 0; i < batch_size * hidden_dim; ++i) {
            input_cpu[i] = dist(gen);
        }

        for (int i = 0; i < hidden_dim; ++i) {
            gamma_cpu[i] = 1.0f + dist(gen) * 0.1f;  // Around 1.0
            beta_cpu[i]  = dist(gen) * 0.1f;         // Around 0.0
        }

        // Allocate GPU memory
        float* input_gpu;
        float* gamma_gpu;
        float* beta_gpu;

        HIP_CHECK(hipMalloc(&input_gpu, batch_size * hidden_dim * sizeof(float)));
        HIP_CHECK(hipMalloc(&gamma_gpu, hidden_dim * sizeof(float)));
        HIP_CHECK(hipMalloc(&beta_gpu, hidden_dim * sizeof(float)));

        // Copy data from CPU to GPU
        HIP_CHECK(
            hipMemcpy(input_gpu, input_cpu.get(), batch_size * hidden_dim * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(gamma_gpu, gamma_cpu.get(), hidden_dim * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(beta_gpu, beta_cpu.get(), hidden_dim * sizeof(float), hipMemcpyHostToDevice));

        std::cout << "Input data generated and copied to GPU successfully.\n";

        // Execute LayerNorm
        std::cout << "Executing LayerNorm...\n";

        float* output_gpu = flashck::layer_norm_fwd(input_gpu, gamma_gpu, beta_gpu, batch_size, hidden_dim, epsilon);

        if (output_gpu) {
            std::cout << "LayerNorm executed successfully!\n";

            // Copy results back to CPU for display
            auto output_cpu = std::make_unique<float[]>(batch_size * hidden_dim);
            HIP_CHECK(hipMemcpy(
                output_cpu.get(), output_gpu, batch_size * hidden_dim * sizeof(float), hipMemcpyDeviceToHost));

            // Print some sample values
            std::cout << "\nSample input values: ";
            for (int i = 0; i < 5; ++i) {
                std::cout << input_cpu[i] << " ";
            }

            std::cout << "\nSample output values: ";
            for (int i = 0; i < 5; ++i) {
                std::cout << output_cpu[i] << " ";
            }
            std::cout << "\n";
        }
        else {
            std::cerr << "LayerNorm execution failed!\n";

            // Clean up GPU memory before returning
            hipFree(input_gpu);
            hipFree(gamma_gpu);
            hipFree(beta_gpu);
            return 1;
        }

        // Clean up GPU memory
        HIP_CHECK(hipFree(input_gpu));
        HIP_CHECK(hipFree(gamma_gpu));
        HIP_CHECK(hipFree(beta_gpu));

        std::cout << "\n✓ Example completed successfully!\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
