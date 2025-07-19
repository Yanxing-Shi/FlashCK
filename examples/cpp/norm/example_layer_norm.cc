#include <iostream>
#include <memory>
#include <random>
#include <vector>

// Include the LayerNorm header-only wrapper
#include "flashck/wrapper/cpp/norm/layer_norm.h"

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

        // Generate random input data
        std::mt19937                          gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        auto input = std::make_unique<float[]>(batch_size * hidden_dim);
        auto gamma = std::make_unique<float[]>(hidden_dim);
        auto beta  = std::make_unique<float[]>(hidden_dim);

        // Initialize data
        for (int i = 0; i < batch_size * hidden_dim; ++i) {
            input[i] = dist(gen);
        }

        for (int i = 0; i < hidden_dim; ++i) {
            gamma[i] = 1.0f + dist(gen) * 0.1f;  // Around 1.0
            beta[i]  = dist(gen) * 0.1f;         // Around 0.0
        }

        std::cout << "Input data generated successfully.\n";

        // Execute LayerNorm
        std::cout << "Executing LayerNorm...\n";

        float* output = flashck::layer_norm_fwd(input.get(), gamma.get(), beta.get(), batch_size, hidden_dim, epsilon);

        if (output) {
            std::cout << "LayerNorm executed successfully!\n";

            // Print some sample values
            std::cout << "\nSample input values: ";
            for (int i = 0; i < 5; ++i) {
                std::cout << input[i] << " ";
            }

            std::cout << "\nSample output values: ";
            for (int i = 0; i < 5; ++i) {
                std::cout << output[i] << " ";
            }
            std::cout << "\n";
        }
        else {
            std::cerr << "LayerNorm execution failed!\n";
            return 1;
        }

        std::cout << "\n✓ Example completed successfully!\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

/*
 * Compilation instructions:
 *
 * 1. Using CMake (recommended):
 *    mkdir build && cd build
 *    cmake .. -DBUILD_TESTS=ON
 *    make -j$(nproc)
 *    ./example_usage
 *
 * 2. Direct compilation:
 *    g++ -std=c++20 -I/path/to/flashck -I/path/to/flashck/3rdparty/composable_kernel/include \
 *        example_usage.cpp -lflashck_static -o example_usage
 *    ./example_usage
 */
