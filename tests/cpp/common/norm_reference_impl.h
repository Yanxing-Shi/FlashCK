/**
 * @file norm_reference_impl.h
 * @brief Reference CPU implementations for normalization operations (unified interface)
 */

#pragma once

#include "norm_test_config.h"
#include <cmath>

namespace flashck {
namespace test {

/**
 * @brief Reference CPU implementation of LayerNorm using unified interface
 */
template<typename T>
class LayerNormReference {
public:
    /**
     * @brief Forward pass implementation compatible with UnifiedTestSuite
     * @param config LayerNorm configuration
     * @param output Output buffer (pre-allocated)
     */
    static void forward(const LayerNormConfig<T>& config, T* output)
    {
        const int   m       = config.m();
        const int   n       = config.n();
        const float epsilon = config.epsilon();

        const T* input = config.input_data();
        const T* gamma = config.gamma_data();
        const T* beta  = config.beta_data();

        for (int i = 0; i < m; ++i) {
            const T* row     = input + i * n;
            T*       out_row = output + i * n;

            // Compute mean
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                sum += static_cast<double>(row[j]);
            }
            double mean = sum / n;

            // Compute variance
            double var_sum = 0.0;
            for (int j = 0; j < n; ++j) {
                double diff = static_cast<double>(row[j]) - mean;
                var_sum += diff * diff;
            }
            double variance = var_sum / n;
            double inv_std  = 1.0 / std::sqrt(variance + epsilon);

            // Apply normalization
            for (int j = 0; j < n; ++j) {
                double normalized = (static_cast<double>(row[j]) - mean) * inv_std;
                out_row[j] = static_cast<T>(static_cast<double>(gamma[j]) * normalized + static_cast<double>(beta[j]));
            }
        }
    }

    /**
     * @brief Legacy forward pass for backward compatibility
     */
    static void forward(const LayerNormConfig<T>& config, const T* input, T* output)
    {
        const int   m       = config.m();
        const int   n       = config.n();
        const float epsilon = config.epsilon();
        const T*    gamma   = config.gamma_data();
        const T*    beta    = config.beta_data();

        for (int i = 0; i < m; ++i) {
            const T* row     = input + i * n;
            T*       out_row = output + i * n;

            // Compute mean
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                sum += static_cast<double>(row[j]);
            }
            double mean = sum / n;

            // Compute variance
            double var_sum = 0.0;
            for (int j = 0; j < n; ++j) {
                double diff = static_cast<double>(row[j]) - mean;
                var_sum += diff * diff;
            }
            double variance = var_sum / n;
            double inv_std  = 1.0 / std::sqrt(variance + epsilon);

            // Apply normalization
            for (int j = 0; j < n; ++j) {
                double normalized = (static_cast<double>(row[j]) - mean) * inv_std;
                out_row[j] = static_cast<T>(static_cast<double>(gamma[j]) * normalized + static_cast<double>(beta[j]));
            }
        }
    }
};

/**
 * @brief Reference CPU implementation of RMSNorm using unified interface
 */
template<typename T>
class RMSNormReference {
public:
    /**
     * @brief Forward pass implementation compatible with UnifiedTestSuite
     * @param config RMSNorm configuration
     * @param output Output buffer (pre-allocated)
     */
    static void forward(const RMSNormConfig<T>& config, T* output)
    {
        const int   m       = config.m();
        const int   n       = config.n();
        const float epsilon = config.epsilon();

        const T* input = config.input_data();
        const T* gamma = config.gamma_data();

        for (int i = 0; i < m; ++i) {
            const T* row     = input + i * n;
            T*       out_row = output + i * n;

            // Compute RMS (root mean square)
            double sum_sq = 0.0;
            for (int j = 0; j < n; ++j) {
                double val = static_cast<double>(row[j]);
                sum_sq += val * val;
            }
            double rms     = std::sqrt(sum_sq / n + epsilon);
            double inv_rms = 1.0 / rms;

            // Apply normalization
            for (int j = 0; j < n; ++j) {
                double normalized = static_cast<double>(row[j]) * inv_rms;
                out_row[j]        = static_cast<T>(static_cast<double>(gamma[j]) * normalized);
            }
        }
    }

    /**
     * @brief Legacy forward pass for backward compatibility
     */
    static void forward(const RMSNormConfig<T>& config, const T* input, T* output)
    {
        const int   m       = config.m();
        const int   n       = config.n();
        const float epsilon = config.epsilon();
        const T*    gamma   = config.gamma_data();

        for (int i = 0; i < m; ++i) {
            const T* row     = input + i * n;
            T*       out_row = output + i * n;

            // Compute RMS (Root Mean Square)
            double sum_sq = 0.0;
            for (int j = 0; j < n; ++j) {
                double val = static_cast<double>(row[j]);
                sum_sq += val * val;
            }
            double rms     = std::sqrt(sum_sq / n + epsilon);
            double inv_rms = 1.0 / rms;

            // Apply normalization
            for (int j = 0; j < n; ++j) {
                double normalized = static_cast<double>(row[j]) * inv_rms;
                out_row[j]        = static_cast<T>(static_cast<double>(gamma[j]) * normalized);
            }
        }
    }
};

}  // namespace test
}  // namespace flashck
