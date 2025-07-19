#pragma once

/**
 * @file flashck_wrapper.h
 * @brief Header-only C++ wrapper for FlashCK operations
 *
 * This header provides convenient C++ interfaces for FlashCK operations
 * including layer normalization, GEMM, and FMHA operations.
 */

#include "gemm/linear.h"
#include "norm/layer_norm.h"
// #include "fmha/attention.h"  // To be implemented

/**
 * @namespace flashck
 * @brief Main namespace for FlashCK operations
 */
namespace flashck {

/**
 * @namespace flashck::wrapper
 * @brief Namespace for FlashCK C++ wrapper functions
 */
namespace wrapper {

/**
 * @brief Check if FlashCK is available for the current system
 * @return true if FlashCK is available, false otherwise
 */
inline bool is_available()
{
    // This could be extended to check for GPU availability, etc.
    return true;
}

/**
 * @brief Get FlashCK version information
 * @return Version string
 */
inline const char* version()
{
    return "1.0.0";  // This should be dynamically generated
}

}  // namespace wrapper
}  // namespace flashck

/**
 * @def FLASHCK_WRAPPER_VERSION_MAJOR
 * @brief Major version number
 */
#define FLASHCK_WRAPPER_VERSION_MAJOR 1

/**
 * @def FLASHCK_WRAPPER_VERSION_MINOR
 * @brief Minor version number
 */
#define FLASHCK_WRAPPER_VERSION_MINOR 0

/**
 * @def FLASHCK_WRAPPER_VERSION_PATCH
 * @brief Patch version number
 */
#define FLASHCK_WRAPPER_VERSION_PATCH 0
