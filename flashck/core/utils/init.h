#pragma once

#include "gflags/gflags.h"
#include <glog/logging.h>

namespace flashck {

/*!
 * @brief Initialize Google logging (glog) with thread-safe guarantee
 * @param argv Command-line arguments (argv[0] is used as program name)
 * @note This function can be called multiple times safely
 * @note Sets FLAGS_logtostderr = true by default for console output
 * @warning argv must be valid and argv[0] must contain the program name
 */
void InitGLOG(char** argv);

/*!
 * @brief Initialize Google command-line flags (gflags) with thread-safe guarantee
 * @param argc Number of command-line arguments
 * @param argv Command-line arguments array
 * @param remove_flags Whether to remove parsed flags from argv (default: true)
 * @note This function can be called multiple times safely
 * @note Parses command-line flags and makes them available via FLAGS_* variables
 * @warning argc and argv must be valid command-line arguments
 */
void InitGflags(int argc, char** argv, bool remove_flags = true);

/*!
 * @brief Convenience function to initialize both glog and gflags
 * @param argc Number of command-line arguments
 * @param argv Command-line arguments array
 * @param remove_flags Whether to remove parsed flags from argv (default: true)
 * @note Initializes glog first, then gflags for proper order
 * @note Thread-safe and can be called multiple times
 */
void InitAll(int argc, char** argv, bool remove_flags = true);

}  // namespace flashck