#include "core/utils/init.h"

#include <mutex>
#include <stdexcept>

namespace flashck {

// Thread-safe initialization flags
static std::once_flag g_glog_init_flag;
static std::once_flag g_gflags_init_flag;

void InitGLOG(char** argv)
{
    // Validate input parameters
    if (!argv || !argv[0]) {
        throw std::invalid_argument("InitGLOG: argv must be valid and argv[0] must contain program name");
    }

    // Thread-safe initialization using std::call_once
    std::call_once(g_glog_init_flag, [argv]() {
        try {
            // Configure glog for console output by default
            FLAGS_logtostderr = true;

            // Initialize Google logging with program name
            google::InitGoogleLogging(argv[0]);

            // Optional: Set additional logging configuration
            FLAGS_colorlogtostderr          = true;   // Enable colored output if supported
            FLAGS_timestamp_in_logfile_name = false;  // Cleaner log file names
        }
        catch (const std::exception& e) {
            // Handle initialization errors gracefully
            throw std::runtime_error("Failed to initialize Google logging: " + std::string(e.what()));
        }
    });
}

void InitGflags(int argc, char** argv, bool remove_flags)
{
    // Validate input parameters
    if (argc < 1 || !argv || !argv[0]) {
        throw std::invalid_argument("InitGflags: argc must be >= 1 and argv must be valid");
    }

    // Thread-safe initialization using std::call_once
    std::call_once(g_gflags_init_flag, [argc, argv, remove_flags]() mutable {
        try {
            // Parse command-line flags
            google::ParseCommandLineFlags(&argc, &argv, remove_flags);
        }
        catch (const std::exception& e) {
            // Handle flag parsing errors gracefully
            throw std::runtime_error("Failed to parse command-line flags: " + std::string(e.what()));
        }
    });
}

void InitAll(int argc, char** argv, bool remove_flags)
{
    // Validate input parameters once for both functions
    if (argc < 1 || !argv || !argv[0]) {
        throw std::invalid_argument("InitAll: argc must be >= 1 and argv must be valid");
    }

    // Initialize glog first (may be needed for flag parsing error messages)
    InitGLOG(argv);

    // Then initialize gflags
    InitGflags(argc, argv, remove_flags);
}

}  // namespace flashck
