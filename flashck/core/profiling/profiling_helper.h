#pragma once

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>

#include "flashck/core/utils/common.h"

#include "flashck/core/profiling/legacy/gemm/gemm_problem.h"
#include "flashck/core/profiling/tile/fmha/fmha_problem.h"
#include "flashck/core/profiling/tile/norm/norm_problem.h"

// Flag declarations for tuning configuration
FC_DECLARE_int32(FC_TUNING_MODE);                ///< Tuning strategy mode (heuristic, autotuning, hybrid)
FC_DECLARE_int32(FC_TUNING_NUM_COLD_ITERATION);  ///< Number of cold iterations for warmup
FC_DECLARE_int32(FC_TUNING_NUM_REPEATS);         ///< Number of repeated measurements
FC_DECLARE_bool(FC_TUNING_GPU_TIMER);            ///< Use GPU-based timing vs CPU timing
FC_DECLARE_bool(FC_TUNING_LOG);                  ///< Enable detailed logging during tuning
FC_DECLARE_bool(FC_TUNING_FLUSH_CACHE);          ///< Flush caches between measurements
FC_DECLARE_int32(FC_TUNING_ROTATING_COUNT);      ///< Rotation count for measurement stability

namespace flashck {

/**
 * @enum Metric
 * @brief Performance metrics used for kernel evaluation and comparison
 *
 * Defines the primary metrics used to evaluate and compare kernel performance.
 * Each metric represents a different aspect of computational efficiency.
 */
enum class Metric {
    LATENCY   = 0,  ///< Execution time in milliseconds (lower is better)
    TFLOPS    = 1,  ///< Trillion floating-point operations per second (higher is better)
    BANDWIDTH = 2   ///< Memory bandwidth utilization in GB/s (higher is better)
};

/**
 * @brief Convert Metric enum to human-readable string representation
 * @param metric The metric enum value to convert
 * @return String representation of the metric
 * @throws std::invalid_argument if metric is not supported
 */
inline std::string MetricToString(Metric metric)
{
    switch (metric) {
        case Metric::LATENCY:
            return "Latency";
        case Metric::TFLOPS:
            return "TFlops";
        case Metric::BANDWIDTH:
            return "Bandwidth";
        default:
            throw std::invalid_argument("Unsupported metric type");
    }
}

/**
 * @enum CodeGenKind
 * @brief Supported code generation types for different kernel operations
 *
 * Categorizes the different types of computational kernels that can be
 * generated and profiled within the FlashCK framework.
 */
enum class CodeGenKind {
    Gemm      = 0,  ///< General Matrix Multiplication kernels
    Norm      = 1,  ///< Normalization kernels (LayerNorm, RMSNorm, etc.)
    Embedding = 2,  ///< Embedding lookup and transformation kernels
    Fmha      = 3,  ///< Fused Multi-Head Attention kernels
};

/**
 * @brief Convert CodeGenKind enum to human-readable string representation
 * @param kind The code generation kind to convert
 * @return String representation of the code generation type
 * @throws std::invalid_argument if kind is not supported
 */

inline std::string CodeGenKindToString(CodeGenKind kind)
{
    switch (kind) {
        case CodeGenKind::Gemm:
            return "Gemm";
        case CodeGenKind::Norm:
            return "Norm";
        case CodeGenKind::Embedding:
            return "Embedding";
        case CodeGenKind::Fmha:
            return "Fmha";
        default:
            throw std::invalid_argument("Unsupported CodeGenKind");
    }
}

/**
 * @enum InitMethod
 * @brief Data initialization methods for kernel input/output tensors
 *
 * Defines various initialization strategies for tensor data used in
 * profiling and testing. Different methods are suitable for different
 * types of kernels and validation requirements.
 */
enum class InitMethod {
    UniformRandomInt          = 0,  ///< Uniform random integers for discrete data
    NormalizedRandomInt       = 1,  ///< Normalized random integers with controlled distribution
    UniformRandomFloat        = 2,  ///< Uniform random floating-point values
    NormalizedRandomFloat     = 3,  ///< Normalized random floats with controlled variance
    TrigFloat                 = 4,  ///< Trigonometric patterns for numerical stability testing
    UniformFloat8Quantization = 5,  ///< Quantized float8 values for low-precision testing
};

/**
 * @brief Mapping of initialization methods to short string identifiers
 *
 * Provides concise string representations for initialization methods,
 * useful for logging, configuration, and result identification.
 */
static const std::unordered_map<InitMethod, std::string> g_init_method_short_names_map = {
    {InitMethod::UniformRandomInt, "uri"},
    {InitMethod::NormalizedRandomInt, "nri"},
    {InitMethod::UniformRandomFloat, "urf"},
    {InitMethod::NormalizedRandomFloat, "nrf"},
    {InitMethod::TrigFloat, "tf"},
    {InitMethod::UniformFloat8Quantization, "uf8q"},
};

/**
 * @class Environment
 * @brief Hardware and software environment information for profiling context
 *
 * Captures the execution environment details necessary for reproducible
 * profiling results and performance analysis. This information is crucial
 * for understanding performance variations across different systems.
 */
class Environment {
public:
    /**
     * @brief Serialize environment information to JSON format
     * @return JSON string representation of the environment
     *
     * Creates a standardized JSON representation for logging and database storage.
     */
    std::string Serialize() const
    {
        return "{\n"
               "   \"device_name\": \""
               + device_name_
               + "\",\n"
                 "   \"rocm_version\": \""
               + rocm_version_
               + "\"\n"
                 "}";
    }

    std::string device_name_;   ///< GPU device identifier (e.g., "gfx906", "gfx90a")
    std::string rocm_version_;  ///< ROCm version string for compatibility tracking
};

/**
 * @class Setting
 * @brief Configuration settings for kernel profiling and performance measurement
 *
 * Encapsulates all tuning and measurement parameters that control how
 * kernel profiling is performed. Initialized from global flags and
 * provides serialization for reproducible results.
 */
class Setting {
public:
    /**
     * @brief Constructor initializing settings from global flags
     *
     * Automatically loads configuration from command-line flags or
     * environment variables to ensure consistent profiling behavior.
     */
    Setting():
        tuning_mode_(FLAGS_FC_TUNING_MODE),
        num_cold_iterations_(FLAGS_FC_TUNING_NUM_COLD_ITERATION),
        num_repeats_(FLAGS_FC_TUNING_NUM_REPEATS),
        is_gpu_timer_(FLAGS_FC_TUNING_GPU_TIMER),
        log_(FLAGS_FC_TUNING_LOG),
        flush_cache_(FLAGS_FC_TUNING_FLUSH_CACHE),
        rotating_count_(FLAGS_FC_TUNING_ROTATING_COUNT)
    {
        // Note: Aliases for backward compatibility and clarity
        n_warmup_ = num_cold_iterations_;
        n_repeat_ = num_repeats_;
    }

    /**
     * @brief Serialize settings to JSON format
     * @return JSON string representation of all profiling settings
     *
     * Creates a complete record of profiling configuration for
     * result reproducibility and debugging.
     */
    std::string Serialize() const
    {
        std::ostringstream oss;
        oss << "{\n"
            << "   \"tuning_mode\": " << tuning_mode_ << ",\n"
            << "   \"n_warmup\": " << n_warmup_ << ",\n"
            << "   \"n_repeat\": " << n_repeat_ << ",\n"
            << "   \"is_gpu_timer\": " << (is_gpu_timer_ ? "true" : "false") << ",\n"
            << "   \"verify\": " << (verify_ ? "true" : "false") << ",\n"
            << "   \"log\": " << (log_ ? "true" : "false") << ",\n"
            << "   \"flush_cache\": " << (flush_cache_ ? "true" : "false") << ",\n"
            << "   \"rotating_count\": " << rotating_count_ << "\n"
            << "}";
        return oss.str();
    }

    // Core tuning parameters
    int  tuning_mode_;          ///< Tuning strategy mode (heuristic, autotuning, hybrid)
    int  num_cold_iterations_;  ///< Number of warmup iterations before measurement
    int  num_repeats_;          ///< Number of measurement repetitions for averaging
    bool is_gpu_timer_;         ///< Use GPU events for timing vs CPU clock

    // Measurement control flags
    bool log_;             ///< Enable detailed logging during profiling
    bool flush_cache_;     ///< Flush GPU caches between measurements
    int  rotating_count_;  ///< Number of rotations for measurement stability

    // Aliases for backward compatibility
    int n_warmup_;  ///< Alias for num_cold_iterations_
    int n_repeat_;  ///< Alias for num_repeats_

    // Future features
    bool verify_ = false;  ///< Enable result verification (TODO: implementation needed)
};

/**
 * @class PerfResult
 * @brief Performance measurement results for kernel execution
 *
 * Stores comprehensive performance metrics from kernel profiling,
 * including timing, computational throughput, and memory bandwidth.
 * Provides comparison and validation functionality.
 */
class PerfResult {
public:
    /**
     * @brief Default constructor with invalid/uninitialized state
     */
    PerfResult(): split_k_(-1), latency_(-1.0), tflops_(-1.0), bandwidth_(-1.0) {}

    /**
     * @brief Constructor with explicit performance values
     * @param split_k Split-K parameter used for the measurement
     * @param latency Execution latency in milliseconds
     * @param tflops Computational throughput in TFlops
     * @param bandwidth Memory bandwidth utilization in GB/s
     */
    PerfResult(int64_t split_k, double latency, double tflops, double bandwidth):
        split_k_(split_k), latency_(latency), tflops_(tflops), bandwidth_(bandwidth)
    {
    }

    /**
     * @brief Serialize performance results to JSON format
     * @return JSON string representation of performance metrics
     *
     * Creates a standardized JSON representation for logging and storage.
     */
    std::string Serialize() const
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        oss << "{\n"
            << "   \"split_k\": " << split_k_ << ",\n"
            << "   \"latency(ms)\": " << latency_ << ",\n"
            << "   \"tflops(TFlops)\": " << tflops_ << ",\n"
            << "   \"bandwidth(GB/s)\": " << bandwidth_ << "\n"
            << "}";
        return oss.str();
    }

    /**
     * @brief Compare two performance results based on specified metric
     * @param a First performance result
     * @param b Second performance result
     * @param m Metric to use for comparison
     * @return true if 'a' is better than 'b' according to the metric
     * @throws std::invalid_argument if metric is not supported
     *
     * Comparison semantics:
     * - LATENCY: lower is better
     * - TFLOPS: higher is better
     * - BANDWIDTH: higher is better
     */
    static bool compare(const PerfResult& a, const PerfResult& b, Metric m)
    {
        switch (m) {
            case Metric::LATENCY:
                return a.latency_ < b.latency_;
            case Metric::TFLOPS:
                return a.tflops_ > b.tflops_;
            case Metric::BANDWIDTH:
                return a.bandwidth_ > b.bandwidth_;
            default:
                throw std::invalid_argument("Unsupported metric type");
        }
    }

    /**
     * @brief Check if the performance result contains valid measurements
     * @return true if all metrics are within valid ranges
     *
     * Valid ranges:
     * - split_k: >= -1 (where -1 indicates not applicable)
     * - latency, tflops, bandwidth: >= 0
     */
    bool IsValid() const
    {
        return split_k_ >= -1 && latency_ >= 0 && tflops_ >= 0 && bandwidth_ >= 0;
    }

    /**
     * @brief Get efficiency score based on primary metric
     * @param metric Primary metric to use for scoring
     * @return Normalized efficiency score (higher is better)
     */
    double GetEfficiencyScore(Metric metric = Metric::TFLOPS) const
    {
        if (!IsValid())
            return 0.0;

        switch (metric) {
            case Metric::LATENCY:
                return latency_ > 0 ? (1000.0 / latency_) : 0.0;  // Inverse latency
            case Metric::TFLOPS:
                return tflops_;
            case Metric::BANDWIDTH:
                return bandwidth_;
            default:
                return 0.0;
        }
    }

    int64_t split_k_;    ///< Split-K parameter (-1 if not applicable)
    double  latency_;    ///< Execution latency in milliseconds
    double  tflops_;     ///< Computational throughput in TFlops
    double  bandwidth_;  ///< Memory bandwidth utilization in GB/s
};

/**
 * @class InstanceData
 * @brief Complete profiling data record for kernel instances
 *
 * Stores comprehensive information about a kernel profiling session,
 * including environment context, configuration settings, problem
 * definition, and performance results. Used for database storage
 * and result analysis.
 */
class InstanceData {
public:
    /**
     * @brief Constructor for query operations (without results)
     * @param env Hardware and software environment information
     * @param setting Profiling configuration settings
     * @param code_gen_kind Type of kernel being profiled
     * @param problem Problem specification for the kernel
     */
    InstanceData(Environment                                         env,
                 Setting                                             setting,
                 CodeGenKind                                         code_gen_kind,
                 std::variant<NormProblem, GemmProblem, FmhaProblem> problem):
        environment_(std::move(env)),
        setting_(std::move(setting)),
        code_gen_kind_(code_gen_kind),
        problem_(std::move(problem))
    {
    }

    /**
     * @brief Constructor for insertion operations (with complete results)
     * @param env Hardware and software environment information
     * @param setting Profiling configuration settings
     * @param code_gen_kind Type of kernel being profiled
     * @param problem Problem specification for the kernel
     * @param instance_name Unique identifier for the kernel instance
     * @param perf_result Performance measurement results
     */
    InstanceData(Environment                                         env,
                 Setting                                             setting,
                 CodeGenKind                                         code_gen_kind,
                 std::variant<NormProblem, GemmProblem, FmhaProblem> problem,
                 std::string                                         instance_name,
                 PerfResult                                          perf_result):
        environment_(std::move(env)),
        setting_(std::move(setting)),
        code_gen_kind_(code_gen_kind),
        problem_(std::move(problem)),
        instance_name_(std::move(instance_name)),
        perf_result_(std::move(perf_result))
    {
    }

    /**
     * @brief Serialize complete instance data to JSON format
     * @return JSON string representation of all instance information
     *
     * Creates a comprehensive record suitable for database storage
     * and detailed analysis.
     */
    std::string Serialize() const
    {
        std::ostringstream oss;
        oss << "{\n"
            << "   \"environment\": " << environment_.Serialize() << ",\n"
            << "   \"setting\": " << setting_.Serialize() << ",\n"
            << "   \"code_gen_kind\": \"" << CodeGenKindToString(code_gen_kind_) << "\",\n"
            << "   \"instance_name\": \"" << instance_name_ << "\",\n"
            << "   \"perf_result\": " << perf_result_.Serialize() << "\n"
            << "}";
        return oss.str();
    }

    /**
     * @brief Apply visitor pattern to the problem variant
     * @tparam Visitor Visitor function/callable type
     * @param vis Visitor to apply to the problem
     * @return Result of visitor application
     *
     * Enables type-safe operations on the problem variant without
     * explicit type checking or casting.
     */
    template<typename Visitor>
    auto VisitProblem(Visitor&& vis)
    {
        return std::visit(std::forward<Visitor>(vis), problem_);
    }

    /**
     * @brief Apply visitor pattern to the problem variant (const version)
     * @tparam Visitor Visitor function/callable type
     * @param vis Visitor to apply to the problem
     * @return Result of visitor application
     */
    template<typename Visitor>
    auto VisitProblem(Visitor&& vis) const
    {
        return std::visit(std::forward<Visitor>(vis), problem_);
    }

    /**
     * @brief Update the problem specification
     * @tparam Problem Problem type to set
     * @param prob New problem specification
     */
    template<typename Problem>
    void SetProblem(Problem&& prob)
    {
        problem_ = std::forward<Problem>(prob);
    }

    /**
     * @brief Update the instance name
     * @param instance_name New instance identifier
     */
    void SetInstanceName(const std::string& instance_name)
    {
        instance_name_ = instance_name;
    }

    /**
     * @brief Update the performance results
     * @param perf_result New performance measurements
     */
    void SetPerfResult(const PerfResult& perf_result)
    {
        perf_result_ = perf_result;
    }

    /**
     * @brief Check if the instance has valid performance data
     * @return true if instance name is set and performance result is valid
     */
    bool HasValidResults() const
    {
        return !instance_name_.empty() && perf_result_.IsValid();
    }

    Environment environment_;    ///< Hardware and software context
    Setting     setting_;        ///< Profiling configuration
    CodeGenKind code_gen_kind_;  ///< Type of kernel operation

    std::variant<NormProblem, GemmProblem, FmhaProblem> problem_;  ///< Problem specification (extensible via variant)
    std::string                                         instance_name_;  ///< Unique kernel instance identifier
    PerfResult                                          perf_result_;    ///< Performance measurement results
};

/**
 * @class RunningItem
 * @brief Runtime kernel instance information for production execution
 *
 * Stores essential information for kernel instances that are ready for
 * production use, including runtime conditions and performance expectations.
 * Used by the runtime system to select optimal kernels.
 */
class RunningItem {
public:
    /**
     * @brief Default constructor with empty state
     */
    RunningItem() = default;

    /**
     * @brief Constructor with complete runtime information
     * @param running_cond Runtime execution conditions
     * @param instance_name Kernel instance identifier
     * @param perf_result Expected performance characteristics
     */
    RunningItem(std::string running_cond, std::string instance_name, PerfResult perf_result):
        running_cond_(std::move(running_cond)),
        instance_name_(std::move(instance_name)),
        perf_result_(std::move(perf_result))
    {
    }

    /**
     * @brief Check if the running item has complete valid information
     * @return true if all required fields are populated and performance result is valid
     *
     * Validates that the running item contains sufficient information for
     * runtime kernel selection and execution.
     */
    bool IsInstanceExist() const
    {
        return !instance_name_.empty() && !running_cond_.empty() && perf_result_.IsValid();
    }

    /**
     * @brief Get the performance score for runtime selection
     * @param metric Metric to use for scoring (default: TFLOPS)
     * @return Performance score for comparison
     */
    double GetPerformanceScore(Metric metric = Metric::TFLOPS) const
    {
        return perf_result_.GetEfficiencyScore(metric);
    }

    /**
     * @brief Serialize running item to JSON format
     * @return JSON string representation
     */
    std::string Serialize() const
    {
        std::ostringstream oss;
        oss << "{\n"
            << "   \"running_cond\": \"" << running_cond_ << "\",\n"
            << "   \"instance_name\": \"" << instance_name_ << "\",\n"
            << "   \"perf_result\": " << perf_result_.Serialize() << "\n"
            << "}";
        return oss.str();
    }

    std::string running_cond_;   ///< Runtime execution conditions and constraints
    std::string instance_name_;  ///< Unique identifier for the kernel instance
    PerfResult  perf_result_;    ///< Expected performance characteristics
};

/**
 * @enum ProfilingStrategy
 * @brief Strategies for dynamically profiling kernels with varying workload shapes
 *
 * Defines different approaches for extracting representative workloads from
 * dynamic shape specifications. Used by the profiler engine to determine
 * which specific problem sizes to profile for optimal kernel selection.
 */
enum class ProfilingStrategy {
    kMax       = 0,  ///< Use maximum values of dynamic shapes as representative workload
    kMin       = 1,  ///< Use minimum values of dynamic shapes as representative workload
    kHint      = 2,  ///< Use user-provided hint values as representative workload
    kIteration = 3,  ///< Profile across range of values with specified step intervals
};

/**
 * @brief Convert ProfilingStrategy enum to human-readable string representation
 * @param strategy The profiling strategy to convert
 * @return String representation of the strategy
 *
 * Provides consistent string representation for logging, configuration,
 * and debugging purposes.
 */
inline std::string ProfilingStrategyToString(ProfilingStrategy strategy)
{
    switch (strategy) {
        case ProfilingStrategy::kMax:
            return "Max";
        case ProfilingStrategy::kMin:
            return "Min";
        case ProfilingStrategy::kHint:
            return "Hint";
        case ProfilingStrategy::kIteration:
            return "Iteration";
        default:
            return "Unknown";
    }
}

/**
 * @brief Get short name for initialization method
 * @param method Initialization method enum value
 * @return Short string identifier, or "unknown" if not found
 *
 * Provides concise identifiers for initialization methods used in
 * configuration and logging contexts.
 */
inline std::string GetInitMethodShortName(InitMethod method)
{
    auto it = g_init_method_short_names_map.find(method);
    return (it != g_init_method_short_names_map.end()) ? it->second : "unknown";
}

/**
 * @brief Compare two performance results using specified metric and tolerance
 * @param a First performance result
 * @param b Second performance result
 * @param metric Metric to use for comparison
 * @param tolerance Relative tolerance for "equal" comparison (default: 1%)
 * @return -1 if a < b, 0 if a ≈ b, 1 if a > b
 *
 * Provides robust comparison with tolerance to handle measurement noise
 * and minor performance variations.
 */
inline int
ComparePerformanceWithTolerance(const PerfResult& a, const PerfResult& b, Metric metric, double tolerance = 0.01)
{
    if (!a.IsValid() && !b.IsValid())
        return 0;
    if (!a.IsValid())
        return -1;
    if (!b.IsValid())
        return 1;

    double val_a = a.GetEfficiencyScore(metric);
    double val_b = b.GetEfficiencyScore(metric);

    double relative_diff = std::abs(val_a - val_b) / std::max(val_a, val_b);
    if (relative_diff < tolerance) {
        return 0;  // Considered equal within tolerance
    }

    return PerfResult::compare(a, b, metric) ? 1 : -1;
}

}  // namespace flashck