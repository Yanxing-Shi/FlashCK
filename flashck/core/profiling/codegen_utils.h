#pragma once

#include <string>

#include "flashck/core/profiling/codegen_common.h"
#include "flashck/core/utils/common.h"

FC_DECLARE_int32(FC_TUNING_WARM_UP);
FC_DECLARE_int32(FC_TUNING_ITERATIONS);
FC_DECLARE_bool(FC_TUNING_GPU_TIMER);
FC_DECLARE_bool(FC_TUNING_VERIFY);
FC_DECLARE_bool(FC_TUNING_LOG);
FC_DECLARE_bool(FC_TUNING_FLUSH_CACHE);
FC_DECLARE_int32(FC_TUNING_ROTATING_COUNT);

namespace flashck {

enum class Metric {
    LATENCY   = 0,
    TFLOPS    = 1,
    BANDWIDTH = 2
};

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

enum class CodeGenKind {
    Gemm      = 0,
    Norm      = 1,
    Embedding = 2,
    Fmha      = 3,
};

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

class Environment {
public:
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

    std::string device_name_;
    std::string rocm_version_;
};

class Setting {
public:
    Setting():
        n_warmup_(FLAGS_FC_TUNING_WARM_UP),
        n_repeat_(FLAGS_FC_TUNING_ITERATIONS),
        is_gpu_timer_(FLAGS_FC_TUNING_GPU_TIMER),
        verify_(FLAGS_FC_TUNING_VERIFY),
        log_(FLAGS_FC_TUNING_LOG),
        flush_cache_(FLAGS_FC_TUNING_FLUSH_CACHE),
        rotating_count_(FLAGS_FC_TUNING_ROTATING_COUNT)
    {
    }

    std::string Serialize() const
    {
        return "{\n"
               "   \"n_warmup\": "
               + std::to_string(n_warmup_)
               + ",\n"
                 "   \"n_repeat\": "
               + std::to_string(n_repeat_)
               + ",\n"
                 "   \"is_gpu_timer\": "
               + (is_gpu_timer_ ? "true" : "false")
               + ",\n"
                 "   \"verify\": "
               + std::to_string(verify_)
               + ",\n"
                 "   \"log\": "
               + (log_ ? "true" : "false")
               + ",\n"
                 "   \"flush_cache\": "
               + (flush_cache_ ? "true" : "false")
               + ",\n"
                 "   \"rotating_count\": "
               + std::to_string(rotating_count_)
               + "\n"
                 "}";
    }

    int  n_warmup_;
    int  n_repeat_;
    bool is_gpu_timer_;
    bool verify_;
    bool log_;
    bool flush_cache_;
    int  rotating_count_;
};

class PerfResult {
public:
    std::string Serialize() const
    {
        return "{\n"
               "   \"split_k\": "
               + std::to_string(split_k_)
               + ",\n"
                 "   \"latency(ms)\": "
               + std::to_string(latency_)
               + ",\n"
                 "   \"tflops(TFlops)\": "
               + std::to_string(tflops_)
               + ",\n"
                 "   \"bandwidth(GB/s)\": "
               + std::to_string(bandwidth_)
               + "\n"
                 "}";
    }

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

    bool IsValid() const
    {
        return split_k_ >= -1 && latency_ >= 0 && tflops_ >= 0 && bandwidth_ >= 0;
    }

    int64_t split_k_;
    double  latency_;
    double  tflops_;
    double  bandwidth_;
};

// This structure is used to store the results of profiling operations
class InstanceData {
public:
    // query
    InstanceData(Environment env, Setting setting, CodeGenKind code_gen_kind, std::variant<NormProblem> problem):
        environment_(std::move(env)),
        setting_(std::move(setting)),
        code_gen_kind_(code_gen_kind),
        problem_(std::move(problem))
    {
    }

    // insert
    InstanceData(Environment               env,
                 Setting                   setting,
                 CodeGenKind               code_gen_kind,
                 std::variant<NormProblem> problem,
                 std::string               instance_name,
                 PerfResult                perf_result):
        environment_(std::move(env)),
        setting_(std::move(setting)),
        code_gen_kind_(code_gen_kind),
        problem_(std::move(problem)),
        instance_name_(std::move(instance_name)),
        perf_result_(std::move(perf_result))
    {
    }

    std::string Serialize() const
    {
        return "{\n"
               "   \"environment\": "
               + environment_.Serialize()
               + ",\n"
                 "   \"setting\": "
               + setting_.Serialize()
               + ",\n"
                 "   \"code_gen_kind\": \""
               + CodeGenKindToString(code_gen_kind_)
               + "\",\n"
                 "   \"instance_name\": \""
               + instance_name_
               + "\",\n"
                 "   \"perf_result\": "
               + perf_result_.Serialize()
               + "\n"
                 "}";
    }

    template<typename Visitor>
    auto VisitProblem(Visitor&& vis)
    {
        return std::visit(std::forward<Visitor>(vis), problem_);
    }

    template<typename Problem>
    void SetProblem(Problem&& prob)
    {
        problem_ = std::forward<Problem>(prob);
    }

    Environment environment_;
    Setting     setting_;
    CodeGenKind code_gen_kind_;

    std::variant<NormProblem> problem_;
    std::string               instance_name_;
    PerfResult                perf_result_;
};

}  // namespace flashck