#pragma once

#include <functional>

#include <torch/cuda.h>
#include <torch/torch.h>

#include "flashck/core/utils/timer.h"

class ProfileBenchmark {
public:
    ProfileBenchmark() = default;

    ProfileBenchmark(int warm_up_iters, int forward_iters): warm_up_iters_(warm_up_iters), forward_iters_(forward_iters)
    {
    }

    template<typename F, typename... Args>
    inline void BenchmarkTorchFunc(F func, Args... args)
    {
        // warm up
        for (int i = 0; i < warm_up_iters_; i++) {
            func(args...);
        }

        // start benchmrak
        flashck::HipTimer hip_timer;
        hip_timer.Start();
        for (int i = 0; i < forward_iters_; i++) {
            func(args...);
        }
        float elapsed_time = hip_timer.Stop();
        profile_time_pt_   = elapsed_time / forward_iters_;
    }

    // function for benchmarking flashck ck kernel.
    template<typename F>
    inline void BenchmarkflashckFunc(F func)
    {
        // warm up
        for (int i = 0; i < warm_up_iters_; i++) {
            func->Forward();
        }

        // start benchmrak
        flashck::HipTimer hip_timer;
        hip_timer.Start();
        for (int i = 0; i < forward_iters_; i++) {
            func->Forward();
        }
        float elapsed_time = hip_timer.Stop();

        profile_time_ater_ = elapsed_time / forward_iters_;
    }

    inline void SetTestName(const std::string& name)
    {
        test_name_ = name;
    }

    inline void PrintProfileResult()
    {
        // print profile result
        std::cout.precision(5);
        std::cout << "--- Profile Result ---" << std::endl;
        std::cout << "Test Name: " << test_name_ << std::endl;
        std::cout << "PyTorch Time: " << profile_time_pt_ << " ms" << std::endl;
        std::cout << "LI Time: " << profile_time_ater_ << " ms" << std::endl;
        std::cout << "Speedup: " << profile_time_pt_ / profile_time_ater_ << std::endl;
        std::cout << "----------------------" << std::endl;
    }

    std::string test_name_;
    int         warm_up_iters_ = 3, forward_iters_ = 20;
    float       profile_time_pt_ = 0.f, profile_time_ater_ = 0.f;
};