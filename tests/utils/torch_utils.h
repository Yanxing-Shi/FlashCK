#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include <torch/torch.h>

// map data type to torch data type map:
template<typename T>
struct CppTypeToTorchType;

template<>
struct CppTypeToTorchType<float> {
    static constexpr auto value = torch::kFloat32;
    static constexpr auto atol  = 3e-4;
    static constexpr auto rtol  = 2e-5;
};

template<>
struct CppTypeToTorchType<_Float16> {
    static constexpr auto value = torch::kHalf;
    static constexpr auto atol  = 6e-3;
    static constexpr auto rtol  = 3e-3;
};

template<>
struct CppTypeToTorchType<ushort> {
    static constexpr auto value = torch::kBFloat16;
    static constexpr auto atol  = 2.8e-2;
    static constexpr auto rtol  = 2e-2;
};

template<>
struct CppTypeToTorchType<int> {
    static constexpr auto value = torch::kInt32;
    static constexpr auto atol  = 1e-1;
    static constexpr auto rtol  = 1e-1;
};

template<>
struct CppTypeToTorchType<int64_t> {
    static constexpr auto value = torch::kInt64;
    static constexpr auto atol  = 1e-1;
    static constexpr auto rtol  = 1e-1;
};

// template<>
// struct CppTypeToTorchType<at::BFloat16> {
//     static constexpr auto value = torch::kBFloat16;
//     static constexpr auto atol  = 3e-1;
//     static constexpr auto rtol  = 3e-1;
// };

// // map epilogue math name to torch function
// inline torch::nn::Functional GetEpilogueMathFunc(const std::string& epilogue_math)
// {
//     std::unordered_map<std::string, torch::nn::Functional> epilogue_math_func_map = {
//         {"relu", torch::nn::Functional(torch::relu)},
//         {"gelu", torch::nn::Functional(torch::gelu)},
//         {"tanh", torch::nn::Functional(torch::tanh)},
//         {"sigmoid", torch::nn::Functional(torch::sigmoid)},
//     };

//     return epilogue_math_func_map[epilogue_math];
// }

template<typename CppType>
inline torch::Tensor GetEmptyTorchTensor(const std::vector<int64_t>& shape, bool is_device = true)
{
    torch::TensorOptions options = torch::TensorOptions().dtype(CppTypeToTorchType<CppType>::value);
    if (is_device) {
        options = options.device(torch::kCUDA);
    }

    return torch::empty(torch::IntArrayRef(shape), options);
}

template<typename CppType>
inline torch::Tensor
GetRandomTorchTensor(const std::vector<int64_t>& shape, float low = 0.f, float high = 1.0f, bool is_device = true)
{
    torch::TensorOptions options = torch::TensorOptions().dtype(CppTypeToTorchType<CppType>::value);
    if (is_device) {
        options = options.device(torch::kCUDA);
    }

    return torch::randn(torch::IntArrayRef(shape), options) * (high - low) + low;
}

inline torch::Tensor
GetRandomIntTorchTensor(const std::vector<int64_t>& shape, int low = 0, int high = 1, bool is_device = true)
{
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt64);
    if (is_device) {
        options = options.device(torch::kCUDA);
    }

    return torch::randint(low, high, torch::IntArrayRef(shape), options);
}

inline torch::Tensor GetRandpermTorchTensor(const int64_t value, bool is_device = true)
{
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt64);
    if (is_device) {
        options = options.device(torch::kCUDA);
    }

    return torch::randperm(value, options);
}

template<typename CppType>
inline torch::Tensor GetZerosTorchTensor(const std::vector<int64_t>& shape, bool is_device = true)
{

    torch::TensorOptions options = torch::TensorOptions().dtype(CppTypeToTorchType<CppType>::value);
    if (is_device) {
        options = options.device(torch::kCUDA);
    }

    return torch::zeros(torch::IntArrayRef(shape), options);
}

template<typename CppType>
inline torch::Tensor GetOnesTorchTensor(const std::vector<int64_t>& shape, bool is_device = true)
{

    torch::TensorOptions options = torch::TensorOptions().dtype(CppTypeToTorchType<CppType>::value);
    if (is_device) {
        options = options.device(torch::kCUDA);
    }

    return torch::ones(torch::IntArrayRef(shape), options);
}

template<typename CppType>
inline torch::Tensor
GetArrangeTorchTensor(const int64_t start, const int64_t end, const int64_t step = 1, bool is_device = true)
{
    torch::TensorOptions options = torch::TensorOptions().dtype(CppTypeToTorchType<CppType>::value);
    if (is_device) {
        options = options.device(torch::kCUDA);
    }

    return torch::arange(start, end, step, options);
}

template<typename CppType>
inline torch::Tensor GetFullTorchTensor(const std::vector<int64_t>& shape, const CppType value, bool is_device = true)
{
    torch::TensorOptions options = torch::TensorOptions().dtype(CppTypeToTorchType<CppType>::value);
    if (is_device) {
        options = options.device(torch::kCUDA);
    }

    return torch::full(torch::IntArrayRef(shape), value, options);
}