/**
 * @file TEST_FRAMEWORK_GUIDE.md
 * @brief Comprehensive guide for the unified FlashCK test framework
 */

# FlashCK Unified Test Framework

## Overview

The FlashCK Unified Test Framework provides a comprehensive, extensible infrastructure for testing both correctness and performance of all FlashCK operations. The framework ensures that all operation inputs are managed on GPU (following the examples pattern) and provides a consistent interface for adding new operations.

## Key Features

- **Unified Testing**: Single framework for both correctness and performance testing
- **GPU Memory Management**: Automatic GPU memory allocation and cleanup  
- **Extensible Architecture**: Easy to add new operations (FMHA, GEMM, etc.)
- **Detailed Error Reporting**: Comprehensive error metrics and debugging information
- **Performance Analysis**: Throughput analysis, statistical summaries, and comparisons
- **Type Safety**: Template-based design supporting multiple data types

## Architecture

### Core Components

```
tests/cpp/
├── common/
│   ├── test_framework.h       # Core unified test framework
│   ├── norm_test_config.h     # LayerNorm/RMSNorm configurations
│   ├── norm_reference_impl.h  # Reference CPU implementations
│   └── op_test_configs.h      # Templates for future operations
├── norm/
│   ├── test_norm_unified.cpp  # Unified norm tests (new)
│   ├── test_*_correctness.cpp # Legacy correctness tests
│   └── bench_*_performance.cpp # Legacy performance tests
└── CMakeLists.txt             # Build configuration
```

### Interface Design

All operation configurations implement the `OpConfigBase<T>` interface:

```cpp
template<typename T>
class OpConfigBase {
public:
    virtual std::string name() const = 0;
    virtual std::string operation_type() const = 0;
    virtual size_t output_size() const = 0;
    virtual size_t total_bytes() const = 0;
    
    virtual void init_test_data(DataGenerator<T>& data_gen) = 0;
    virtual void setup_gpu_inputs(GpuMemoryManager<T>& gpu_mem) = 0;
    virtual T* get_gpu_output(GpuMemoryManager<T>& gpu_mem) = 0;
    virtual void get_cpu_inputs_for_reference(std::vector<const T*>& inputs) const = 0;
};
```

## Usage Guide

### 1. Adding a New Operation

To add a new operation (e.g., GEMM), follow these steps:

#### Step 1: Create Configuration Class

```cpp
template<typename T>
class GemmConfig : public OpConfigBase<T> {
public:
    GemmConfig(int m, int n, int k, /* other params */) {
        // Initialize dimensions and allocate CPU memory
        // See op_test_configs.h for complete example
    }
    
    // Implement OpConfigBase interface
    std::string name() const override { return name_; }
    std::string operation_type() const override { return "GEMM"; }
    size_t output_size() const override { return m_ * n_; }
    size_t total_bytes() const override { return /* compute bytes */; }
    
    void setup_gpu_inputs(GpuMemoryManager<T>& gpu_mem) override {
        gpu_a_ = gpu_mem.allocate_and_copy(a_data_.get(), a_size_, "matrix_A");
        gpu_b_ = gpu_mem.allocate_and_copy(b_data_.get(), b_size_, "matrix_B");
        gpu_output_ = gpu_mem.allocate(output_size_, "output");
    }
    // ... other methods
};
```

#### Step 2: Create Reference Implementation

```cpp
template<typename T>
class GemmReference {
public:
    static void forward(const GemmConfig<T>& config, T* output) {
        // CPU reference implementation
    }
};
```

#### Step 3: Create FlashCK Wrapper

```cpp
template<typename T>
T* run_flashck_gemm(const GemmConfig<T>& config, GpuMemoryManager<T>& gpu_mem) {
    try {
        return gemm_fwd(config.gpu_a(), config.gpu_b(), config.gpu_c(), 
                       config.m(), config.n(), config.k());
    } catch (const std::exception& e) {
        std::cerr << "GEMM execution failed: " << e.what() << std::endl;
        return nullptr;
    }
}
```

#### Step 4: Write Tests

```cpp
TEST_F(UnifiedTestSuite<float>, GemmCorrectnessTest) {
    auto configs = OpTestConfigFactory<float>::create_gemm_configs();
    
    auto reference_impl = [](const GemmConfig<float>& config, float* output) {
        GemmReference<float>::forward(config, output);
    };
    
    auto flashck_impl = [](const GemmConfig<float>& config, GpuMemoryManager<float>& gpu_mem) -> float* {
        return run_flashck_gemm(config, gpu_mem);
    };
    
    run_correctness_test(configs, reference_impl, flashck_impl);
}

TEST_F(UnifiedTestSuite<float>, GemmPerformanceTest) {
    auto configs = OpTestConfigFactory<float>::create_gemm_configs();
    
    auto flashck_impl = [](const GemmConfig<float>& config, GpuMemoryManager<float>& gpu_mem) -> float* {
        return run_flashck_gemm(config, gpu_mem);
    };
    
    auto results = run_performance_test(configs, flashck_impl);
    EXPECT_GT(results.size(), 0);
}
```

### 2. Running Tests

#### Build Tests
```bash
mkdir -p build && cd build
cmake .. -DBUILD_TESTING=ON
make -j$(nproc)
```

#### Run Unified Tests (Recommended)
```bash
# Run all unified norm tests (correctness + performance)
./tests/norm/test_norm_unified

# Run with specific filters
./tests/norm/test_norm_unified --gtest_filter="*Correctness*"
./tests/norm/test_norm_unified --gtest_filter="*Performance*"
```

#### Run Legacy Tests (For Comparison)
```bash
# Individual correctness tests
./tests/norm/test_layer_norm_correctness
./tests/norm/test_rms_norm_correctness

# Individual performance tests  
./tests/norm/bench_layer_norm_performance
./tests/norm/bench_rms_norm_performance
```

#### Run via CTest
```bash
# Run all tests
ctest

# Run specific test pattern
ctest -R "norm"
ctest -R "layer_norm"
```

## Framework Components

### 1. GpuMemoryManager<T>

Automatic GPU memory management with debugging support:

```cpp
GpuMemoryManager<float> gpu_mem;
float* gpu_data = gpu_mem.allocate_and_copy(cpu_data, size, "debug_name");
// Automatic cleanup on destruction
```

### 2. DataGenerator<T>

Flexible data generation for testing:

```cpp
DataGenerator<float> data_gen(seed);
data_gen.uniform(data, size, -1.0f, 1.0f);     // Uniform distribution
data_gen.normal(data, size, 0.0f, 1.0f);       // Normal distribution  
data_gen.special_pattern(data, size, "zeros");  // Special patterns
```

### 3. TensorComparator<T>

Advanced tensor comparison with detailed error reporting:

```cpp
bool is_close = TensorComparator<float>::allclose(a, b, size, rtol, atol);
auto metrics = TensorComparator<float>::compute_error_metrics(a, b, size);
TensorComparator<float>::print_error_metrics(metrics, size);
```

### 4. UnifiedTestSuite<T>

Main test framework supporting both correctness and performance:

```cpp
// Correctness testing
run_correctness_test(configs, reference_impl, flashck_impl, rtol, atol, verbose);

// Performance testing
auto results = run_performance_test(configs, flashck_impl, num_runs, warmup_runs);
```

## Performance Analysis

The framework provides comprehensive performance metrics:

- **Timing**: Min/Max/Average execution time with standard deviation
- **Throughput**: GB/s based on total memory bandwidth
- **Comparison**: Automatic sorting and ranking of configurations
- **Statistical Analysis**: Error bars and confidence intervals

Example output:
```
=== PERFORMANCE SUMMARY ===
Configuration               Avg (ms)   Min (ms)   Throughput  StdDev
-----------------------------------------------------------------------
LayerNorm_256x1024           2.345      2.123     145.2 GB/s   0.089
LayerNorm_128x768            1.234      1.156     132.8 GB/s   0.045
RMSNorm_256x1024             1.987      1.845     156.7 GB/s   0.067
```

## Error Handling and Debugging

### GPU Memory Debugging
```cpp
gpu_mem.print_allocations();  // Show all allocations
size_t total = gpu_mem.get_total_allocated_bytes();
```

### Error Metrics
```cpp
auto metrics = TensorComparator<float>::compute_error_metrics(output, reference, size);
std::cout << "MAE: " << metrics.mae << ", RMSE: " << metrics.rmse << std::endl;
std::cout << "Max errors: abs=" << metrics.max_abs_err << ", rel=" << metrics.max_rel_err << std::endl;
```

### Verbose Testing
```cpp
run_correctness_test(configs, reference_impl, flashck_impl, 1e-3f, 1e-4f, true);  // verbose=true
```

## Future Extensions

The framework is designed to easily support additional operations:

1. **Flash Attention (FMHA)**: Template provided in `op_test_configs.h`
2. **GEMM Operations**: Template provided in `op_test_configs.h`  
3. **Convolution Operations**: Follow the same pattern
4. **Custom Operations**: Implement `OpConfigBase<T>` interface

## Best Practices

1. **GPU-Only Inputs**: Always ensure operation inputs are on GPU
2. **Reference Implementations**: Keep reference implementations simple and correct
3. **Error Tolerances**: Choose appropriate rtol/atol based on operation precision
4. **Performance Testing**: Use sufficient warmup runs and iterations
5. **Configuration Naming**: Use descriptive names for easy identification
6. **Memory Management**: Let `GpuMemoryManager` handle all GPU allocations

## Migration from Legacy Tests

To migrate existing tests to the unified framework:

1. **Extract Configuration**: Move test parameters to a config class
2. **Implement Interface**: Implement `OpConfigBase<T>` methods
3. **GPU Memory Setup**: Use `setup_gpu_inputs()` instead of manual allocation
4. **Reference Wrapper**: Adapt reference implementation to new signature
5. **Unified Tests**: Replace separate correctness/performance files with unified tests

The legacy tests remain available for comparison and gradual migration.
