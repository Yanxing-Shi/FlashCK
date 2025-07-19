# FlashCK Unified Test Framework

## 📋 Overview

The FlashCK Unified Test Framework is a comprehensive, extensible testing infrastructure that combines correctness testing and performance benchmarking for all FlashCK operations. All operation inputs are managed on GPU following the examples pattern.

## 🎯 Key Features

✅ **Unified Testing**: Single framework for both correctness and performance  
✅ **GPU Memory Management**: Automatic allocation and cleanup  
✅ **Extensible Architecture**: Easy to add new operations (FMHA, GEMM, etc.)  
✅ **Detailed Error Reporting**: Comprehensive metrics and debugging  
✅ **Performance Analysis**: Throughput analysis and statistical summaries  
✅ **Type Safety**: Template-based design supporting multiple data types  

## 🏗️ Architecture

```
tests/cpp/
├── common/                     # Core framework components
│   ├── test_framework.h        # 🔧 Unified test framework
│   ├── norm_test_config.h      # ⚙️ LayerNorm/RMSNorm configurations  
│   ├── norm_reference_impl.h   # 📝 Reference CPU implementations
│   └── op_test_configs.h       # 📋 Templates for future operations
├── norm/                       # Normalization tests
│   ├── test_norm_unified.cpp   # ✨ New unified tests (recommended)
│   ├── test_*_correctness.cpp  # 📊 Legacy correctness tests
│   └── bench_*_performance.cpp # ⚡ Legacy performance tests
├── test_operation_template.cpp # 📝 Template for new operations
├── TEST_FRAMEWORK_GUIDE.md     # 📚 Comprehensive guide
└── CMakeLists.txt              # 🔨 Build configuration
```

## 🚀 Quick Start

### Running Unified Tests (Recommended)

```bash
# Build tests
mkdir -p build && cd build
cmake .. -DBUILD_TESTING=ON
make -j$(nproc)

# Run all unified norm tests (correctness + performance)
./tests/norm/test_norm_unified

# Run specific test types
./tests/norm/test_norm_unified --gtest_filter="*Correctness*"
./tests/norm/test_norm_unified --gtest_filter="*Performance*"
./tests/norm/test_norm_unified --gtest_filter="*LayerNorm*"
./tests/norm/test_norm_unified --gtest_filter="*RMSNorm*"
```

### Running Legacy Tests (For Comparison)

```bash
# Individual tests
./tests/norm/test_layer_norm_correctness
./tests/norm/test_rms_norm_correctness
./tests/norm/bench_layer_norm_performance
./tests/norm/bench_rms_norm_performance

# All tests via CTest
ctest
ctest -R "norm"
```

## 📊 Sample Output

### Correctness Testing
```
=== CORRECTNESS TESTING ===

Testing: LayerNorm_32x768 (LayerNorm)
Error Metrics:
  MAE: 0.000045
  RMSE: 0.000123
  Max Abs Error: 0.001234
  Max Rel Error: 0.000567
  Mismatches: 0/24576 (0.000%)
✓ LayerNorm_32x768 passed

Testing: RMSNorm_128x1024 (RMSNorm)
✓ RMSNorm_128x1024 passed

Correctness testing completed: 6 configurations
```

### Performance Testing
```
=== PERFORMANCE BENCHMARKING ===
Benchmarking: LayerNorm_256x1024... 145.2 GB/s
Benchmarking: RMSNorm_256x1024... 156.7 GB/s

=== PERFORMANCE SUMMARY ===
Configuration               Avg (ms)   Min (ms)   Throughput  StdDev
-----------------------------------------------------------------------
RMSNorm_256x1024             1.987      1.845     156.7 GB/s   0.067
LayerNorm_256x1024           2.345      2.123     145.2 GB/s   0.089
LayerNorm_128x768            1.234      1.156     132.8 GB/s   0.045

Best RMSNorm performance: RMSNorm_256x1024 with 156.7 GB/s
```

## 🔧 Adding New Operations

The framework is designed for easy extension. To add a new operation:

### 1. Use the Template
```bash
cp tests/cpp/test_operation_template.cpp tests/cpp/test_my_operation_unified.cpp
```

### 2. Implement Configuration Class
```cpp
template<typename T>
class MyOpConfig : public OpConfigBase<T> {
    // Implement interface methods
    // See op_test_configs.h for examples
};
```

### 3. Write Reference Implementation
```cpp
template<typename T>
class MyOpReference {
public:
    static void forward(const MyOpConfig<T>& config, T* output);
};
```

### 4. Create FlashCK Wrapper
```cpp
template<typename T>
T* run_flashck_my_op(const MyOpConfig<T>& config, GpuMemoryManager<T>& gpu_mem) {
    return my_op_fwd(config.gpu_input(), /* other params */);
}
```

### 5. Write Tests
```cpp
TEST_F(UnifiedTestSuite<float>, MyOpCorrectnessTest) {
    // Use the unified framework
    run_correctness_test(configs, reference_impl, flashck_impl);
}
```

## 🧪 Framework Components

### Core Classes

- **`UnifiedTestSuite<T>`**: Main test framework class
- **`OpConfigBase<T>`**: Interface for operation configurations
- **`GpuMemoryManager<T>`**: Automatic GPU memory management
- **`DataGenerator<T>`**: Flexible test data generation
- **`TensorComparator<T>`**: Advanced tensor comparison

### Utilities

- **Error Metrics**: MAE, RMSE, max errors, mismatch counts
- **Performance Analysis**: Throughput, timing statistics, comparisons
- **Debug Support**: Memory allocation tracking, verbose error reporting

## 📚 Documentation

- **[TEST_FRAMEWORK_GUIDE.md](TEST_FRAMEWORK_GUIDE.md)**: Comprehensive usage guide
- **[test_operation_template.cpp](test_operation_template.cpp)**: Template for new operations
- **Header files**: Extensive Doxygen documentation

## 🎛️ Configuration

### Test Tolerances
```cpp
run_correctness_test(configs, reference_impl, flashck_impl, 
                    1e-3f,  // relative tolerance
                    1e-4f,  // absolute tolerance
                    true);  // verbose output
```

### Performance Settings
```cpp
auto results = run_performance_test(configs, flashck_impl,
                                   20,     // benchmark runs
                                   5,      // warmup runs
                                   true);  // sort by performance
```

## 🔍 Benefits of Unified Framework

### Before (Legacy)
- ❌ Separate files for correctness and performance
- ❌ Duplicated setup code across tests
- ❌ Inconsistent error reporting
- ❌ Manual GPU memory management
- ❌ Different interfaces for each operation

### After (Unified)
- ✅ Single file for both correctness and performance
- ✅ Shared configuration and setup code
- ✅ Consistent, detailed error reporting
- ✅ Automatic GPU memory management
- ✅ Unified interface for all operations
- ✅ Easy to extend for new operations
- ✅ Comprehensive performance analysis

## 🚦 Status

| Operation | Unified Tests | Legacy Tests | Status |
|-----------|---------------|--------------|---------|
| LayerNorm | ✅ Complete   | ✅ Available | ✅ Ready |
| RMSNorm   | ✅ Complete   | ✅ Available | ✅ Ready |
| GEMM      | 📋 Template   | ❌ None      | 🔧 Template Ready |
| FMHA      | 📋 Template   | ❌ None      | 🔧 Template Ready |

## 💡 Best Practices

1. **GPU-Only Inputs**: Always ensure operation inputs are on GPU
2. **Use Unified Tests**: Prefer unified tests over legacy for new development
3. **Descriptive Names**: Use clear configuration names for easy identification
4. **Appropriate Tolerances**: Choose rtol/atol based on operation precision
5. **Performance Testing**: Use sufficient warmup and benchmark runs
6. **Error Analysis**: Use verbose mode for debugging failures

## 🤝 Migration Guide

To migrate existing tests:

1. Extract test parameters to configuration classes
2. Implement `OpConfigBase<T>` interface
3. Use `GpuMemoryManager` for GPU allocations
4. Adapt reference implementations to new signature
5. Replace separate test files with unified tests

Legacy tests remain available for comparison during transition.

---

**Ready to test your operations with confidence! 🚀**
