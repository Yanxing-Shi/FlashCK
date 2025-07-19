# FlashCK LayerNorm Header-Only Wrapper - 编译和使用指南

## 系统概述

FlashCK LayerNorm Header-Only Wrapper 提供了一个模块化、可扩展的C++接口来使用FlashCK的LayerNorm操作。系统设计为完全header-only，便于集成和使用。

## 目录结构

```
flashck/wrapper/cpp/                    # 主要wrapper组件
├── CMakeLists.txt                      # 模块化CMake配置
├── README.md                           # 详细文档
└── norm/
    └── layer_norm.h                    # LayerNorm头文件

examples/cpp/norm/                      # 使用示例
├── CMakeLists.txt                      # 示例构建配置
└── example_layernorm.cpp               # LayerNorm使用示例

tests/cpp/norm/                         # 测试和基准
├── CMakeLists.txt                      # 测试构建配置
├── test_layer_norm_correctness.cpp     # 正确性测试
└── bench_layer_norm_performance.cpp    # 性能基准测试
```

## 编译过程

### 方法一：完整项目构建（推荐）

```bash
# 1. 进入项目根目录
cd /home/yanxishi/flash_ck

# 2. 创建构建目录
mkdir -p build && cd build

# 3. 配置CMake（启用示例和测试）
cmake .. -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON -DBUILD_WRAPPER=ON

# 4. 编译项目
make -j$(nproc)

# 5. 运行示例
./examples/norm/example_layernorm

# 6. 运行测试
./tests/norm/test_layer_norm_correctness
./tests/norm/bench_layer_norm_performance
```

### 方法二：仅构建wrapper组件

```bash
cd /home/yanxishi/flash_ck
mkdir -p build && cd build

# 只构建wrapper相关目标
cmake .. -DBUILD_WRAPPER=ON
make flashck_cpp_wrapper flashck_layernorm_header_only -j$(nproc)
```

## 使用说明

### 1. 在你的项目中使用LayerNorm

#### CMakeLists.txt 配置
```cmake
# 找到FlashCK包
find_package(FlashCK REQUIRED)

# 创建你的可执行文件
add_executable(my_app main.cpp)

# 链接LayerNorm组件
target_link_libraries(my_app PRIVATE FlashCK::LayerNorm)

# 或者链接所有wrapper组件
# target_link_libraries(my_app PRIVATE FlashCK::cpp_wrapper)
```

#### C++ 代码使用
```cpp
#include "flashck/wrapper/cpp/norm/layer_norm.h"
#include <memory>

int main() {
    const int batch_size = 32;
    const int hidden_dim = 768;
    const float epsilon = 1e-5f;
    
    // 分配内存
    auto input = std::make_unique<float[]>(batch_size * hidden_dim);
    auto gamma = std::make_unique<float[]>(hidden_dim);
    auto beta = std::make_unique<float[]>(hidden_dim);
    
    // 初始化数据...
    
    // 执行LayerNorm
    float* output = flashck::layer_norm_fwd(
        input.get(), gamma.get(), beta.get(), 
        batch_size, hidden_dim, epsilon
    );
    
    // 使用结果...
    
    return 0;
}
```

### 2. 直接编译（不使用CMake）

```bash
g++ -std=c++20 \
    -I/home/yanxishi/flash_ck \
    -I/home/yanxishi/flash_ck/3rdparty/composable_kernel/include \
    your_code.cpp \
    -lflashck_static \
    -o your_program
```

## 测试验证

### 正确性测试
```bash
cd build
./tests/norm/test_layer_norm_correctness

# 预期输出：
# [==========] Running 6 tests from 1 test suite.
# [----------] Global test environment set-up.
# [----------] 6 tests from LayerNormCorrectnessTest
# [ RUN      ] LayerNormCorrectnessTest.BasicFloat32Test
# [       OK ] LayerNormCorrectnessTest.BasicFloat32Test
# ...
# [==========] 6 tests from 1 test suite ran.
# [  PASSED  ] 6 tests.
```

### 性能基准测试
```bash
cd build
./tests/norm/bench_layer_norm_performance

# 预期输出示例：
# === LayerNorm Performance Benchmarks ===
# 
# Small Sequence Length Benchmarks:
# LayerNorm Float32 [32x768]:
#   Min: 1.234 ms
#   Max: 1.456 ms
#   Avg: 1.345 ms
#   Throughput: 2.345 GB/s
```

## 系统特性

### 模块化设计
- **独立组件**: 每个操作（LayerNorm）都是独立的CMake target
- **可选依赖**: 可以只链接需要的组件
- **易扩展**: 添加新操作只需要几行CMake代码

### 编译优化
- **C++20标准**: 统一在wrapper级别设置，子组件自动继承
- **编译器警告**: 在wrapper级别统一配置
- **优化级别**: 测试和示例可以独立配置优化级别

### 错误处理
- **输入验证**: 所有函数都包含完整的参数验证
- **异常安全**: 使用标准C++异常处理机制
- **清晰错误信息**: 详细的错误描述便于调试

## 扩展新操作

### 1. 添加新操作目录
```bash
mkdir flashck/wrapper/cpp/your_op
```

### 2. 创建头文件
```cpp
// flashck/wrapper/cpp/your_op/your_op.h
#pragma once
#include <stdexcept>

namespace flashck {
template<typename T>
T* your_op_fwd(T* input, /* params */) {
    if (!input) {
        throw std::runtime_error("YourOp: null input");
    }
    // 实现...
    return output;
}
}
```

### 3. 更新CMakeLists.txt
在 `flashck/wrapper/cpp/CMakeLists.txt` 中添加：
```cmake
# Component N: YourOp Operations  
add_library(flashck_your_op_header_only INTERFACE)
add_library(FlashCK::YourOp ALIAS flashck_your_op_header_only)

target_include_directories(flashck_your_op_header_only INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/composable_kernel/include>
    $<INSTALL_INTERFACE:include>
)

# 链接到主wrapper
target_link_libraries(flashck_cpp_wrapper INTERFACE
    flashck_layernorm_header_only
    flashck_your_op_header_only  # 添加新组件
)
```

## 故障排除

### 常见问题

1. **找不到FlashCK::LayerNorm目标**
   ```
   解决：确保在项目CMakeLists.txt中包含wrapper子目录
   add_subdirectory(flashck/wrapper/cpp)
   ```

2. **头文件找不到**
   ```
   解决：检查include路径，确保指向项目根目录
   target_include_directories(your_target PRIVATE ${PROJECT_SOURCE_DIR})
   ```

3. **链接错误**
   ```
   解决：确保flashck_static或相关核心库已正确构建
   make flashck_static
   ```

### 调试技巧

1. **查看CMake变量**
   ```bash
   cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON
   ```

2. **编译时显示详细信息**
   ```bash
   make VERBOSE=1
   ```

3. **检查目标依赖**
   ```bash
   cmake --build . --target help
   ```

## 性能建议

1. **编译优化**: 
   - Release模式：`cmake .. -DCMAKE_BUILD_TYPE=Release`
   - 特定优化：benchmark目标自动使用`-O3 -march=native`

2. **内存管理**:
   - 使用aligned内存分配
   - 避免频繁的小内存分配

3. **批次大小优化**:
   - 较大的批次大小通常性能更好
   - 根据GPU内存调整批次大小

## 总结

这个header-only wrapper系统提供了：
- ✅ 模块化、可扩展的架构
- ✅ 统一的编译标准和选项管理
- ✅ 完整的测试和示例
- ✅ 清晰的使用文档
- ✅ 简化的集成流程

使用此系统，你可以轻松地在项目中集成FlashCK的LayerNorm操作，并根据需要扩展更多操作。
