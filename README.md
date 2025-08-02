# FlashCK: Fast and Memory-Efficient CK Kernel

FlashCK is a high-performance kernel library optimized for attention mechanisms and various deep learning operations, built on top of Composable Kernel (CK).

## Overview

FlashCK provides optimized implementations for:
- Flash Attention (FMHA) operations including batch prefill, forward pass, and paged KV
- GEMM operations with various configurations
- Normalization layers
- Mixture of Experts (MoE) operations

## Key Features

### Unified Configuration System
- **Three Config Types**: backup (pre-validated), default (parameter ranges), user (custom)
- **Three Execution Modes**: heuristic (0), autotuning (1), hybrid (2)
- **Type-Safe**: Strong typing with nlohmann::json serialization
- **Template-Based**: Single `LoadConfigJson<T>()` interface for all config types

### Advanced Code Generation
- **Hierarchical Tiling**: Multi-level optimization from block to warp to thread level
- **Jinja2 Templates**: Flexible code generation with template inheritance
- **Intelligent Filtering**: Heuristic-based instance filtering for optimal performance
- **CartesianProduct**: Comprehensive parameter space exploration

### Performance Optimization
- **Problem-Specific Heuristics**: Tailored filtering for different kernel types
- **Memory Efficiency**: Optimized memory access patterns and allocation
- **Hardware Adaptation**: GPU-specific optimizations for different architectures

## Architecture

```
FlashCK/
├── core/                    # Core infrastructure
│   ├── graph/              # Computation graph management
│   ├── memory/             # Memory allocation and management
│   ├── module/             # Kernel modules and operations
│   ├── profiling/          # Performance profiling and tuning
│   ├── pybind/             # Python bindings
│   └── utils/              # Utilities including json_config.h
├── configs/                # JSON configuration files
│   └── fmha/              # FMHA-specific configurations
├── examples/               # Usage examples
├── tests/                  # Test suites
└── flash_ck/              # Python package
```

## Quick Start

### Building FlashCK
```bash
# Configure build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)

# Install
cmake --install build
```

### Using FlashCK
```cpp
#include "flashck/fmha_batch_prefill.h"
#include "core/utils/json_config.h"

// Load configuration
auto config = LoadConfigJson<FmhaBatchPrefillConfig>("configs/fmha/default_config.json");

// Create and run FMHA instance
auto emitter = FmhaBatchPrefillEmitter::GetInstance();
auto instances = emitter->GenerateInstances(problem_size);
```

### Python Interface
```python
import flash_ck

# Configure attention parameters
config = flash_ck.FMHAConfig(
    head_dim=64,
    batch_size=32,
    seq_len=2048
)

# Run flash attention
output = flash_ck.fmha_batch_prefill(query, key, value, config)
```

## Configuration System

### Environment Variables
- `FC_TUNING_MODE`: Execution mode (0=heuristic, 1=autotuning, 2=hybrid)
- `FC_ENABLE_BACKUP_JSON`: Enable backup config loading
- `FC_ENABLE_DEFAULT_JSON`: Enable default config loading
- `FC_ENABLE_USER_JSON`: Enable user config loading
- `FC_CONFIG_JSON_PATH`: Base configuration directory

### Configuration Types
1. **Default Configs**: Parameter ranges for autotuning
2. **Backup Configs**: Pre-validated single configurations
3. **User Configs**: Custom parameter ranges

### Execution Modes
- **Mode 0 (Heuristic)**: Fast execution with intelligent filtering
- **Mode 1 (Autotuning)**: Comprehensive search for maximum performance
- **Mode 2 (Hybrid)**: Balanced approach combining heuristics and search

## Development

### Adding New Kernels
1. Define configuration struct in `core/module/*/config.h`
2. Implement emitter class following singleton pattern
3. Create Jinja2 templates for code generation
4. Add heuristic filtering logic
5. Update configuration files

### Testing
```bash
# Run all tests
cd build && ctest

# Run specific test
./tests/test_fmha_batch_prefill

# Test configuration system
./tests/test_json_config
```

## Performance

FlashCK achieves significant performance improvements through:
- **Memory Coalescing**: Optimized memory access patterns
- **Tiling Strategies**: Hierarchical blocking for cache efficiency
- **Register Optimization**: Efficient register usage and data reuse
- **Kernel Fusion**: Fused operations to reduce memory bandwidth

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the coding style guidelines
4. Add comprehensive tests
5. Submit a pull request

## License

FlashCK is licensed under [LICENSE]. See the LICENSE file for details.

## Citations

If you use FlashCK in your research, please cite:
```bibtex
@article{flashck2024,
  title={FlashCK: Fast and Memory-Efficient Attention Kernels},
  author={FlashCK Team},
  year={2024}
}
```