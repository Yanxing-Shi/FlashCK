# FlashCK: Fast and Memory-Efficient CK Kernel

FlashCK is a high-performance kernel library optimized for attention mechanisms and various deep learning operations, built on top of Composable Kernel (CK).

## Overview

FlashCK provides optimized implementations for:
- Flash Attention (FMHA) operations including batch prefill, forward pass, and paged KV
- GEMM operations with various configurations
- Normalization layers
- Mixture of Experts (MoE) operations

## Key Features

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
```

### Using FlashCK
```cpp
```

### Python Interface
```python
import flash_ck

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

```

## Performance

FlashCK achieves significant performance improvements through:
- **Memory Coalescing**: Optimized memory access patterns
- **Tiling Strategies**: Hierarchical blocking for cache efficiency
- **Register Optimization**: Efficient register usage and data reuse
- **Kernel Fusion**: Fused operations to reduce memory bandwidth

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