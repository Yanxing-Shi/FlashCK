# JSON Configuration System

This directory contains configuration files for various kernel types in FlashCK.

## File Structure

- `default_config.json` - Contains parameter ranges for tuning/search
- `backup_config.json` - Contains pre-validated single-value configurations
- `user_config.json` - Contains custom user-defined configurations

## Configuration Types

### Default Configurations
Default config files contain parameter ranges for automated tuning:
```json
{
  "tile_shape": {
    "block_tile": {
      "m0": { "values": [128, 256, 512] },  // Multiple values for search
      "n0": { "values": [64, 128] },
      ...
    }
  }
}
```

### Backup Configurations  
Backup config files contain arrays of single-value configurations:
```json
[
  {
    "tile_shape": {
      "block_tile": {
        "m0": { "values": [128] },  // Single value, ready to use
        "n0": { "values": [64] },
        ...
      }
    }
  },
  {
    // Another complete config with different single values
    ...
  }
]
```

### User Configurations
User config files follow the same format as default configs but allow custom parameter ranges.

## Usage

### Loading Configurations
```cpp
#include "core/utils/json_config.h"

// Load single config (default/user)
auto config = LoadConfigJson<FmhaFwdBatchPrefillConfig>("path/to/default_config.json");

// Load array of configs (backup)
auto backup_configs = LoadConfigJson<std::vector<FmhaFwdBatchPrefillConfig>>("path/to/backup_config.json");

// Use CartesianProduct for parameter sweeping
std::vector<std::vector<int64_t>> param_lists = {
    /* extract values from config */
};
CartesianProduct(param_lists, [](const std::vector<int64_t>& params) {
    // Test each parameter combination
});
```

## Execution Modes

The system supports three execution modes controlled by `FC_TUNING_MODE`:

### Mode 0: Heuristic
- Applies intelligent filtering to reduce search space
- Randomly selects one optimal configuration
- Best for fast execution with reasonable performance

### Mode 1: Autotuning  
- Uses all valid instances for comprehensive search
- Maximum performance through exhaustive tuning
- Best for production deployment optimization

### Mode 2: Hybrid
- Combines heuristic filtering with comprehensive search
- Balances performance and execution time
- Best for development and testing scenarios

## Configuration Flags

Control which configuration files are loaded:

- `FC_ENABLE_BACKUP_JSON=true` - Load backup_config.json
- `FC_ENABLE_DEFAULT_JSON=true` - Load default_config.json  
- `FC_ENABLE_USER_JSON=true` - Load user_config.json
- `FC_CONFIG_JSON_PATH=/path/to/configs` - Base configuration directory

## Supported Config Types

- `FmhaFwdBatchPrefillConfig` - Batch prefill FMHA
- `FmhaFwdConfig` - Forward FMHA  
- `FmhaFwdPagedKVPrefillConfig` - Paged KV prefill FMHA
- `FmhaFwdSplitKVConfig` - Split KV forward FMHA
- `FmhaFwdSplitKVCombineConfig` - Split KV combine FMHA
- `GemmConfig` - GEMM operations
- `NormConfig` - Normalization operations
- `MoeConfig` - Mixture of Experts operations

## Testing

Run tests to verify configuration system:
```bash
# Test unified config loading functionality
./test_json_config

# Test backup config validation
./test_backup_config
```

## Design Principles

1. **Unified Interface**: Single `LoadConfigJson<T>()` function for all config types
2. **Separation of Concerns**: Different config types for different use cases
3. **Type Safety**: Strong typing with nlohmann::json serialization
4. **Mode Flexibility**: Three execution modes for different optimization strategies
5. **Validation**: Comprehensive instance validation and filtering
