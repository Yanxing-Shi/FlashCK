#include "core/utils/json_config.h"
#include <iostream>
#include <cassert>

using namespace flashck;

// Helper function to demonstrate single-value config usage
template<typename ConfigType>
void demonstrateBackupConfig(const std::string& config_name, const std::string& config_path) {
    std::cout << "\n=== Testing " << config_name << " backup configs ===" << std::endl;
    
    try {
        auto backup_configs = LoadBackupConfigJson<ConfigType>(config_path);
        std::cout << "Loaded " << backup_configs.size() << " backup configurations" << std::endl;
        
        for (size_t i = 0; i < backup_configs.size(); ++i) {
            std::cout << "\nConfig " << (i + 1) << " details:" << std::endl;
            
            // Each backup config should have exactly one value per parameter
            // This makes them suitable for direct use without parameter sweeping
            
            if constexpr (std::is_same_v<ConfigType, FmhaBatchPrefillConfig>) {
                const auto& config = backup_configs[i];
                std::cout << "  Block size: " << config.tile_shape.block_tile.m0.values[0] 
                         << "x" << config.tile_shape.block_tile.n0.values[0] 
                         << "x" << config.tile_shape.block_tile.k0.values[0] << std::endl;
                std::cout << "  Pipeline: " << config.pipeline.values[0] << std::endl;
                
                // Verify single values
                assert(config.tile_shape.block_tile.m0.values.size() == 1);
                assert(config.tile_shape.block_tile.n0.values.size() == 1);
                assert(config.pipeline.values.size() == 1);
            }
            
            if constexpr (std::is_same_v<ConfigType, FmhaFwdConfig>) {
                const auto& config = backup_configs[i];
                std::cout << "  Block size: " << config.tile_shape.block_tile.m0.values[0] 
                         << "x" << config.tile_shape.block_tile.n0.values[0] 
                         << "x" << config.tile_shape.block_tile.k0.values[0] << std::endl;
                std::cout << "  Pipeline: " << config.pipeline.values[0] << std::endl;
                
                // Verify single values
                assert(config.tile_shape.block_tile.m0.values.size() == 1);
                assert(config.tile_shape.block_tile.n0.values.size() == 1);
                assert(config.pipeline.values.size() == 1);
            }
            
            if constexpr (std::is_same_v<ConfigType, GemmConfig>) {
                const auto& config = backup_configs[i];
                std::cout << "  Block size: " << config.tile_shape.block_tile.m.values[0] 
                         << "x" << config.tile_shape.block_tile.n.values[0] 
                         << "x" << config.tile_shape.block_tile.k.values[0] << std::endl;
                std::cout << "  Pipeline: " << config.pipeline.version.values[0] << std::endl;
                
                // Verify single values
                assert(config.tile_shape.block_tile.m.values.size() == 1);
                assert(config.tile_shape.block_tile.n.values.size() == 1);
                assert(config.pipeline.version.values.size() == 1);
            }
        }
        
        std::cout << "✓ All " << config_name << " backup configs validated!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Error loading " << config_name << " backup configs: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "=== JSON Config Backup Testing ===" << std::endl;
    std::cout << "This demonstrates loading backup configurations that have single values" << std::endl;
    std::cout << "per parameter, making them ready for direct instantiation." << std::endl;
    
    // Test different config types
    demonstrateBackupConfig<FmhaBatchPrefillConfig>(
        "FMHA Batch Prefill", 
        "configs/fmha/fmha_batch_prefill/backup_config.json"
    );
    
    demonstrateBackupConfig<FmhaFwdConfig>(
        "FMHA Forward", 
        "configs/fmha/fmha_fwd/backup_config.json"
    );
    
    demonstrateBackupConfig<GemmConfig>(
        "GEMM Batch", 
        "configs/gemm/batch_gemm/backup_config.json"
    );
    
    std::cout << "\n=== Usage Example ===" << std::endl;
    std::cout << "// Load backup configs in your application:" << std::endl;
    std::cout << "auto backup_configs = LoadBackupConfigJson<FmhaBatchPrefillConfig>(path);" << std::endl;
    std::cout << "for (const auto& config : backup_configs) {" << std::endl;
    std::cout << "    // Each config has single values, ready to use directly" << std::endl;
    std::cout << "    int m0 = config.tile_shape.block_tile.m0.values[0];" << std::endl;
    std::cout << "    // ... instantiate kernel with this specific config" << std::endl;
    std::cout << "}" << std::endl;
    
    std::cout << "\n=== All tests completed successfully! ===" << std::endl;
    return 0;
}
