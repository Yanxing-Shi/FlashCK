#include "core/utils/json_config.h"
#include <iostream>
#include <cassert>

using namespace flashck;

int main() {
    // Test reading single config (default_config.json)
    std::cout << "Testing single config loading..." << std::endl;
    try {
        auto single_config = LoadDefaultConfigJson<FmhaBatchPrefillConfig>("configs/attention/fmha_batch_prefill/default_config.json");
        std::cout << "Single config loaded successfully!" << std::endl;
        std::cout << "Pipeline values count: " << single_config.pipeline.GetAllValues().size() << std::endl;
        std::cout << "Block tile m0 values count: " << single_config.tile_shape.block_tile.m0.GetAllValues().size() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error loading single config: " << e.what() << std::endl;
    }

    // Test reading backup config array (backup_config.json)
    std::cout << "\nTesting backup config array loading..." << std::endl;
    try {
        auto backup_configs = LoadBackupConfigJson<FmhaBatchPrefillConfig>("configs/attention/fmha_batch_prefill/backup_config.json");
        std::cout << "Backup configs loaded successfully!" << std::endl;
        std::cout << "Number of backup configs: " << backup_configs.size() << std::endl;
        
        for (size_t i = 0; i < backup_configs.size(); ++i) {
            const auto& config = backup_configs[i];
            std::cout << "Config " << i << ":" << std::endl;
            std::cout << "  - Block tile m0: " << config.tile_shape.block_tile.m0.GetAllValues()[0] << std::endl;
            std::cout << "  - Block tile n0: " << config.tile_shape.block_tile.n0.GetAllValues()[0] << std::endl;
            std::cout << "  - Block tile k0: " << config.tile_shape.block_tile.k0.GetAllValues()[0] << std::endl;
            std::cout << "  - Padding s: " << (config.padding.s.GetAllValues()[0] ? "true" : "false") << std::endl;
            std::cout << "  - Pipeline: " << config.pipeline.GetAllValues()[0] << std::endl;
            
            // Verify each config has exactly one value for each parameter
            assert(config.tile_shape.block_tile.m0.GetAllValues().size() == 1);
            assert(config.tile_shape.block_tile.n0.GetAllValues().size() == 1);
            assert(config.tile_shape.block_tile.k0.GetAllValues().size() == 1);
            assert(config.padding.s.GetAllValues().size() == 1);
            assert(config.pipeline.GetAllValues().size() == 1);
        }
        
        std::cout << "All backup configs validated - each has exactly one value per parameter!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error loading backup configs: " << e.what() << std::endl;
    }

    std::cout << "\nAll tests completed!" << std::endl;
    return 0;
}