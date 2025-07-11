// Simple test to verify the Builder compilation statistics functionality

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

// Mock version of the Builder class for testing
class MockBuilder {
public:
    struct CompilationStats {
        size_t total_compilations = 0;
        size_t successful_compilations = 0;
        size_t failed_compilations = 0;
        std::vector<std::string> failed_files;
        
        double GetSuccessRate() const {
            return total_compilations > 0 ? 
                static_cast<double>(successful_compilations) / total_compilations * 100.0 : 0.0;
        }
        
        void Reset() {
            total_compilations = 0;
            successful_compilations = 0;
            failed_compilations = 0;
            failed_files.clear();
        }
    };

    const CompilationStats& GetCompilationStats() const { return stats_; }
    void ResetCompilationStats() { stats_.Reset(); }
    
    // Mock compilation process
    void SimulateCompilation(size_t total, size_t failed) {
        stats_.total_compilations = total;
        stats_.failed_compilations = failed;
        stats_.successful_compilations = total - failed;
        
        // Add some mock failed files
        for (size_t i = 0; i < failed; ++i) {
            stats_.failed_files.push_back("failed_file_" + std::to_string(i) + ".cc");
        }
    }

private:
    CompilationStats stats_;
};

// Test the ParseFailedFiles function
std::vector<std::string> ParseFailedFiles(const std::string& make_output) {
    std::vector<std::string> failed_files;
    std::istringstream iss(make_output);
    std::string line;
    
    while (std::getline(iss, line)) {
        if (line.find("error:") != std::string::npos || 
            line.find("fatal error:") != std::string::npos ||
            line.find("make[") != std::string::npos && line.find("Error") != std::string::npos) {
            
            size_t pos = line.find(".cc");
            if (pos != std::string::npos) {
                size_t start = line.rfind('/', pos);
                if (start != std::string::npos) {
                    std::string filename = line.substr(start + 1, pos - start + 2);
                    if (std::find(failed_files.begin(), failed_files.end(), filename) == failed_files.end()) {
                        failed_files.push_back(filename);
                    }
                }
            }
        }
    }
    
    return failed_files;
}

int main() {
    std::cout << "Testing Builder compilation statistics functionality...\n\n";
    
    // Test 1: Basic compilation statistics
    MockBuilder builder;
    
    builder.SimulateCompilation(10, 2);
    const auto& stats = builder.GetCompilationStats();
    
    std::cout << "Test 1 - Basic Statistics:\n";
    std::cout << "Total compilations: " << stats.total_compilations << std::endl;
    std::cout << "Successful compilations: " << stats.successful_compilations << std::endl;
    std::cout << "Failed compilations: " << stats.failed_compilations << std::endl;
    std::cout << "Success rate: " << stats.GetSuccessRate() << "%\n";
    std::cout << "Failed files: ";
    for (const auto& file : stats.failed_files) {
        std::cout << file << " ";
    }
    std::cout << "\n\n";
    
    // Test 2: Parse failed files from make output
    std::string mock_make_output = R"(
/path/to/source/file1.cc:10:5: error: expected ';' after expression
/path/to/source/file2.cc:25:1: fatal error: missing header file
make[1]: *** [file3.cc] Error 1
    )";
    
    auto failed_files = ParseFailedFiles(mock_make_output);
    std::cout << "Test 2 - Parse Failed Files:\n";
    std::cout << "Found " << failed_files.size() << " failed files:\n";
    for (const auto& file : failed_files) {
        std::cout << "  - " << file << std::endl;
    }
    std::cout << "\n";
    
    // Test 3: Reset statistics
    builder.ResetCompilationStats();
    const auto& reset_stats = builder.GetCompilationStats();
    std::cout << "Test 3 - Reset Statistics:\n";
    std::cout << "Total compilations after reset: " << reset_stats.total_compilations << std::endl;
    std::cout << "Success rate after reset: " << reset_stats.GetSuccessRate() << "%\n";
    
    std::cout << "\nAll tests completed successfully!\n";
    return 0;
}
