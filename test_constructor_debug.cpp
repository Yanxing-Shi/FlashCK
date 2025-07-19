#include <iostream>
#include "flashck/wrapper/cpp/norm/layer_norm.h"

int main() {
    std::cout << "Starting LayerNorm test..." << std::endl;
    
    try {
        const int m = 2, n = 4;
        const float epsilon = 1e-5f;
        
        // 创建测试数据
        float input[2*4] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float gamma[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        float beta[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        
        std::cout << "About to call layer_norm_fwd..." << std::endl;
        
        // 这里应该触发LayerNormOp的构造函数
        float* output = flashck::layer_norm_fwd(input, gamma, beta, m, n, epsilon);
        
        std::cout << "layer_norm_fwd completed" << std::endl;
        
        if (output) {
            std::cout << "Output received" << std::endl;
        } else {
            std::cout << "Output is null" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }
    
    std::cout << "Test completed" << std::endl;
    return 0;
}
