#!/bin/bash

# FlashCK C++ Wrapper Build and Usage Example

echo "FlashCK C++ Wrapper Build Script"
echo "================================="

# Check if we're in the correct directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Please run this script from the FlashCK root directory"
    exit 1
fi

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_TESTS=ON \
    -DBUILD_WRAPPER=ON

# Build the project
echo "Building FlashCK with C++ wrapper..."
make -j$(nproc)

# Run tests
echo "Running C++ wrapper tests..."
if [ -f "tests/test_layer_norm" ]; then
    echo "Running test_layer_norm..."
    ./tests/test_layer_norm
else
    echo "test_layer_norm not found, searching..."
    find . -name "test_layer_norm" -type f -executable -exec echo "Found: {}" \; -exec {} \;
fi

if [ -f "tests/test_layer_norm_dynamic" ]; then
    echo "Running test_layer_norm_dynamic..."
    ./tests/test_layer_norm_dynamic
else
    echo "test_layer_norm_dynamic not found, searching..."
    find . -name "test_layer_norm_dynamic" -type f -executable -exec echo "Found: {}" \; -exec {} \;
fi

if [ -f "tests/benchmark_layer_norm" ]; then
    echo "Running benchmark_layer_norm..."
    ./tests/benchmark_layer_norm
else
    echo "benchmark_layer_norm not found, searching..."
    find . -name "benchmark_layer_norm" -type f -executable -exec echo "Found: {}" \; -exec {} \;
fi

# Run examples
echo "Running C++ wrapper examples..."
if [ -f "examples/layer_norm_example" ]; then
    echo "Running layer_norm_example..."
    ./examples/layer_norm_example
else
    echo "layer_norm_example not found, searching..."
    find . -name "layer_norm_example" -type f -executable -exec echo "Found: {}" \; -exec {} \;
fi

if [ -f "examples/layer_norm_dynamic_example" ]; then
    echo "Running layer_norm_dynamic_example..."
    ./examples/layer_norm_dynamic_example
else
    echo "layer_norm_dynamic_example not found, searching..."
    find . -name "layer_norm_dynamic_example" -type f -executable -exec echo "Found: {}" \; -exec {} \;
fi

# Run CTest if available
echo "Running CTest..."
if command -v ctest &> /dev/null; then
    ctest --output-on-failure
else
    echo "CTest not available, skipping..."
fi

echo "Build and test completed!"
echo ""
echo "To use the FlashCK C++ wrapper in your project:"
echo "1. Include the header: #include <flashck/wrapper/c++/flashck_wrapper.h>"
echo "2. Link against flashck_cpp_wrapper in your CMakeLists.txt:"
echo "   target_link_libraries(your_target PRIVATE flashck_cpp_wrapper)"
echo "3. Use C++20 standard: target_compile_features(your_target PRIVATE cxx_std_20)"
echo "4. Add include directory: target_include_directories(your_target PRIVATE path/to/flashck/wrapper/c++)"
