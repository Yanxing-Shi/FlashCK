#!/bin/bash
set -e

# Check and install gflags
if [ ! -f /usr/local/lib/libgflags.so ]; then
  echo "Installing gflags..."
  cd 3rdparty/gflags
  mkdir -p build && cd build
  cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON -DGFLAGS_NAMESPACE=google -DCMAKE_CXX_FLAGS=-fPIC ..
  make -j$(nproc)
  sudo make install
  cd ../../..
else
  echo "gflags already installed, skipping."
fi

# Check and install glog
if [ ! -f /usr/local/lib/libglog.so ]; then
  echo "Installing glog..."
  cd 3rdparty/glog
  mkdir -p build && cd build
  cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON -DWITH_GFLAGS=ON ..
  make -j$(nproc)
  sudo make install
  cd ../../..
else
  echo "glog already installed, skipping."
fi

# Check and install Jinja2Cpp
if [ ! -f /usr/local/lib/libjinja2cpp.so ]; then
  echo "Installing Jinja2Cpp..."
  cd 3rdparty/Jinja2Cpp
  mkdir -p build && cd build
  cmake -DJINJA2CPP_BUILD_SHARED=TRUE ..
  make -j$(nproc)
  sudo make install
  cd ../../..
else
  echo "Jinja2Cpp already installed, skipping."
fi

# Check and install googletest
if [ ! -f /usr/local/lib/libgtest.so ]; then
  echo "Installing googletest..."
  cd 3rdparty/googletest
  mkdir -p build && cd build
  cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
  make -j$(nproc)
  sudo make install
  cd ../../..
else
  echo "googletest already installed, skipping."
fi

# Check and install nlohmann_json (header only, usually no .so)
if [ ! -d /usr/local/include/nlohmann ]; then
  echo "Installing nlohmann_json (header only)..."
  cd 3rdparty/json
  mkdir -p build && cd build
  cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local ..
  make -j$(nproc)
  sudo make install
  cd ../../..
else
  echo "nlohmann_json already installed, skipping."
fi

# Check and install nanobind
if [ ! -f /usr/local/lib/libnanobind.so ]; then
  echo "Installing nanobind..."
  cd 3rdparty/nanobind
  mkdir -p build && cd build
  cmake -DNB_TEST_SHARED_BUILD=ON -DCMAKE_INSTALL_PREFIX=/usr/local ..
  make -j$(nproc)
  sudo make install
  cd ../../..
else
  echo "nanobind already installed, skipping."
fi

# Check and install cxxopts (header only, usually no .so)
if [ ! -d /usr/local/include/cxxopts ]; then
  echo "Installing cxxopts (header only)..."
  cd 3rdparty/cxxopts
  mkdir -p build && cd build
  cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
  make -j$(nproc)
  sudo make install
  cd ../../..
else
  echo "cxxopts already installed, skipping."
fi

echo "All 3rdparty libraries processed."