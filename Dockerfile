# Usage (to build flashck docker image):
# docker build --build-arg BASE_IMAGE="rocm/pytorch:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.5.1" -t flashck:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.5.1 .

ARG BASE_IMAGE="rocm/pytorch:rocm6.4.1_ubuntu22.04_py3.10_pytorch_release_2.5.1"

FROM ${BASE_IMAGE} as base
USER root

# gflags
WORKDIR /tmp
RUN git clone https://github.com/gflags/gflags.git -b v2.2.2 && \
    cd gflags && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DBUILD_SHARED_LIBS=ON \
          -DGFLAGS_NAMESPACE=google \
          -DCMAKE_CXX_FLAGS=-fPIC .. && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && rm -rf gflags

# glog
WORKDIR /tmp
RUN git clone https://github.com/google/glog.git -b v0.7.1 && \
    cd glog && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DBUILD_SHARED_LIBS=ON \
          -DWITH_GFLAGS=ON .. && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && rm -rf glog

# sqlite
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*

# Jinja2Cpp
WORKDIR /tmp
RUN git clone https://github.com/jinja2cpp/Jinja2Cpp.git -b 1.2.1 && \
    cd Jinja2Cpp && \
    mkdir build && cd build && \
    cmake -DJINJA2CPP_BUILD_SHARED=TRUE .. && \
    cmake --build . --config Release --target install && \
    cd /tmp && rm -rf Jinja2Cpp

CMD ["/bin/bash"]




