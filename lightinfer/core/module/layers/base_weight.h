
#pragma once

#include <string>
#include <vector>

namespace lightinfer {

template<typename T>
struct Weight {

public:
    std::string         name_;
    std::vector<size_t> shape_;
    size_t              size_ = 0;
    T*                  ptr_  = nullptr;

    Weight() {}
    Weight(const std::string name, const std::vector<size_t> shape, T* ptr): name_(name), shape_(shape), ptr_(ptr)
    {
        size_ = 1;
        for (uint i = 0; i < shape_.size(); i++) {
            size_ *= shape_[i];
        }
    }

    ~Weight()
    {
        size_ = 0;
        ptr_  = nullptr;
    }
};

}  // namespace lightinfer