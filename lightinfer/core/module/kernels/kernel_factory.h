#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "lightinfer/core/utils/dtype.h"
#include "lightinfer/core/utils/layout.h"
#include "lightinfer/core/utils/source_type.h"

#include "lightinfer/core/module/kernels/kernel.h"

namespace lightinfer {

// struct OpTypeCount {
//     OpTypeCount()
//     {
//         fp16_count_ = 0;
//         fp32_count_ = 0;
//     }

//     int fp16_count_ = 0;
//     int fp32_count_ = 0;
// };

// kernel key base class
class KernelKey {
public:
    // class Constructor
    KernelKey() = default;
    KernelKey(SourceType source_type, DataLayout layout, DataType dtype):
        source_type_(source_type), layout_(layout), dtype_(dtype)
    {
    }
    explicit KernelKey(const SourceType& source_type):
        source_type_(source_type), layout_(DataLayout::ALL_LAYOUT), dtype_(DataType::ALL_DTYPE)
    {
    }

    KernelKey(const SourceType& source_type, const DataLayout& layout):
        source_type_(source_type), layout_(layout), dtype_(DataType::ALL_DTYPE)
    {
    }

    // get property
    SourceType GetSource() const
    {
        return source_type_;
    }
    DataLayout GetLayout() const
    {
        return layout_;
    }
    DataType GetDtype() const
    {
        return dtype_;
    }

    // set property
    void SetLayout(const DataLayout& layout)
    {
        layout_ = layout;
    }
    void SetDtype(const DataType& dtype)
    {
        dtype_ = dtype;
    }

    // hash function
    struct Hash {
        uint32_t operator()(const KernelKey& key) const;
    };

    uint32_t HashValue() const
    {
        return Hash()(*this);
    }

    bool operator==(const KernelKey& key) const
    {
        return HashValue() == key.HashValue();
    }

    bool operator!=(const KernelKey& key) const
    {
        return HashValue() != key.HashValue();
    }

    bool operator<(const KernelKey& key) const
    {
        return HashValue() < key.HashValue();
    }

private:
    SourceType source_type_;
    DataLayout layout_;
    DataType   dtype_;

    // hash settings, In total should be smaller than 32.
    constexpr static int kSourceBitLength = 8;
    constexpr static int kLayoutBitLength = 4;
    constexpr static int kDtypeBitLength  = 8;
};

typedef Kernel* (*KernelPtr)();
using KernelKeyMap  = std::unordered_map<KernelKey, KernelPtr, KernelKey::Hash>;
using KernelNameMap = std::unordered_map<std::string, KernelKeyMap>;

class KernelFactory {
public:
    static KernelFactory& Instance();

    // Get kernel map
    // kernel_name::kernel_key::kernel
    KernelNameMap& GetKernelsMap()
    {
        return kernel_map_;
    }

    // detemine if kernel is registered
    bool HasRegisteredKernel(const std::string& kernel_name, const KernelKey& kernel_key);

    // Get kernel instance according to kernel name and kernel key
    std::shared_ptr<Kernel> SelectKernel(const std::string& kernel_name, const KernelKey& kernel_key);

    // Get kernel key map according to kernel name
    KernelKeyMap SelectKernelMap(const std::string& kernel_name);

private:
    KernelNameMap kernel_map_;
};

std::ostream& operator<<(std::ostream& os, const KernelKey& kernel_key);

}  // namespace lightinfer