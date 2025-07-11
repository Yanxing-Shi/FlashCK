#include "flashck/core/module/kernels/kernel_factory.h"

#include <string>
#include <unordered_set>

#include "flashck/core/utils/enforce.h"

namespace flashck {

// hash kernel key
uint32_t KernelKey::Hash::operator()(const KernelKey& kernel_key) const
{
    // |----31-20------|---19-12---|---11-8----|---7-0---|
    // | For extension | DataType | DataLayout | source  |

    uint32_t hash_value = 0;
    hash_value |= (static_cast<uint8_t>(kernel_key.GetSource()) << KernelKey::kSourceBitLength);
    hash_value |= (static_cast<uint8_t>(kernel_key.GetLayout()) << KernelKey::kLayoutBitLength);
    hash_value |= (static_cast<uint32_t>(kernel_key.GetDtype()) << KernelKey::kDtypeBitLength);
    return hash_value;
}

/*------------------------------------------kernel factory-----------------------------------------------*/
// Get kernel instance
KernelFactory& KernelFactory::Instance()
{
    static KernelFactory g_op_kernel_factory;
    return g_op_kernel_factory;
}

// check if kernel is registered, return true
bool KernelFactory::HasRegisteredKernel(const std::string& kernel_name, const KernelKey& kernel_key)
{
    auto kernel_map_iter = kernel_map_.find(kernel_name);
    if (kernel_map_iter == kernel_map_.end()) {
        FC_THROW(NotFound("The kernel {} is not registered", kernel_name));
        return false;
    }
    auto kernel_key_iter = kernel_map_iter->second.find(kernel_key);
    if (kernel_key_iter == kernel_map_iter->second.end()) {
        FC_THROW(NotFound("The kernel {} is not registered", kernel_name));
        return false;
    }
    return true;
}

// Get kernel instance according to kernel name and kernel key
std::shared_ptr<Kernel> KernelFactory::SelectKernel(const std::string& kernel_name, const KernelKey& kernel_key)
{
    auto iter = kernel_map_.find(kernel_name);
    if (iter == kernel_map_.end()) {
        FC_THROW(NotFound("The kernel {} is not registered.", kernel_name));
    }

    auto kernel_iter = iter->second.find(kernel_key);
    if (kernel_iter == iter->second.end()) {
        FC_THROW(NotFound("The kernel {} is not registered.", kernel_name));
    }

    auto kernel_ptr = kernel_map_[kernel_name][kernel_key];
    return std::shared_ptr<Kernel>(kernel_ptr());
}

// Get kernel key map according to kernel name
KernelKeyMap KernelFactory::SelectKernelMap(const std::string& kernel_name)
{
    auto iter = kernel_map_.find(kernel_name);
    if (iter == kernel_map_.end()) {
        return KernelKeyMap();
    }
    return iter->second;
}

// output stream for kernel key
std::ostream& operator<<(std::ostream& os, const KernelKey& kernel_key)
{
    os << "(" << SourceTypeToString(kernel_key.GetSource()) << ", " << DataLayoutToString(kernel_key.GetLayout())
       << ", " << DataTypeToString(kernel_key.GetDtype()) << ")";
    return os;
}

}  // namespace flashck