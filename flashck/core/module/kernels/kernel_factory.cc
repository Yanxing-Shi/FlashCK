#include "flashck/core/module/kernels/kernel_factory.h"

#include <string>
#include <unordered_set>

#include "flashck/core/utils/enforce.h"

namespace flashck {

// Hash implementation for KernelKey
uint32_t KernelKey::Hash::operator()(const KernelKey& kernel_key) const
{
    // Bit layout: |31-20 (reserved)|19-12 (dtype)|11-8 (layout)|7-0 (source)|
    uint32_t hash_value = 0;
    hash_value |= static_cast<uint8_t>(kernel_key.GetSource());
    hash_value |= (static_cast<uint8_t>(kernel_key.GetLayout()) << 8);
    hash_value |= (static_cast<uint32_t>(kernel_key.GetDtype()) << 12);
    return hash_value;
}

// Kernel factory singleton
KernelFactory& KernelFactory::Instance()
{
    static KernelFactory g_kernel_factory;
    return g_kernel_factory;
}

// Check if kernel is registered
bool KernelFactory::HasRegisteredKernel(const std::string& kernel_name, const KernelKey& kernel_key)
{
    auto kernel_map_iter = kernel_map_.find(kernel_name);
    if (kernel_map_iter == kernel_map_.end()) {
        return false;
    }

    auto kernel_key_iter = kernel_map_iter->second.find(kernel_key);
    return kernel_key_iter != kernel_map_iter->second.end();
}

// Get kernel instance by name and key
std::shared_ptr<Kernel> KernelFactory::SelectKernel(const std::string& kernel_name, const KernelKey& kernel_key)
{
    auto iter = kernel_map_.find(kernel_name);
    FC_ENFORCE_NE(iter, kernel_map_.end(), Unavailable("Kernel {} is not registered", kernel_name));

    auto kernel_iter = iter->second.find(kernel_key);
    FC_ENFORCE_NE(kernel_iter,
                  iter->second.end(),
                  Unavailable("Kernel {} with key {} is not registered", kernel_name, kernel_key.ToString()));

    auto kernel_ptr = kernel_iter->second;
    return std::shared_ptr<Kernel>(kernel_ptr());
}

// Get kernel key map according to kernel name
KernelKeyMap KernelFactory::SelectKernelMap(const std::string& kernel_name)
{
    auto iter = kernel_map_.find(kernel_name);
    if (iter == kernel_map_.end()) {
        return KernelKeyMap{};  // Return empty map if not found
    }
    return iter->second;
}

// Stream output operator for kernel key
std::ostream& operator<<(std::ostream& os, const KernelKey& kernel_key)
{
    os << "(" << SourceTypeToString(kernel_key.GetSource()) << ", " << DataLayoutToString(kernel_key.GetLayout())
       << ", " << DataTypeToString(kernel_key.GetDtype()) << ")";
    return os;
}

}  // namespace flashck