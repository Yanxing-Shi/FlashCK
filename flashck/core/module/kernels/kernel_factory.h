#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "flashck/core/module/kernels/kernel.h"

namespace flashck {

/// @brief Source type enumeration for kernel implementations
enum class SourceType {
    LEGACY = 0,  ///< Legacy kernel implementations
    TILE   = 1,  ///< Tile-based kernel implementations
};

/// @brief Convert source type to string representation
/// @param type Source type to convert
/// @return String representation
inline std::string SourceTypeToString(SourceType type)
{
    switch (type) {
        case SourceType::LEGACY:
            return "legacy";
        case SourceType::TILE:
            return "tile";
        default:
            return "unknown";
    }
}

/// @brief Data layout enumeration for kernel requirements
enum class DataLayout {
    ALL_LAYOUT = 0,  ///< Compatible with all layouts
};

/// @brief Convert data layout to string representation
/// @param layout Data layout to convert
/// @return String representation
inline std::string DataLayoutToString(DataLayout layout)
{
    switch (layout) {
        case DataLayout::ALL_LAYOUT:
            return "all_layout";
        default:
            return "unknown";
    }
}

// struct OpTypeCount {
//     OpTypeCount()
//     {
//         fp16_count_ = 0;
//         fp32_count_ = 0;
//     }

//     int fp16_count_ = 0;
//     int fp32_count_ = 0;
// };

/**
 * @brief Unique key for kernel identification and lookup
 *
 * Combines source type, data layout, and data type into a compact
 * hash-based key for efficient kernel lookup in the registry.
 */
class KernelKey {
public:
    /// @brief Default constructor
    KernelKey() = default;

    /// @brief Parameterized constructor
    /// @param source_type Source implementation type
    /// @param layout Data layout requirement
    /// @param dtype Data type requirement
    KernelKey(SourceType source_type, DataLayout layout, DataType dtype):
        source_type_(source_type), layout_(layout), dtype_(dtype)
    {
    }

    /// @brief Get source type
    /// @return Source type of the kernel
    SourceType GetSource() const
    {
        return source_type_;
    }

    /// @brief Get data layout
    /// @return Data layout requirement
    DataLayout GetLayout() const
    {
        return layout_;
    }

    /// @brief Get data type
    /// @return Data type requirement
    DataType GetDtype() const
    {
        return dtype_;
    }

    /// @brief Set data layout
    /// @param layout New data layout
    void SetLayout(const DataLayout& layout)
    {
        layout_ = layout;
    }

    /// @brief Set data type
    /// @param dtype New data type
    void SetDtype(const DataType& dtype)
    {
        dtype_ = dtype;
    }

    /// @brief Hash functor for use in unordered containers
    struct Hash {
        uint32_t operator()(const KernelKey& key) const;
    };

    /// @brief Get hash value for this key
    /// @return 32-bit hash value
    uint32_t HashValue() const
    {
        return Hash()(*this);
    }

    /// @brief Convert kernel key to string representation
    /// @return String representation of the kernel key
    std::string ToString() const
    {
        return "(" + SourceTypeToString(source_type_) + ", " + DataLayoutToString(layout_) + ", "
               + DataTypeToString(dtype_) + ")";
    }

    /// @brief Equality comparison operator
    /// @param key Other kernel key to compare
    /// @return true if keys are equal
    bool operator==(const KernelKey& key) const
    {
        return HashValue() == key.HashValue();
    }

    /// @brief Inequality comparison operator
    /// @param key Other kernel key to compare
    /// @return true if keys are not equal
    bool operator!=(const KernelKey& key) const
    {
        return HashValue() != key.HashValue();
    }

    /// @brief Less than comparison operator for ordering
    /// @param key Other kernel key to compare
    /// @return true if this key is less than other
    bool operator<(const KernelKey& key) const
    {
        return HashValue() < key.HashValue();
    }

private:
    SourceType source_type_;  ///< Source implementation type
    DataLayout layout_;       ///< Data layout requirement
    DataType   dtype_;        ///< Data type requirement

    /// @brief Hash configuration constants (total must be <= 32 bits)
    constexpr static int kSourceBitLength = 8;  ///< Bits for source type
    constexpr static int kLayoutBitLength = 4;  ///< Bits for layout
    constexpr static int kDtypeBitLength  = 8;  ///< Bits for data type
};

/// @brief Type alias for kernel factory function pointer
typedef Kernel* (*KernelPtr)();

/// @brief Type alias for kernel key to factory function mapping
using KernelKeyMap = std::unordered_map<KernelKey, KernelPtr, KernelKey::Hash>;

/// @brief Type alias for kernel name to key map mapping
using KernelNameMap = std::unordered_map<std::string, KernelKeyMap>;

/**
 * @brief Singleton factory for kernel creation and management
 *
 * Provides centralized registration and creation of kernel implementations.
 * Supports hierarchical lookup by kernel name and key (source, layout, dtype).
 */
class KernelFactory {
public:
    /// @brief Get singleton instance
    /// @return Reference to the singleton factory
    static KernelFactory& Instance();

    /// @brief Get the complete kernel registry map
    /// @return Reference to kernel_name -> (kernel_key -> factory_function) map
    KernelNameMap& GetKernelsMap()
    {
        return kernel_map_;
    }

    /// @brief Check if a specific kernel is registered
    /// @param kernel_name Name of the kernel
    /// @param kernel_key Key specifying kernel variant
    /// @return true if kernel is registered
    bool HasRegisteredKernel(const std::string& kernel_name, const KernelKey& kernel_key);

    /// @brief Create kernel instance by name and key
    /// @param kernel_name Name of the kernel to create
    /// @param kernel_key Key specifying kernel variant
    /// @return Shared pointer to created kernel instance
    /// @throws NotFound if kernel is not registered
    std::shared_ptr<Kernel> SelectKernel(const std::string& kernel_name, const KernelKey& kernel_key);

    /// @brief Get all registered variants for a kernel name
    /// @param kernel_name Name of the kernel
    /// @return Map of kernel keys to factory functions
    KernelKeyMap SelectKernelMap(const std::string& kernel_name);

private:
    KernelNameMap kernel_map_;  ///< Main registry storage
};

/// @brief Stream output operator for kernel keys
/// @param os Output stream
/// @param kernel_key Kernel key to output
/// @return Reference to output stream
std::ostream& operator<<(std::ostream& os, const KernelKey& kernel_key);

}  // namespace flashck