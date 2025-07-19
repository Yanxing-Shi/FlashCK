#pragma once

#include <map>
#include <string>
#include <variant>

#include "gflags/gflags.h"

namespace flashck {

// ==============================================================================
// Flag Declaration Macros (for headers)
// ==============================================================================

/** @brief Declare boolean flag (header) */
#define FC_DECLARE_bool(name) DECLARE_bool(name)

/** @brief Declare 32-bit int flag (header) */
#define FC_DECLARE_int32(name) DECLARE_int32(name)

/** @brief Declare 64-bit int flag (header) */
#define FC_DECLARE_int64(name) DECLARE_int64(name)

/** @brief Declare unsigned 64-bit int flag (header) */
#define FC_DECLARE_uint64(name) DECLARE_uint64(name)

/** @brief Declare double flag (header) */
#define FC_DECLARE_double(name) DECLARE_double(name)

/** @brief Declare string flag (header) */
#define FC_DECLARE_string(name) DECLARE_string(name)

// ==============================================================================
// Flag Definition Macros (for source files)
// ==============================================================================

/** @brief Define bool flag with default value
 *  @param name Flag name
 *  @param value Default value
 *  @param txt Help text
 */
#define FC_DEFINE_bool(name, value, txt) DEFINE_bool(name, value, txt)

/** @brief Define 32-bit int flag
 *  @param name Flag name
 *  @param val Default value
 *  @param txt Help text
 */
#define FC_DEFINE_int32(name, val, txt) DEFINE_int32(name, val, txt)

/** @brief Define unsigned 32-bit int flag
 *  @param name Flag name
 *  @param val Default value
 *  @param txt Help text
 */
#define FC_DEFINE_uint32(name, val, txt) DEFINE_uint32(name, val, txt)

/** @brief Define 64-bit int flag
 *  @param name Flag name
 *  @param val Default value
 *  @param txt Help text
 */
#define FC_DEFINE_int64(name, val, txt) DEFINE_int64(name, val, txt)

/** @brief Define unsigned 64-bit int flag
 *  @param name Flag name
 *  @param val Default value
 *  @param txt Help text
 */
#define FC_DEFINE_uint64(name, val, txt) DEFINE_uint64(name, val, txt)

/** @brief Define double flag
 *  @param name Flag name
 *  @param val Default value
 *  @param txt Help text
 */
#define FC_DEFINE_double(name, val, txt) DEFINE_double(name, val, txt)

/** @brief Define string flag
 *  @param name Flag name
 *  @param val Default value
 *  @param txt Help text
 */
#define FC_DEFINE_string(name, val, txt) DEFINE_string(name, val, txt)

// ==============================================================================
// Flag Registry and Management
// ==============================================================================

/**
 * @brief Information about a registered flag
 */
struct FlagInfo {
    /// Type-erased value storage supporting multiple flag types
    using ValueType = std::variant<bool, int32_t, int64_t, uint64_t, double, std::string>;

    std::string   name;          /**< Flag name (e.g. "verbose") */
    mutable void* value_ptr;     /**< Type-erased pointer to flag variable (mutable for const access) */
    ValueType     default_value; /**< Default value preserving original type information */
    std::string   doc;           /**< Help text describing flag purpose */
    bool          is_writable;   /**< Flag modification permission (false for internal flags) */
};

using ExportedFlagInfoMap = std::map<std::string, FlagInfo>;

/**
 * @brief Get mutable reference to the global flag registry
 * @return Mutable reference to flag registry
 */
inline ExportedFlagInfoMap* GetMutableExportedFlagInfoMap()
{
    static ExportedFlagInfoMap g_exported_flag_info_map;
    return &g_exported_flag_info_map;
}

/**
 * @brief Get const reference to the global flag registry
 * @return Const reference to flag registry
 */
inline const ExportedFlagInfoMap& GetExportedFlagInfoMap()
{
    return *GetMutableExportedFlagInfoMap();
}

/**
 * @brief Set flag value by name
 * @param name Flag name
 * @param value New value as string
 * @return true if successful, false otherwise
 */
bool SetFlagValue(const char* name, const char* value);

/**
 * @brief Check if flag exists
 * @param name Flag name
 * @return true if flag exists, false otherwise
 */
bool FindFlag(const char* name);

/**
 * @brief Get current flag value as string
 * @param name Flag name
 * @param value Output parameter for the flag value
 * @return true if successful, false if flag doesn't exist
 */
bool GetFlagValue(const char* name, std::string& value);

/**
 * @brief Print all available flags with their current values and descriptions
 * @param writable_only If true, only show writable flags
 */
void PrintAllFlags(bool writable_only = false);

/**
 * @brief Print detailed information about a specific flag
 * @param name Flag name
 * @return true if flag exists and was printed, false otherwise
 */
bool PrintFlagInfo(const char* name);

/**
 * @brief Get all flag information as a map
 * @param writable_only If true, only return writable flags
 * @return Map of flag names to FlagInfo structures
 */
std::map<std::string, FlagInfo> GetAllFlagInfo(bool writable_only = false);

/**
 * @brief Set flag value with type checking
 * @param name Flag name
 * @param value New value
 * @return true if successful, false otherwise
 */
template<typename T>
bool SetTypedFlagValue(const char* name, const T& value);

/**
 * @brief Get flag value with type checking
 * @param name Flag name
 * @param value Output parameter for the flag value
 * @return true if successful, false if flag doesn't exist or type mismatch
 */
template<typename T>
bool GetTypedFlagValue(const char* name, T& value);

// ==============================================================================
// Template Function Implementations
// ==============================================================================

template<typename T>
bool SetTypedFlagValue(const char* name, const T& value)
{
    const auto& flag_map = GetExportedFlagInfoMap();
    auto        it       = flag_map.find(name);
    if (it == flag_map.end()) {
        return false;
    }

    const auto& flag_info = it->second;
    if (!flag_info.is_writable) {
        return false;
    }

    // Convert value to string and use gflags to set it
    std::string str_value;
    if constexpr (std::is_same_v<T, bool>) {
        str_value = value ? "true" : "false";
    }
    else if constexpr (std::is_arithmetic_v<T>) {
        str_value = std::to_string(value);
    }
    else if constexpr (std::is_same_v<T, std::string>) {
        str_value = value;
    }
    else if constexpr (std::is_convertible_v<T, std::string>) {
        str_value = static_cast<std::string>(value);
    }
    else {
        return false;  // Unsupported type
    }

    return SetFlagValue(name, str_value.c_str());
}

template<typename T>
bool GetTypedFlagValue(const char* name, T& value)
{
    std::string str_value;
    if (!GetFlagValue(name, str_value)) {
        return false;
    }

    const auto& flag_map = GetExportedFlagInfoMap();
    auto        it       = flag_map.find(name);
    if (it == flag_map.end()) {
        return false;
    }

    const auto& flag_info = it->second;

    // Type-safe conversion based on the flag's actual type
    try {
        if constexpr (std::is_same_v<T, bool>) {
            if (std::holds_alternative<bool>(flag_info.default_value)) {
                value = (str_value == "true" || str_value == "1");
                return true;
            }
        }
        else if constexpr (std::is_same_v<T, int32_t>) {
            if (std::holds_alternative<int32_t>(flag_info.default_value)) {
                value = std::stoi(str_value);
                return true;
            }
        }
        else if constexpr (std::is_same_v<T, int64_t>) {
            if (std::holds_alternative<int64_t>(flag_info.default_value)) {
                value = std::stoll(str_value);
                return true;
            }
        }
        else if constexpr (std::is_same_v<T, uint64_t>) {
            if (std::holds_alternative<uint64_t>(flag_info.default_value)) {
                value = std::stoull(str_value);
                return true;
            }
        }
        else if constexpr (std::is_same_v<T, double>) {
            if (std::holds_alternative<double>(flag_info.default_value)) {
                value = std::stod(str_value);
                return true;
            }
        }
        else if constexpr (std::is_same_v<T, std::string>) {
            if (std::holds_alternative<std::string>(flag_info.default_value)) {
                value = str_value;
                return true;
            }
        }
    }
    catch (const std::exception&) {
        return false;
    }

    return false;  // Type mismatch
}

// ==============================================================================
// Exported Flag Definition Macros
// ==============================================================================

/**
 * @brief Internal macro for defining exported flags with registry
 * @param __name Flag name
 * @param __is_writable Whether flag can be modified at runtime
 * @param __cpp_type C++ type of the flag
 * @param __gflag_type gflags type identifier
 * @param __default_value Default value
 * @param __doc Help documentation
 */
#define __FC_DEFINE_EXPORTED_FLAG(__name, __is_writable, __cpp_type, __gflag_type, __default_value, __doc)             \
    FC_DEFINE_##__gflag_type(__name, __default_value, __doc);                                                          \
    namespace {                                                                                                        \
    struct __FCRegisterFlag_##__name {                                                                                 \
        __FCRegisterFlag_##__name()                                                                                    \
        {                                                                                                              \
            /* Type validation for supported flag types */                                                             \
            using FlagDeclaredType = std::remove_reference_t<decltype(FLAGS_##__name)>;                                \
            static_assert(std::is_same_v<FlagDeclaredType, std::string> || std::is_arithmetic_v<FlagDeclaredType>,     \
                          "FLAGS should be std::string or arithmetic type");                                           \
            /* Register to global flag registry */                                                                     \
            auto* registry     = ::flashck::GetMutableExportedFlagInfoMap();                                           \
            auto& info         = (*registry)[#__name];                                                                 \
            info.name          = #__name;                                                                              \
            info.value_ptr     = &(FLAGS_##__name);                                                                    \
            info.default_value = static_cast<__cpp_type>(__default_value);                                             \
            info.doc           = __doc;                                                                                \
            info.is_writable   = __is_writable;                                                                        \
        }                                                                                                              \
    };                                                                                                                 \
    /* Static instance ensures registration during startup */                                                          \
    [[maybe_unused]] static __FCRegisterFlag_##__name __FCRegisterFlagInstance_##__name;                               \
    }

/**
 * @brief Force flag registration to prevent optimization
 * @param __name Flag name
 */
#define FC_FORCE_FLAG_REGISTRATION(__name)                                                                             \
    do {                                                                                                               \
        [[maybe_unused]] volatile void* __force_link = &FLAGS_##__name;                                                \
    } while (0)

// ==============================================================================
// Public Exported Flag Definition Macros
// ==============================================================================

/** @brief Define exported boolean flag (writable) */
#define FC_DEFINE_EXPORTED_bool(name, default_value, doc)                                                              \
    __FC_DEFINE_EXPORTED_FLAG(name, true, bool, bool, default_value, doc)

/** @brief Define exported read-only boolean flag */
#define FC_DEFINE_EXPORTED_READONLY_bool(name, default_value, doc)                                                     \
    __FC_DEFINE_EXPORTED_FLAG(name, false, bool, bool, default_value, doc)

/** @brief Define exported 32-bit int flag (writable) */
#define FC_DEFINE_EXPORTED_int32(name, default_value, doc)                                                             \
    __FC_DEFINE_EXPORTED_FLAG(name, true, int32_t, int32, default_value, doc)

/** @brief Define exported 64-bit int flag (writable) */
#define FC_DEFINE_EXPORTED_int64(name, default_value, doc)                                                             \
    __FC_DEFINE_EXPORTED_FLAG(name, true, int64_t, int64, default_value, doc)

/** @brief Define exported unsigned 64-bit int flag (writable) */
#define FC_DEFINE_EXPORTED_uint64(name, default_value, doc)                                                            \
    __FC_DEFINE_EXPORTED_FLAG(name, true, uint64_t, uint64, default_value, doc)

/** @brief Define exported double flag (writable) */
#define FC_DEFINE_EXPORTED_double(name, default_value, doc)                                                            \
    __FC_DEFINE_EXPORTED_FLAG(name, true, double, double, default_value, doc)

/** @brief Define exported string flag (writable) */
#define FC_DEFINE_EXPORTED_string(name, default_value, doc)                                                            \
    __FC_DEFINE_EXPORTED_FLAG(name, true, std::string, string, default_value, doc)

}  // namespace flashck